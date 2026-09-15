from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_runs_metadata_only_post_bnm_probe(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "GetEnv", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent", "RunReadOnlyAbiProbe",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-post-bnm-abi", "nativeStage=post-bnm-read-only-abi",
            "probeComplete=1", "gameHooksInstalled=0",
            "abi.SFB.class=", "abi.Mobile.CanvasScaler.SetScaleFactor1=",
            "abi.Touch.EventSystem.RaycastAll=", "abi.FPS.Application.setTargetFrameRate1=",
            "abi.Event.scrCamera.SetCustomFrameRateBoolInt=",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "BasicHook", "InstallAllHooks", "InstallSfbHooks", "InstallMobileHooks",
            "V240SettingsOverlay", "V240EventCompat", "V240TouchAssist", "FileSelector",
            "FindClass", "CallStatic", "CallObject", "NewGlobalRef", "pthread_create",
            ".Call(", ".Set(", "CreateNewObject",
        ):
            self.assertNotIn(forbidden, source)

    def test_probe_only_resolves_metadata_after_bnm_callback(self):
        source = LOADER.read_text(encoding="utf-8")
        callback = source.index("Loading::AddOnLoadedEvent")
        probe_call = source.index("RunReadOnlyAbiProbe();", callback)
        self.assertGreater(probe_call, callback)
        self.assertIn('Class browser("SFB", "StandaloneFileBrowser")', source)
        self.assertIn('Class eventSystem("UnityEngine.EventSystems", "EventSystem")', source)
        self.assertIn('Class scrCamera("", "scrCamera")', source)
        self.assertIn('GetMethod("SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()})', source)

    def test_cache_native_build_pins_bnm_but_excludes_feature_runtime(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.cpp", build)
        self.assertIn("HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git", build)
        self.assertIn("74bcc7a0d8c8be1267504e21e28a35e199b5d4eb", build)
        self.assertIn("UNITY_VER 213", build)
        self.assertIn("UNITY_PATCH_VER 10", build)
        self.assertIn("BNM/src/Loading.cpp", build)
        self.assertIn("nativeProbe=cache-post-bnm-abi", build)
        self.assertIn("post-bnm-read-only-abi", build)
        self.assertNotIn("build-v240-fixed-native.sh", build)
        self.assertNotIn("V240Fix.cpp", build)
        self.assertNotIn("V240TouchAssist.cpp", build)
        self.assertNotIn("V240EventCompat.cpp", build)

    def test_channel_packages_read_only_abi_probe_not_full_hook_runtime(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("Build read-only post-BNM ABI cache probe", workflow)

    def test_rollout_is_boolean_and_enabled_rollout_stays_feature_inert(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertIn("Recovery channel v3 deliberately installs no gameplay/event/UI hooks.", dynamic)
            self.assertIn("metadata-only ABI discovery", dynamic)
            self.assertNotIn("V240EventCompat", dynamic)
            self.assertNotIn("V240SettingsOverlay", dynamic)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
