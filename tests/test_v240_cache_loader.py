from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_is_bnm_only_and_has_read_only_report(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "GetEnv", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-bnm-loader-only", "gameHooksInstalled=0",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "BasicHook", "InstallAllHooks", "V240SettingsOverlay", "V240EventCompat",
            "V240TouchAssist", "FileSelector", "FindClass", "CallStatic", "CallObject",
            "NewGlobalRef", "pthread_create",
        ):
            self.assertNotIn(forbidden, source)

    def test_cache_native_build_pins_bnm_but_excludes_feature_runtime(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.cpp", build)
        self.assertIn("HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git", build)
        self.assertIn("74bcc7a0d8c8be1267504e21e28a35e199b5d4eb", build)
        self.assertIn("UNITY_VER 213", build)
        self.assertIn("UNITY_PATCH_VER 10", build)
        self.assertIn("BNM/src/Loading.cpp", build)
        self.assertIn("Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport", build)
        self.assertNotIn("build-v240-fixed-native.sh", build)
        self.assertNotIn("V240Fix.cpp", build)
        self.assertNotIn("V240TouchAssist.cpp", build)
        self.assertNotIn("V240EventCompat.cpp", build)

    def test_channel_packages_bnm_probe_not_full_hook_runtime(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("Build BNM loader-only arm64 cache probe", workflow)

    def test_rollout_is_boolean_and_enabled_rollout_stays_feature_inert(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertIn("Recovery channel v2 deliberately enables no gameplay/event/UI hooks.", dynamic)
            self.assertIn("BNM IL2CPP loading boundary", dynamic)
            self.assertNotIn("V240EventCompat", dynamic)
            self.assertNotIn("V240SettingsOverlay", dynamic)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
