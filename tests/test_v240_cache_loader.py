from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.c"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_is_inert_on_load(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn("JNI_OnLoad", source)
        self.assertIn("GetEnv", source)
        for forbidden in (
            "universe.h", "BNM", "Loading::", "BasicHook", "Dobby", "dlopen",
            "pthread_create", "FindClass", "CallStatic", "CallObject", "NewGlobalRef",
            "V240SettingsOverlay", "V240EventCompat", "InstallAllHooks",
        ):
            self.assertNotIn(forbidden, source)

    def test_cache_native_build_has_no_feature_runtime_dependency(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.c", build)
        self.assertIn('NDK_BUILD="${NDK}/ndk-build"', build)
        self.assertIn("APP_ABI := arm64-v8a", build)
        self.assertIn("--no-undefined", build)
        self.assertIn("JNI_OnLoad", build)
        self.assertNotIn("build-v240-fixed-native.sh", build)
        self.assertNotIn("HitMargin", build)

    def test_channel_packages_inert_loader_not_full_hook_runtime(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)

    def test_rollout_is_boolean_and_enabled_rollout_is_recovery_only(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertIn("Deliberately no event/diagnostic activation in recovery channel v1.", dynamic)
            self.assertNotIn("V240EventCompat", dynamic)
            self.assertNotIn("V240SettingsOverlay", dynamic)
            self.assertNotIn("System.load", dynamic)


if __name__ == "__main__":
    unittest.main()
