from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.c"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"


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
        self.assertIn("aarch64-linux-android${API}-clang", build)
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

    def test_rollout_remains_off_until_exact_device_baseline_is_proven(self):
        self.assertEqual(ROLLOUT.read_text(encoding="utf-8").strip(), "false")


if __name__ == "__main__":
    unittest.main()
