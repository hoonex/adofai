from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_installs_only_runtime_proven_sfb_filter_hook(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "GetEnv", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent", "RunProbeAndInstallNarrowFix",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-post-bnm-narrow-fix", "nativeStage=post-bnm-narrow-sfb-fix",
            "HookOpenFilePanelFilters", "FileSelector", "BasicHook",
            "sfbOpenFiltersHookInstalled=", "sfbFilterMemoryRead=0",
            "sfbFilterPolicy=broad-safe-fallback",
            "abi.SFB.OpenFilePanel.filtersExact=", "abi.SFB.ExtensionFilter.Name=",
            "abi.SFB.ExtensionFilter.Extensions=", "abi.Settings.PauseMenu.ShowSettingsMenu0=",
        ):
            self.assertIn(required, source)
        self.assertEqual(source.count("BasicHook("), 1)
        for forbidden in (
            "InstallAllHooks", "InstallSfbHooks", "InstallMobileHooks",
            "V240SettingsOverlay", "V240EventCompat", "V240TouchAssist",
            "HookOpenString", "HookSaveString", "HookSaveFilters", "HookFolder",
            "HookSetTargetFrameRate", "HookSetVSyncCount", "HookInsideUI",
            ".Call(", ".Set(", "CreateNewObject",
        ):
            self.assertNotIn(forbidden, source)

    def test_sfb_hook_is_exact_overload_and_fail_closed(self):
        source = LOADER.read_text(encoding="utf-8")
        exact = '"OpenFilePanel", {"title", "directory", "extensions", "multiselect"}'
        self.assertIn(exact, source)
        self.assertIn('Class extensionFilter("SFB", "ExtensionFilter")', source)
        self.assertIn('extensionFilter.GetField("Name").IsValid()', source)
        self.assertIn('extensionFilter.GetField("Extensions").IsValid()', source)
        self.assertIn("selectorReady && openFiltersExact", source)
        self.assertIn("InstallExactOpenFiltersHook(browser, filterMetadataSurface)", source)
        self.assertIn("if (!browser || !filterMetadataSurface) return false;", source)
        self.assertIn("if (!method.IsValid()) return false;", source)

    def test_extension_filter_value_layout_is_never_dereferenced(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn("kSafeFallbackExtensions", source)
        self.assertIn('"adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg"', source)
        self.assertIn("RunOpenPicker(kSafeFallbackExtensions, multiselect)", source)
        self.assertIn("sfbFilterMemoryRead=0", source)
        self.assertIn("sfbFilterPolicy=broad-safe-fallback", source)
        for forbidden in (
            "struct ExtensionFilterValue",
            "reinterpret_cast<Array<ExtensionFilterValue>*>(",
            "m_Items[i].Extensions",
            "PickerExtensions(",
            "kMaxExtensionFilters",
            "kMaxExtensionsPerFilter",
        ):
            self.assertNotIn(forbidden, source)

    def test_picker_bridge_is_bounded_and_open_only(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn("kPickerPollCount = 18000", source)
        self.assertIn("kPickerPollMs = 50", source)
        self.assertIn('"selectFile", "(Ljava/lang/String;Z)V"', source)
        self.assertIn('"getFilePath", "()Ljava/lang/String;"', source)
        self.assertIn('"isDone", "Z"', source)
        self.assertNotIn('"saveAs"', source)
        self.assertNotIn('"selectFolder"', source)

    def test_other_abi_discovery_remains_read_only(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn('Class eventSystem("UnityEngine.EventSystems", "EventSystem")', source)
        self.assertIn('Class scrCamera("", "scrCamera")', source)
        self.assertIn('"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}).IsValid()', source)
        self.assertIn('Class pauseMenu("", "PauseMenu")', source)
        self.assertIn('pauseMenu.GetMethod("ShowSettingsMenu", 0).IsValid()', source)
        self.assertNotIn("HookShowSettings", source)

    def test_cache_native_build_pins_bnm_and_contracts_narrow_hook(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.cpp", build)
        self.assertIn("HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git", build)
        self.assertIn("74bcc7a0d8c8be1267504e21e28a35e199b5d4eb", build)
        self.assertIn("UNITY_VER 213", build)
        self.assertIn("UNITY_PATCH_VER 10", build)
        self.assertIn("BNM/src/Loading.cpp", build)
        self.assertIn("nativeProbe=cache-post-bnm-narrow-fix", build)
        self.assertIn("sfbOpenFiltersHookInstalled=", build)
        self.assertIn("assert s.count('BasicHook(') == 1", build)
        self.assertNotIn("build-v240-fixed-native.sh", build)
        self.assertNotIn("V240Fix.cpp", build)
        self.assertNotIn("V240TouchAssist.cpp", build)
        self.assertNotIn("V240EventCompat.cpp", build)

    def test_dynamic_entry_only_repositions_transitional_settings_button(self):
        dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
        self.assertIn('LEGACY_GEAR_TAG = "adofai-v240-settings-button"', dynamic)
        self.assertIn("scheduleLegacyGearRelocation", dynamic)
        self.assertIn("Gravity.CENTER_VERTICAL | Gravity.END", dynamic)
        self.assertIn("lp.width = dp(activity, 44)", dynamic)
        self.assertIn("transitional placement", dynamic)
        self.assertNotIn("V240EventCompat", dynamic)
        self.assertNotIn("System.load(", dynamic)

    def test_channel_packages_narrow_recovery_runtime(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("Build narrow SFB recovery cache runtime", workflow)

    def test_rollout_is_boolean_and_enabled_runtime_stays_narrow(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertIn("Recovery channel v5 activates only the exact SFB ExtensionFilter[] open overload", dynamic)
            self.assertIn("never reads the incoming managed", dynamic)
            self.assertIn("No event, FPS, timing or gameplay hook", dynamic)
            self.assertNotIn("V240EventCompat", dynamic)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
