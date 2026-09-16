from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_installs_only_exact_pass_through_canary(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "GetEnv", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent", "RunAbiProbeAndInstallCanary",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-post-bnm-sfb-pass-through-v1",
            "nativeStage=post-bnm-sfb-pass-through-canary",
            "abiProbeRevision=4",
            "sfbHookPolicy=bootstrap3-exact-pass-through-canary",
            "sfbCanaryAbiGuard=", "sfbCanaryOriginalCaptured=",
            "sfbOpenFiltersCanaryCalls=", "sfbFilterMemoryRead=0",
            "struct ExtensionFilterValue", "String* Name;",
            "Array<String*>* Extensions;",
            "using OpenFiltersFn = Array<String*>* (*)",
            "HookOpenFilePanelFilters", "g_oldOpenFilters",
            "BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters)",
            "return original(title, directory, filters, multiselect);",
        ):
            self.assertIn(required, source)
        self.assertEqual(source.count("BasicHook("), 1)
        for forbidden in (
            "InstallAllHooks", "InstallSfbHooks", "InstallMobileHooks",
            "V240SettingsOverlay", "V240EventCompat", "V240TouchAssist", "FileSelector",
            "FindClass", "CallStatic", "CallObject", "NewGlobalRef", "pthread_create",
            ".Call(", ".Set(", "CreateNewObject", "RunOpenPicker(",
            "m_Items[", "filters->",
        ):
            self.assertNotIn(forbidden, source)

    def test_canary_is_runtime_guarded_by_proven_v240_abi(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            'browser.GetMethod(\n            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"})',
            "openInfo->methodPointer != nullptr",
            "openFilters._isStatic",
            "openInfo->parameters_count == 4",
            "SameClass(returnClass, stringArrayClass)",
            "SameClass(param0Class, stringClass)",
            "SameClass(param1Class, stringClass)",
            "SameClass(param2Class, filterArrayClass)",
            "SameClass(param3Class, boolClass)",
            "TypeByRef(param2Type) == 0",
            "TypeValueType(param2Type) == 0",
            "filterType->valuetype",
            "sizeof(IL2CPP::Il2CppObject) + sizeof(ExtensionFilterValue)",
            "filterInstanceSize == expectedBoxedSize",
            "filterActualSize == expectedBoxedSize",
            "filterNameOffset == 0",
            "filterExtensionsOffset == static_cast<long long>(sizeof(void*))",
            "SameClass(filterNameType, stringClass)",
            "SameClass(filterExtensionsType, stringArrayClass)",
        ):
            self.assertIn(required, source)

    def test_canary_never_reads_filter_array_or_changes_result(self):
        source = LOADER.read_text(encoding="utf-8")
        hook_start = source.index("Array<String*>* HookOpenFilePanelFilters")
        hook_end = source.index("\n}\n", hook_start) + 2
        hook = source[hook_start:hook_end]
        self.assertIn("g_sfbOpenFiltersCanaryCalls.fetch_add", hook)
        self.assertIn("return original(title, directory, filters, multiselect);", hook)
        self.assertNotIn("m_Items", hook)
        self.assertNotIn("Extensions", hook)
        self.assertNotIn("Name", hook)
        self.assertNotIn("CreateMonoString", hook)

    def test_cache_native_build_pins_bnm_and_allows_only_canary_hook(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.cpp", build)
        self.assertIn("HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git", build)
        self.assertIn("74bcc7a0d8c8be1267504e21e28a35e199b5d4eb", build)
        self.assertIn("UNITY_VER 213", build)
        self.assertIn("UNITY_PATCH_VER 10", build)
        self.assertIn("BNM/src/Loading.cpp", build)
        self.assertIn("nativeProbe=cache-post-bnm-sfb-pass-through-v1", build)
        self.assertIn("bootstrap3-exact-pass-through-canary", build)
        self.assertIn("assert s.count('BasicHook(') == 1", build)
        self.assertNotIn("build-v240-fixed-native.sh", build)
        self.assertNotIn("V240Fix.cpp", build)
        self.assertNotIn("V240TouchAssist.cpp", build)
        self.assertNotIn("V240EventCompat.cpp", build)

    def test_dynamic_entry_remains_java_only(self):
        dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
        self.assertIn('LEGACY_GEAR_TAG = "adofai-v240-settings-button"', dynamic)
        self.assertIn("scheduleLegacyGearRelocation", dynamic)
        self.assertIn("Gravity.CENTER_VERTICAL | Gravity.END", dynamic)
        self.assertNotIn("V240EventCompat", dynamic)
        self.assertNotIn("System.load(", dynamic)

    def test_channel_requires_bootstrap3_before_canary_delivery(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("Build bootstrap3-gated SFB pass-through canary", workflow)
        self.assertIn("'minBootstrap': 3", workflow)

    def test_rollout_is_boolean_and_canary_scope_stays_single_hook(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            loader = LOADER.read_text(encoding="utf-8")
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertEqual(loader.count("BasicHook("), 1)
            self.assertIn("sfbFilterMemoryRead=0", loader)
            self.assertIn("bootstrap3-exact-pass-through-canary", loader)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
