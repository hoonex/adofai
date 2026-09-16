from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_installs_only_self_fused_exact_pass_through_canary(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "GetEnv", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent", "RunAbiProbeAndMaybeInstallCanary",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-post-bnm-sfb-self-fused-v1",
            "nativeStage=post-bnm-sfb-self-fused-canary",
            "abiProbeRevision=5",
            "sfbHookPolicy=bootstrap1-self-fused-methodinfo-pass-through",
            "sfbCanarySelfFuse=1", "sfbCanaryMarkerReady=",
            "sfbCanaryRecoveryState=", "sfbCanaryHiddenMethodInfo=1",
            "sfbOpenFiltersCanaryCalls=", "sfbOpenFiltersCanaryReturns=",
            "sfbCanaryMarkerWriteFailures=", "sfbFilterMemoryRead=0",
            "struct ExtensionFilterValue", "String* Name;",
            "Array<String*>* Extensions;",
            "IL2CPP::MethodInfo* methodInfo",
            "HookOpenFilePanelFilters", "g_oldOpenFilters",
            "BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters)",
            "return result;",
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

    def test_canary_preserves_hidden_methodinfo_abi(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn(
            "Array<String*>* (*)(\n        String*, String*, Array<ExtensionFilterValue>*, bool, IL2CPP::MethodInfo*)",
            source,
        )
        self.assertIn("bool multiselect,\n        IL2CPP::MethodInfo* methodInfo)", source)
        self.assertIn(
            "original(title, directory, filters, multiselect, methodInfo)", source
        )
        self.assertNotIn(
            "original(title, directory, filters, multiselect);", source
        )

    def test_canary_is_runtime_guarded_by_proven_v240_abi(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            'browser.GetMethod(\n            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"})',
            "openInfo->methodPointer != nullptr", "openFilters._isStatic",
            "openInfo->parameters_count == 4", "SameClass(returnClass, stringArrayClass)",
            "SameClass(param0Class, stringClass)", "SameClass(param1Class, stringClass)",
            "SameClass(param2Class, filterArrayClass)", "SameClass(param3Class, boolClass)",
            "TypeByRef(param2Type) == 0", "TypeValueType(param2Type) == 0",
            "filterType->valuetype", "sizeof(IL2CPP::Il2CppObject) + sizeof(ExtensionFilterValue)",
            "filterInstanceSize == expectedBoxedSize", "filterActualSize == expectedBoxedSize",
            "filterNameOffset == 0", "filterExtensionsOffset == static_cast<long long>(sizeof(void*))",
            "filterNameString", "filterExtensionsStringArray",
        ):
            self.assertIn(required, source)

    def test_canary_self_fuse_covers_install_and_call_abort_boundaries(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "dladdr(", "sfb-canary-r5-install.pending", "sfb-canary-r5-call.pending",
            "sfb-canary-r5-marker-probe.tmp", "O_CREAT | O_EXCL | O_CLOEXEC",
            "fsync(fd)", "MarkerExists(g_installMarkerPath)",
            "MarkerExists(g_callMarkerPath)", "WriteMarker(g_installMarkerPath)",
            "ClearMarker(g_installMarkerPath)", "WriteMarker(g_callMarkerPath)",
            "ClearMarker(g_callMarkerPath)", "g_sfbCanaryCallsInFlight",
        ):
            self.assertIn(required, source)
        hook_start = source.index("Array<String*>* HookOpenFilePanelFilters")
        hook_end = source.index("\n}\n", hook_start) + 2
        hook = source[hook_start:hook_end]
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
        self.assertIn("cache-post-bnm-sfb-self-fused-v1", build)
        self.assertIn("bootstrap1-self-fused-methodinfo-pass-through", build)
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

    def test_channel_delivers_self_fused_canary_to_bootstrap1(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("Build self-fused SFB pass-through canary", workflow)
        self.assertIn("'minBootstrap': 1", workflow)

    def test_rollout_is_boolean_and_canary_scope_stays_single_hook(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            loader = LOADER.read_text(encoding="utf-8")
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertEqual(loader.count("BasicHook("), 1)
            self.assertIn("sfbFilterMemoryRead=0", loader)
            self.assertIn("bootstrap1-self-fused-methodinfo-pass-through", loader)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
