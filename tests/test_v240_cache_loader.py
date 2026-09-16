from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
ROLLOUT = ROOT / "android/v240-dynamic-runtime/channel-rollout.txt"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"


class V240CacheNativeLoaderContract(unittest.TestCase):
    def test_loader_installs_only_direct_document_saf_hook(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "JNI_OnLoad", "g_vm = vm", "universe.h", "Loading::TryLoadByJNI",
            "Loading::AddOnLoadedEvent", "RunAbiProbeAndMaybeInstallCanary",
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            "nativeProbe=cache-post-bnm-sfb-saf-direct-v1",
            "nativeStage=post-bnm-sfb-saf-direct-document",
            "abiProbeRevision=8",
            "sfbHookPolicy=bootstrap1-self-fused-saf-direct-document",
            "sfbPickerBackend=direct-document", "sfbFileSelectorBypassed=1",
            "sfbCanarySelfFuse=1", "sfbCanaryMarkerReady=",
            "sfbCanaryRecoveryState=", "sfbCanaryHiddenMethodInfo=1",
            "sfbSafBridgeReady=", "sfbOriginalCallUsed=0",
            "sfbSafPickerCalls=", "sfbSafPickerReturns=", "sfbSafLastState=",
            "sfbLastMime=", "sfbOpenFiltersCanaryCalls=", "sfbOpenFiltersCanaryReturns=",
            "sfbCanaryMarkerWriteFailures=", "sfbFilterMemoryRead=1",
            "sfbFilterReadBounded=1", "sfbFilterReadAttempts=",
            "sfbFilterReadSuccess=", "sfbFilterFallbacks=",
            "sfbFilterCount=", "sfbExtensionCount=", "sfbLastExtensions=",
            "struct ExtensionFilterValue", "String* Name;",
            "Array<String*>* Extensions;", "IL2CPP::MethodInfo* methodInfo",
            "HookOpenFilePanelFilters", "g_oldOpenFilters",
            "BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters)",
        ):
            self.assertIn(required, source)
        self.assertEqual(source.count("BasicHook("), 1)
        for forbidden in (
            "InstallAllHooks", "InstallSfbHooks", "InstallMobileHooks",
            "V240SettingsOverlay", "V240EventCompat", "V240TouchAssist",
            ".Call(", ".Set(", "CreateNewObject", "RunOpenPicker(",
            "cache-post-bnm-sfb-saf-v1", "bootstrap1-self-fused-saf-broad-open",
        ):
            self.assertNotIn(forbidden, source)

    def test_direct_bridge_bypasses_file_selector_tree_policy(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            '"com/unity3d/player/V240AndroidBridge"',
            '"com.unity3d.player.V240AndroidBridge"',
            '"beginOpen", "(Ljava/lang/String;Z)I"',
            '"await", "(IJ)Ljava/lang/String;"',
            "CallStaticIntMethod", "CallStaticObjectMethod",
            "sfbPickerBackend=direct-document", "sfbFileSelectorBypassed=1",
            "ResolvePickerMime", "sfbLastMime=",
        ):
            self.assertIn(required, source)
        self.assertNotIn('"com/unity3d/player/FileSelector"', source)
        self.assertNotIn('"selectFile", "(Ljava/lang/String;Z)V"', source)
        self.assertNotIn("GetStaticBooleanField", source)

    def test_filter_read_is_bounded_and_falls_back_broad(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "kMaxExtensionFilters = 32", "kMaxExtensionsPerFilter = 64",
            "kMaxUniqueExtensions = 128", "kMaxExtensionChars = 32",
            "kMaxJoinedExtensions = 512", "ReadFilterExtensions",
            "filters->capacity", "filters->m_Items[i].Extensions",
            "extensions->capacity", "extensions->m_Items[j]",
            "NormalizeExtension", "ContainsExtension", "JoinExtensions",
            "ResolvePickerExtensions", "g_filterFallbacks.fetch_add(1)",
            "return kBroadExtensions;",
        ):
            self.assertIn(required, source)
        read_start = source.index("bool ReadFilterExtensions")
        read_end = source.index("\n}\n\nbool JoinExtensions", read_start) + 2
        read_body = source[read_start:read_end]
        self.assertNotIn(".Name", read_body)
        self.assertIn("if (filterCount > kMaxExtensionFilters) return false;", read_body)
        self.assertIn("if (extensionCount > kMaxExtensionsPerFilter) return false;", read_body)

    def test_mime_mapping_keeps_custom_level_on_document_picker(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            'if (extension == "png") return "image/png";',
            'if (extension == "jpg" || extension == "jpeg") return "image/jpeg";',
            'if (extension == "ogg") return "audio/ogg";',
            'if (extension == "mp3") return "audio/mpeg";',
            'if (extension == "wav") return "audio/wav";',
            'if (extension == "zip" || extension == "adozip") return "application/zip";',
            'if (extension == "json") return "application/json";',
            'return "*/*";',
        ):
            self.assertIn(required, source)
        # .adofai deliberately has no fake MIME mapping; it must stay a normal */* document open.
        self.assertNotIn('extension == "adofai") return', source)

    def test_hook_marks_abort_boundary_before_filter_read_and_never_calls_original(self):
        source = LOADER.read_text(encoding="utf-8")
        hook_start = source.index("Array<String*>* HookOpenFilePanelFilters")
        hook_end = source.index("\n}\n", hook_start) + 2
        hook = source[hook_start:hook_end]
        self.assertIn("WriteMarker(g_callMarker)", hook)
        self.assertIn("ResolvePickerExtensions(filters)", hook)
        self.assertIn("RunSafPicker(multiselect, extensions)", hook)
        self.assertLess(hook.index("WriteMarker(g_callMarker)"), hook.index("ResolvePickerExtensions(filters)"))
        self.assertNotIn("original(", hook)
        self.assertNotIn("g_oldOpenFilters(", hook)

    def test_hidden_methodinfo_signature_is_preserved(self):
        source = LOADER.read_text(encoding="utf-8")
        self.assertIn(
            "Array<String*>* (*)(\n        String*, String*, Array<ExtensionFilterValue>*, bool, IL2CPP::MethodInfo*)",
            source,
        )
        self.assertIn("bool multiselect, IL2CPP::MethodInfo* methodInfo)", source)
        self.assertIn("(void)methodInfo;", source)

    def test_hook_is_runtime_guarded_by_proven_v240_abi_and_bridge(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            'browser.GetMethod(\n            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"})',
            "info && info->methodPointer", "openFilters._isStatic",
            "info->parameters_count == 4", "SameClass(returnClass, stringArrayClass)",
            "SameClass(Class(p0), stringClass)", "SameClass(Class(p1), stringClass)",
            "SameClass(Class(p2), filterArrayClass)", "SameClass(Class(p3), boolClass)",
            "TypeByRef(p2) == 0", "TypeValueType(p2) == 0",
            "filterType->valuetype", "sizeof(IL2CPP::Il2CppObject) + sizeof(ExtensionFilterValue)",
            "filterClass->instance_size == expectedBoxedSize", "filterClass->actualSize == expectedBoxedSize",
            "nameField.GetOffset() == 0",
            "extensionsField.GetOffset() == static_cast<int32_t>(sizeof(void*))",
            "SameClass(nameField.GetType(), stringClass)",
            "SameClass(extensionsField.GetType(), stringArrayClass)", "ProbeSafBridge()",
            "abiGuard && safReady && fuseReady",
        ):
            self.assertIn(required, source)

    def test_self_fuse_covers_install_filter_read_and_direct_picker_boundaries(self):
        source = LOADER.read_text(encoding="utf-8")
        for required in (
            "dladdr(", "sfb-canary-r8-install.pending", "sfb-canary-r8-call.pending",
            "sfb-canary-r8-marker-probe.tmp", "O_CREAT | O_EXCL | O_CLOEXEC",
            "fsync(fd)", "MarkerExists(g_installMarker)",
            "MarkerExists(g_callMarker)", "WriteMarker(g_installMarker)",
            "ClearMarker(g_installMarker)", "WriteMarker(g_callMarker)",
            "ClearMarker(g_callMarker)", "g_callsInFlight",
        ):
            self.assertIn(required, source)

    def test_cache_native_build_pins_bnm_and_enforces_direct_document_scope(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("V240CacheLoader.cpp", build)
        self.assertIn("HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git", build)
        self.assertIn("74bcc7a0d8c8be1267504e21e28a35e199b5d4eb", build)
        self.assertIn("UNITY_VER 213", build)
        self.assertIn("UNITY_PATCH_VER 10", build)
        self.assertIn("BNM/src/Loading.cpp", build)
        self.assertIn("cache-post-bnm-sfb-saf-direct-v1", build)
        self.assertIn("bootstrap1-self-fused-saf-direct-document", build)
        self.assertIn("assert s.count('BasicHook(') == 1", build)
        self.assertIn("ReadFilterExtensions", build)
        self.assertIn("V240AndroidBridge", build)
        self.assertIn("CallStaticIntMethod", build)
        self.assertIn("CreateMonoString", build)
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

    def test_channel_delivers_runtime_to_bootstrap1(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-v240-cache-native.sh dist/v240-channel-native", workflow)
        self.assertNotIn("build-v240-fixed-native.sh dist/v240-channel-native", workflow)
        self.assertIn("cp dist/v240-channel-native/libv240fix.so", workflow)
        self.assertIn("test_v240_cache_loader.py", workflow)
        self.assertIn("'minBootstrap': 1", workflow)

    def test_rollout_is_boolean_and_scope_stays_single_hook(self):
        rollout = ROLLOUT.read_text(encoding="utf-8").strip()
        self.assertIn(rollout, ("true", "false"))
        if rollout == "true":
            loader = LOADER.read_text(encoding="utf-8")
            dynamic = DYNAMIC_ENTRY.read_text(encoding="utf-8")
            self.assertEqual(loader.count("BasicHook("), 1)
            self.assertIn("sfbFilterMemoryRead=1", loader)
            self.assertIn("sfbFilterReadBounded=1", loader)
            self.assertIn("sfbOriginalCallUsed=0", loader)
            self.assertIn("sfbFileSelectorBypassed=1", loader)
            self.assertIn("bootstrap1-self-fused-saf-direct-document", loader)
            self.assertNotIn("System.load(", dynamic)


if __name__ == "__main__":
    unittest.main()
