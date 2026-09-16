from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / 'android/v240-dynamic-runtime/native/V240CacheLoader.cpp'
BRIDGE = ROOT / 'android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/DirectDocumentBridge.java'
ENTRY = ROOT / 'android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java'
BUILD = ROOT / 'scripts/build-v240-cache-native.sh'
WORKFLOW = ROOT / '.github/workflows/v240-runtime-channel.yml'
ROLLOUT = ROOT / 'android/v240-dynamic-runtime/channel-rollout.txt'


class RuntimeR9Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = LOADER.read_text(encoding='utf-8')
        cls.bridge = BRIDGE.read_text(encoding='utf-8')
        cls.entry = ENTRY.read_text(encoding='utf-8')

    def test_revision_and_scope(self):
        s = self.native
        for marker in (
            'nativeProbe=cache-post-bnm-sfb-dynamic-import-uihit-v1',
            'nativeStage=post-bnm-dynamic-document-and-uihit',
            'abiProbeRevision=9',
            'sfbHookPolicy=dynamic-document-preprocess-before-bind',
            'sfbPickerBackend=dynamic-document',
            'sfbEmbeddedBridgeBypassed=1',
            'uiHitPolicy=pinned-upstream-eventsystem-raycast',
            'uiHitSourceCommit=74bcc7a0d8c8be1267504e21e28a35e199b5d4eb',
        ):
            self.assertIn(marker, s)
        self.assertEqual(s.count('BasicHook('), 2)
        self.assertNotIn('InstallAllHooks', s)
        self.assertNotIn('InstallSfbHooks', s)
        self.assertNotIn('InstallMobileHooks', s)

    def test_dynamic_bridge_is_registered_before_sfb_install(self):
        s = self.native
        for marker in (
            'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeRegisterDynamicBridge',
            'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeReconcileDynamicRuntime',
            '"begin", "(Ljava/lang/String;Ljava/lang/String;Z)I"',
            '"await", "(IJ)Ljava/lang/String;"',
            '"diagnostics", "()Ljava/lang/String;"',
            'g_dynamicBridgeReady.store(true, std::memory_order_release)',
            '!g_dynamicBridgeReady.load(std::memory_order_acquire)',
        ):
            self.assertIn(marker, s)
        self.assertNotIn('"com/unity3d/player/FileSelector"', s)
        self.assertNotIn('"com/unity3d/player/V240AndroidBridge"', s)

    def test_sfb_abi_and_filter_read_are_bounded(self):
        s = self.native
        for marker in (
            'Array<String*>* (*)(\n        String*, String*, Array<ExtensionFilterValue>*, bool, IL2CPP::MethodInfo*)',
            'kMaxExtensionFilters = 32', 'kMaxExtensionsPerFilter = 64',
            'kMaxUniqueExtensions = 128', 'kMaxExtensionChars = 32',
            'kMaxJoinedExtensions = 512', 'filters->m_Items[i].Extensions',
            'extensions->m_Items[j]', 'ResolvePickerExtensions(filters)',
            'sfbOriginalCallUsed=0',
        ):
            self.assertIn(marker, s)
        hook = s[s.index('Array<String*>* HookOpenFilePanelFilters'):]
        hook = hook[:hook.index('\n}\n') + 2]
        self.assertIn('RunDynamicPicker(multiselect, extensions)', hook)
        self.assertNotIn('g_oldOpenFilters(', hook)
        self.assertNotIn('original(', hook)

    def test_two_independent_self_fuses(self):
        s = self.native
        for marker in (
            'sfb-r9-install.pending', 'sfb-r9-call.pending', 'sfb-r9-probe.tmp',
            'uihit-r9-install.pending', 'uihit-r9-call.pending', 'uihit-r9-probe.tmp',
            'WriteMarker(g_sfbInstallMarker)', 'WriteMarker(g_sfbCallMarker)',
            'WriteMarker(g_uiInstallMarker)', 'WriteMarker(g_uiCallMarker)',
            'fsync(fd)',
        ):
            self.assertIn(marker, s)

    def test_ui_hook_matches_pinned_upstream_algorithm(self):
        s = self.native
        for marker in (
            'Class controller("", "scrController")',
            'controller.GetMethod("IsScreenPointInsideUIElements", 1)',
            'Class eventSystem("UnityEngine.EventSystems", "EventSystem")',
            'Class pointerEventData("UnityEngine.EventSystems", "PointerEventData")',
            'Class raycastResult("UnityEngine.EventSystems", "RaycastResult")',
            'g_eventSystemCurrent = eventSystem.GetProperty("current")',
            'g_pointerPosition[eventData].Set(position)',
            'g_raycastAll[eventSystem].Call(eventData, results)',
            'return g_listCount[results].Get() > 0',
            'BasicHook(uiMethod, HookUiHit, g_oldUiHit)',
            'uiHitOriginalCalls=',
        ):
            self.assertIn(marker, s)
        self.assertEqual(s.count('g_oldUiHit(self, position, methodInfo)'), 1)

    def test_dynamic_import_preprocesses_before_save_binding(self):
        j = self.bridge
        for marker in (
            'Intent.ACTION_OPEN_DOCUMENT',
            'V240ChartBackport', 'V240HallLegacyFix', 'V240OpaqueEventBridge',
            'V240MapCompatibility', 'V240AndroidBridge',
            'backportForV240', 'applyIfNeeded', 'prepareForV240', 'repairMap', 'bindSave',
            'BIND_SAVE.setAccessible(true)',
            'MAX_FILES = 128', 'MAX_BYTES = 512L * 1024L * 1024L',
        ):
            self.assertIn(marker, j)
        self.assertNotIn('Intent.ACTION_OPEN_DOCUMENT_TREE', j)
        self.assertLess(j.index('invokeBoolean(BACKPORT'), j.index('BIND_SAVE.invoke'))
        self.assertLess(j.index('invokeBoolean(HALL_FIX'), j.index('BIND_SAVE.invoke'))
        self.assertLess(j.index('invokeBoolean(OPAQUE_PREPARE'), j.index('BIND_SAVE.invoke'))
        self.assertLess(j.index('invokeVoid(MAP_REPAIR'), j.index('BIND_SAVE.invoke'))

    def test_level_open_allows_sibling_asset_selection_without_changing_sfb_return_shape(self):
        j = self.bridge
        self.assertIn('final boolean bundleMode = isOnlyLevelExtension(safeExtensions);', j)
        self.assertIn('final boolean allowMultiple = requestedMultiselect || bundleMode;', j)
        self.assertIn('intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, allowMultiple);', j)
        self.assertIn('if (!requestedMulti && chosenChart != null)', j)
        self.assertIn('encoded = chosenChart.file.getAbsolutePath();', j)
        for marker in ('directFilesCopied=', 'directChartCount=', 'directBundleMode=',
                       'directMultiAssetSelection=', 'directBackportChanged=',
                       'directOpaquePrepared=', 'directSaveBound=', 'directImportErrors='):
            self.assertIn(marker, j)

    def test_runtime_entry_registers_bridge_without_loading_native_again(self):
        j = self.entry
        for marker in (
            'nativeRegisterDynamicBridge(DirectDocumentBridge.class)',
            'nativeReconcileDynamicRuntime()',
            'public static void install(Context context)',
            'scheduleLegacyGearRelocation()',
        ):
            self.assertIn(marker, j)
        self.assertNotIn('System.load(', j)
        self.assertNotIn('System.loadLibrary(', j)

    def test_build_and_channel_preserve_pinned_toolchain_and_hot_swap(self):
        build = BUILD.read_text(encoding='utf-8')
        workflow = WORKFLOW.read_text(encoding='utf-8')
        self.assertIn('74bcc7a0d8c8be1267504e21e28a35e199b5d4eb', build)
        self.assertIn('UNITY_VER 213', build)
        self.assertIn('UNITY_PATCH_VER 10', build)
        self.assertIn("assert s.count('BasicHook(') == 2", build)
        self.assertIn('build-v240-cache-native.sh dist/v240-channel-native', workflow)
        self.assertIn('build-v240-dynamic-runtime.sh dist/v240-channel-dynamic', workflow)
        self.assertIn("'minBootstrap': 1", workflow)

    def test_rollout_scope(self):
        rollout = ROLLOUT.read_text(encoding='utf-8').strip()
        self.assertIn(rollout, ('true', 'false'))
        if rollout == 'true':
            self.assertEqual(self.native.count('BasicHook('), 2)
            self.assertIn('sfbOriginalCallUsed=0', self.native)
            self.assertNotIn('System.load(', self.entry)


if __name__ == '__main__':
    unittest.main()
