from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / 'android/v240-dynamic-runtime/native/V240CacheLoader.cpp'
BRIDGE = ROOT / 'android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/DirectDocumentBridge.java'
ENTRY = ROOT / 'android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java'
BUILD = ROOT / 'scripts/build-v240-cache-native.sh'
R10 = ROOT / 'scripts/apply-v240-r10-raycast-probe.py'
R11 = ROOT / 'scripts/apply-v240-r11-editor-probe.py'
WORKFLOW = ROOT / '.github/workflows/v240-runtime-channel.yml'
ROLLOUT = ROOT / 'android/v240-dynamic-runtime/channel-rollout.txt'


class RuntimeR11Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = LOADER.read_text(encoding='utf-8')
        cls.bridge = BRIDGE.read_text(encoding='utf-8')
        cls.entry = ENTRY.read_text(encoding='utf-8')
        cls.build = BUILD.read_text(encoding='utf-8')
        cls.r10 = R10.read_text(encoding='utf-8')
        cls.r11 = R11.read_text(encoding='utf-8')
        cls.workflow = WORKFLOW.read_text(encoding='utf-8')

    def test_committed_native_source_remains_r9_bounded_base(self):
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

    def test_base_sfb_abi_and_filter_read_are_bounded(self):
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

    def test_base_self_fuses_are_preserved(self):
        s = self.native
        for marker in (
            'sfb-r9-install.pending', 'sfb-r9-call.pending', 'sfb-r9-probe.tmp',
            'uihit-r9-install.pending', 'uihit-r9-call.pending', 'uihit-r9-probe.tmp',
            'WriteMarker(g_sfbInstallMarker)', 'WriteMarker(g_sfbCallMarker)',
            'WriteMarker(g_uiInstallMarker)', 'WriteMarker(g_uiCallMarker)',
            'fsync(fd)',
        ):
            self.assertIn(marker, s)

    def test_r9_ui_hook_is_retained_only_as_forensic_base(self):
        s = self.native
        for marker in (
            'Class controller("", "scrController")',
            'controller.GetMethod("IsScreenPointInsideUIElements", 1)',
            'Class eventSystem("UnityEngine.EventSystems", "EventSystem")',
            'g_pointerPosition[eventData].Set(position)',
            'g_raycastAll[eventSystem].Call(eventData, results)',
            'BasicHook(uiMethod, HookUiHit, g_oldUiHit)',
        ):
            self.assertIn(marker, s)
        self.assertIn('uiHitCalls=', s)

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

    def test_runtime_entry_uses_parent_native_entry_and_context_classloader(self):
        j = self.entry
        for marker in (
            'public static void install(Context context)',
            'registerDynamicBridgeViaParent()',
            'Thread.currentThread()',
            'thread.getContextClassLoader()',
            'RuntimeEntry.class.getClassLoader()',
            'thread.setContextClassLoader(dynamic)',
            'Class.forName("com.unity3d.player.V240CompatibilityReport", true,',
            'getDeclaredMethod("nativeGetCompatibilityReport")',
            'nativeReport.setAccessible(true)',
            'nativeReport.invoke(null)',
            'thread.setContextClassLoader(previous)',
            'scheduleLegacyGearRelocation()',
        ):
            self.assertIn(marker, j)
        self.assertNotIn('private static native', j)
        self.assertNotIn('nativeRegisterDynamicBridge(', j)
        self.assertNotIn('nativeReconcileDynamicRuntime(', j)
        self.assertNotIn('System.load(', j)
        self.assertNotIn('System.loadLibrary(', j)

    def test_r11_overlay_discovers_child_bridge_through_context_loader(self):
        s = self.r11
        for marker in (
            'DiscoverDynamicBridgeFromContextLoader',
            'getContextClassLoader',
            'dev.hoonex.adofai.v240.dynamic.DirectDocumentBridge',
            'RegisterDynamicBridgeClass',
            'dynamicBridgeRegistrationPath=context-classloader-parent-native',
            'dynamicBridgeContextLoaderSeen=',
            'dynamicBridgeLoadClass=',
            'dynamicBridgeMethodResolution=',
            'Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport',
        ):
            self.assertIn(marker, s)

    def test_r11_editor_probe_is_exact_pass_through_and_supersedes_broad_ui_probes(self):
        s = self.r11
        for marker in (
            'abiProbeRevision=11',
            'nativeProbe=cache-post-bnm-scneditor-input-observe-v1',
            'HandleMouseActions', 'SelectFloor',
            'BasicHook(handleMethod, HookEditorMouse, g_oldEditorMouse)',
            'BasicHook(selectMethod, HookSelectFloor, g_oldSelectFloor)',
            'g_oldEditorMouse(self, methodInfo)',
            'g_oldSelectFloor(self, floor, cameraJump, methodInfo)',
            'editorProbePolicy=scnEditor-pass-through-observe-only',
            'editorProbeMutation=0',
            'raycastProbePolicy=disabled-r10-superseded-by-scnEditor',
            'editor-r11-install.pending', 'editor-r11-handle.pending',
            'editor-r11-select.pending',
        ):
            self.assertIn(marker, s)
        self.assertIn('if "MaybeInstallRaycastProbe();" in s:', s)
        self.assertIn('if "MaybeInstallUiHook();" in s:', s)
        self.assertIn('s.count("BasicHook(") != 5', s)

    def test_build_applies_r10_then_r11_on_pinned_upstream(self):
        build = self.build
        self.assertIn('74bcc7a0d8c8be1267504e21e28a35e199b5d4eb', build)
        self.assertIn('UNITY_VER 213', build)
        self.assertIn('UNITY_PATCH_VER 10', build)
        self.assertIn('apply-v240-r10-raycast-probe.py', build)
        self.assertIn('apply-v240-r11-editor-probe.py', build)
        self.assertLess(build.index('python3 "${R10_OVERLAY}"'),
                        build.index('python3 "${R11_OVERLAY}"'))
        for marker in (
            'abiProbeRevision=11',
            'nativeProbe=cache-post-bnm-scneditor-input-observe-v1',
            'editorProbePolicy=scnEditor-pass-through-observe-only',
            'dynamicBridgeRegistrationPath=context-classloader-parent-native',
        ):
            self.assertIn(marker, build)

    def test_channel_requires_rollback_capable_bootstrap_for_active_hooks(self):
        workflow = self.workflow
        self.assertIn('build-v240-cache-native.sh dist/v240-channel-native', workflow)
        self.assertIn('build-v240-dynamic-runtime.sh dist/v240-channel-dynamic', workflow)
        self.assertIn("'minBootstrap': 3", workflow)
        self.assertIn("scripts/apply-v240-r11-editor-probe.py", workflow)

    def test_rollout_scope(self):
        rollout = ROLLOUT.read_text(encoding='utf-8').strip()
        self.assertIn(rollout, ('true', 'false'))
        if rollout == 'true':
            self.assertIn('editorProbeMutation=0', self.r11)
            self.assertIn('sfbOriginalCallUsed=0', self.native)
            self.assertNotIn('System.load(', self.entry)


if __name__ == '__main__':
    unittest.main()
