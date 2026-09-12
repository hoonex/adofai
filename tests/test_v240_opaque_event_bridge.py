from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"
SCANNER = JAVA / "V240ChartCompatibilityScanner.java"
ANDROID_BRIDGE = JAVA / "V240AndroidBridge.java"
LEVEL_BRIDGE = JAVA / "V240LevelFolderBridge.java"
SELECTOR = JAVA / "FileSelector.java"
BUILD = ROOT / "scripts/build-v240-fixed-java.sh"


class V240OpaqueEventBridgeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.android = ANDROID_BRIDGE.read_text(encoding="utf-8")
        cls.level = LEVEL_BRIDGE.read_text(encoding="utf-8")
        cls.selector = SELECTOR.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")

    def test_preserve_only_placeholders_are_inactive_and_sidecar_backed(self):
        self.assertIn('MARKER_PREFIX = "__V240_OPAQUE__:"', self.opaque)
        self.assertIn('"eventType\\":\\"EditorComment\\"', self.opaque)
        self.assertIn('"eventType\\":\\"AddDecoration\\"', self.opaque)
        self.assertIn('\\"active\\":false', self.opaque)
        self.assertIn('writeSmallFile(new File(session, token + ".json"), eventObject)', self.opaque)
        self.assertIn('copyUtf8File(original, writer)', self.opaque)
        self.assertNotIn('LevelEventType', self.opaque)
        self.assertNotIn('libil2cpp', self.opaque)

    def test_bridge_reuses_scanner_inventory_instead_of_duplicating_event_types(self):
        self.assertIn('V240ChartCompatibilityScanner.scanAndLog(chart)', self.opaque)
        self.assertIn('Set<String> targetTypes = scan.postV240', self.opaque)
        for event_type in (
            "SetFrameRate", "SetInputEvent", "AddParticle", "SetParticle", "EmitParticle",
            "SetFilterAdvanced", "TileDimensions",
        ):
            self.assertIn('"' + event_type + '"', self.scanner)
            self.assertNotIn('"' + event_type + '"', self.opaque)

    def test_external_save_builds_restored_private_export_before_opening_saf_output(self):
        self.assertIn('buildRestoredExport', self.opaque)
        self.assertIn('releaseRestoredExport', self.opaque)
        for source in (self.android, self.level):
            self.assertIn('V240OpaqueEventBridge.buildRestoredExport', source)
            self.assertIn('V240OpaqueEventBridge.releaseRestoredExport', source)
            self.assertLess(
                source.index('V240OpaqueEventBridge.buildRestoredExport'),
                source.index('requireOutput(resolver, binding.uri)'),
            )

    def test_save_as_clones_sidecar_before_new_binding(self):
        self.assertIn('beginSave(String suggestedName, String mime, String opaqueSourcePath)', self.android)
        self.assertIn('V240OpaqueEventBridge.cloneSession', self.android)
        clone = self.android.index('V240OpaqueEventBridge.cloneSession')
        bind = self.android.index('bindSave(context, uri, working)', clone)
        self.assertLess(clone, bind)
        self.assertIn('mimeForFilename(suggestedName), filePath', self.selector)

    def test_prepare_happens_before_writable_bindings_and_file_selector_does_not_rescan(self):
        direct_prepare = self.android.index('V240OpaqueEventBridge.prepareForV240')
        direct_bind = self.android.index('bindSave(context, uris[0], workingFiles.get(0))')
        self.assertLess(direct_prepare, direct_bind)

        tree_prepare = self.level.index('V240OpaqueEventBridge.prepareForV240')
        tree_bind = self.level.index('installed = new SaveBinding')
        self.assertLess(tree_prepare, tree_bind)

        self.assertIn('V240OpaqueEventBridge.prepareForV240(chart)', self.selector)
        self.assertNotIn('V240ChartCompatibilityScanner.scanAndLog(chart)', self.selector)

    def test_runtime_dex_contract_contains_scanner_and_opaque_bridge(self):
        self.assertIn('Lcom/unity3d/player/V240ChartCompatibilityScanner;', self.build)
        self.assertIn('Lcom/unity3d/player/V240OpaqueEventBridge;', self.build)


if __name__ == "__main__":
    unittest.main()
