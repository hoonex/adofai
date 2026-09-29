from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/DirectDocumentBridge.java"
R24 = ROOT / "scripts/apply-v240-r24-diagnostic-integrity.py"
R26 = ROOT / "scripts/apply-v240-r26-sfb-save-folder.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR26Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.r24 = R24.read_text(encoding="utf-8")
        cls.r26 = R26.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_child_bridge_reuses_stable_parent_save_and_folder(self):
        s = self.bridge
        for marker in (
            "public static String save(String suggestedName, String mime, long timeoutMs)",
            "public static String folder(long timeoutMs)",
            'Class.forName("com.unity3d.player.V240AndroidBridge", true, loader)',
            '"beginSave", String.class, String.class, String.class',
            'getDeclaredMethod("beginFolder")',
            'getDeclaredMethod("await", int.class, long.class)',
            "PARENT_BEGIN_SAVE.invoke",
            "PARENT_BEGIN_FOLDER.invoke",
            "PARENT_AWAIT.invoke",
        ):
            self.assertIn(marker, s)
        self.assertNotIn("Intent.ACTION_CREATE_DOCUMENT", s)
        self.assertNotIn("Intent.ACTION_OPEN_DOCUMENT_TREE", s)

    def test_save_as_tracks_opened_chart_for_opaque_sidecar_clone(self):
        s = self.bridge
        self.assertIn("LAST_CHART_PATH = chosenChart.file.getAbsolutePath()", s)
        self.assertIn("String source = LAST_CHART_PATH.length() == 0 ? null : LAST_CHART_PATH", s)
        self.assertIn("LAST_CHART_PATH = state.substring(2)", s)
        self.assertIn("directLastChartKnown=", s)

    def test_native_hooks_only_exact_sync_v240_signatures(self):
        s = self.r26
        for marker in (
            'FindSfbMethod(browser, "SaveFilePanel"',
            '"SFB.ExtensionFilter[]"',
            'FindSfbMethod(browser, "OpenFolderPanel"',
            '"System.String[]"',
            "BasicHook(saveString, HookSfbSaveString, g_oldSfbSaveString)",
            "BasicHook(saveFilters, HookSfbSaveFilters, g_oldSfbSaveFilters)",
            "BasicHook(folder, HookSfbFolder, g_oldSfbFolder)",
            "sfb-r26-extra-install.pending",
            "sfb-r26-extra-call.pending",
            "sfbExtraPolicy=exact-v240-sync-SaveFilePanel-OpenFolderPanel-parent-SAF",
        ):
            self.assertIn(marker, s)
        self.assertNotIn("SaveFilePanelAsync", s)
        self.assertNotIn("OpenFolderPanelAsync", s)

    def test_r26_keeps_original_open_and_tile_calibration_repairs(self):
        s = self.r26
        self.assertIn("MaybeInstallSfbHook();", s)
        self.assertIn("MaybeInstallTileR21();", s)
        self.assertIn("MaybeInstallSfbExtraHooks();", s)
        self.assertIn("MaybeInstallCalibrationR22();", self.build)
        self.assertIn("s.count(\"BasicHook(\") != 16", s)

    def test_r24_chains_r26_and_ci_tracks_it(self):
        self.assertIn("apply-v240-r26-sfb-save-folder.py", self.r24)
        self.assertIn("apply-v240-r26-sfb-save-folder.py", self.workflow)
        self.assertIn("test_v240_runtime_r26.py", self.workflow)
        for marker in (
            "sfbExtraRevision=26",
            "sfbExtraPolicy=exact-v240-sync-SaveFilePanel-OpenFolderPanel-parent-SAF",
        ):
            self.assertIn(marker, self.build)


if __name__ == "__main__":
    unittest.main()
