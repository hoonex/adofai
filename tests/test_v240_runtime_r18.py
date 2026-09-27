from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240WindowCompat.java"
ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
R17 = ROOT / "scripts/apply-v240-r17-editor-physics-sync.py"
R18 = ROOT / "scripts/apply-v240-r18-window-audio-baseline.py"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"

class RuntimeR18WindowContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = WINDOW.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.r17 = R17.read_text(encoding="utf-8")
        cls.r18 = R18.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_full_width_cutout_policy_is_preserved(self):
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.window)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.window)
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.entry)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.entry)

    def test_hot_runtime_reasserts_window_after_parent_bootstrap(self):
        for marker in ("scheduleLegacyWindowNormalization();", "normalizeLegacyWindowViewport()",
                       "1750L", "3000L", "6500L", "decor.requestLayout()"):
            self.assertIn(marker, self.entry)

    def test_r18_disables_speculative_editor_mutations(self):
        for marker in (
            "activeTilePolicy=original-v240-editor-path-plus-full-width-window",
            "windowViewportPolicy=short-edges-full-width",
            "editorProbePolicy=disabled-r18-original-editor-path-window-viewport-fix",
            "editorObjectsPolicy=disabled-r18-original-editor-path-window-viewport-fix",
            "editorPhysicsPolicy=disabled-r18-window-viewport-fix",
        ):
            self.assertIn(marker, self.r18)
        for forbidden_call in (
            "'    MaybeInstallEditorProbe();\\n'",
            "'    MaybeInstallEditorObjectsProbe();\\n'",
            "'    MaybeInstallEditorPhysicsSync();\\n'",
            "'    MaybeNeutralizeUnsetCalibration();\\n'",
        ):
            self.assertIn(forbidden_call, self.r18)

    def test_r17_chains_r18(self):
        self.assertIn("apply-v240-r18-window-audio-baseline.py", self.r17)

    def test_channel_tracks_r18(self):
        self.assertIn("scripts/apply-v240-r18-window-audio-baseline.py", self.workflow)

if __name__ == "__main__":
    unittest.main()
