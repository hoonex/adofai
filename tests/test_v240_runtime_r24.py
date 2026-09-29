from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R23 = ROOT / "scripts/apply-v240-r23-exact-raycast-resolution.py"
R24 = ROOT / "scripts/apply-v240-r24-diagnostic-integrity.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"

class RuntimeR24DiagnosticIntegrityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r23 = R23.read_text(encoding="utf-8")
        cls.r24 = R24.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r23_chains_diagnostic_only_successor(self):
        self.assertIn("apply-v240-r24-diagnostic-integrity.py", self.r23)
        self.assertIn('subprocess").run([sys.executable, str(r24), str(path)]', self.r23)

    def test_final_counter_owns_only_current_active_hook_sites(self):
        s = self.r24
        for marker in (
            "(g_sfbHookInstalled.load() ? 1 : 0)",
            "(g_calibrationR22HookInstalled.load() ? 1 : 0)",
            "(g_tileR21HookInstalled.load() ? 2 : 0)",
            "diagnosticRevision=24",
            "activeHookPolicy=sfb-1-calibrationR22-1-tileR21-2",
            "gameHooksInstalledSemantics=active-installed-hook-sites",
            "historicalInactiveHookSitesExcluded=1",
        ):
            self.assertIn(marker, s)
        new_block = s[s.index("new_count ="):s.index("once(old_count, new_count)")]
        self.assertNotIn("(g_editorProbeInstalled.load() ? 2 : 0) +", new_block)
        self.assertNotIn("(g_editorObjectsHookInstalled.load() ? 1 : 0) +", new_block)

    def test_r24_does_not_change_r23_behavior(self):
        s = self.r24
        self.assertIn("stabilityRevision=23", s)
        self.assertIn("activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window", s)
        self.assertIn("calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced", s)
        self.assertIn("MaybeInstallCalibrationR22();", s)
        self.assertIn("MaybeInstallTileR21();", s)
        self.assertIn("MaybeInstallSfbHook();", s)
        self.assertEqual(s.count("BasicHook("), 2)  # only the compiled-site count assertion

    def test_build_and_channel_publish_r24_diagnostics(self):
        for marker in (
            "apply-v240-r24-diagnostic-integrity.py",
            "diagnosticRevision=24",
            "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2",
            "gameHooksInstalledSemantics=active-installed-hook-sites",
            "historicalInactiveHookSitesExcluded=1",
        ):
            self.assertIn(marker, self.build)
        self.assertIn("scripts/apply-v240-r24-diagnostic-integrity.py", self.workflow)
        self.assertIn("tests/test_v240_runtime_r24.py", self.workflow)
        self.assertIn("python3 -m unittest discover -s tests -p 'test_v240_*.py' -v", self.workflow)

if __name__ == "__main__":
    unittest.main()
