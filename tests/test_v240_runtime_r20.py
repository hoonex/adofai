from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R20 = ROOT / "scripts/apply-v240-r20-playerprefsjson-calibration.py"
ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"

class RuntimeR20PlayerPrefsJsonContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r20 = R20.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_uses_exact_game_backend_not_unity_playerprefs(self):
        s = self.r20
        for marker in (
            'Class persistence("", "Persistence")',
            'Class prefsClass("", "PlayerPrefsJson")',
            'persistence.GetMethod("GetInputOffset", 0)',
            'persistence.GetMethod("SetInputOffset", 1)',
            'persistence.GetMethod("get_generalPrefs", 0)',
            'prefsClass.GetMethod("Save", 0)',
            'savePrefs[prefs].Call();',
            'calibrationR20Backend=PlayerPrefsJson',
        ):
            self.assertIn(marker, s)
        self.assertNotIn('Class playerPrefs("UnityEngine", "PlayerPrefs")', s)
        self.assertNotIn('CreateMonoString("offset")', s)

    def test_preserves_real_calibration_and_only_repairs_exact_sentinel(self):
        s = self.r20
        for marker in (
            "kCalibrationR20Unset = 999.0f",
            "kCalibrationR20Neutral = 0.0f",
            "if (before != kCalibrationR20Unset) return;",
            "setOffset.Call(kCalibrationR20Neutral);",
            "if (after == kCalibrationR20Neutral)",
            "calibration-r20-playerprefsjson-write.pending",
        ):
            self.assertIn(marker, s)

    def test_waits_for_general_prefs_without_consuming_attempt(self):
        s = self.r20
        self.assertLess(s.index("if (!prefs) return;"),
                        s.index("g_calibrationR20Attempted.store(1"))
        for marker in (
            "scheduleNativeReconciliation();",
            "1200L", "3500L", "6500L",
            "registerDynamicBridgeViaParent();",
        ):
            self.assertIn(marker, self.entry)

    def test_r19_wrong_backend_is_not_active(self):
        self.assertIn("MaybeRepairCalibrationR20();", self.r20)
        self.assertIn("if '    MaybeRepairCalibrationR19();\\n' in s:", self.r20)
        self.assertIn("calibrationR19Policy=disabled-r20-wrong-backend-forensic-only", self.r20)

    def test_r20_chains_exact_root_fix_successor(self):
        self.assertIn("R21 supersedes the active R20 repair", self.r20)
        self.assertIn("apply-v240-r21-root-fixes.py", self.r20)

    def test_build_validates_r20_is_forensic_only_after_r21(self):
        for marker in (
            "stabilityRevision=23",
            "calibrationR20Policy=disabled-r21-symptom-write-forensic-only",
            "MaybeInstallCalibrationR22();",
            "MaybeInstallTileR21();",
        ):
            self.assertIn(marker, self.build)

    def test_channel_keeps_r20_contract_and_runs_r21(self):
        self.assertIn("tests/test_v240_runtime_r20.py", self.workflow)
        self.assertIn("test_v240_runtime_r20.py", self.workflow)
        self.assertIn("test_v240_runtime_r21.py", self.workflow)
        self.assertIn("test_v240_runtime_r22.py", self.workflow)
        self.assertIn("Build r28 exact ObjectsAtMouse collider synchronization", self.workflow)

if __name__ == "__main__":
    unittest.main()
