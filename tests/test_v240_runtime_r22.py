from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R21 = ROOT / "scripts/apply-v240-r21-root-fixes.py"
R22 = ROOT / "scripts/apply-v240-r22-calibration-persist.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR22PersistenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r21 = R21.read_text(encoding="utf-8")
        cls.r22 = R22.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r21_chains_r22(self):
        self.assertIn("apply-v240-r22-calibration-persist.py", self.r21)
        self.assertIn('subprocess").run([sys.executable, str(r22), str(path)]', self.r21)

    def test_r22_uses_actual_persistence_boundary(self):
        s = self.r22
        for marker in (
            "scrConductor.SaveCurrentPreset at RVA 0x218576C",
            "Persistence.WriteSaveToDisk (RVA 0x1169E64)",
            "Persistence.Save",
            'conductor.GetMethod("SaveCurrentPreset", 0)',
            'persistence.GetMethod("Save", 0)',
            "g_oldCalibrationR22SaveCurrentPreset(methodInfo);",
            "g_calibrationR22PersistenceSave.Call();",
            "calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced",
            "calibrationR22Mutation=persistence-schedule-only-no-preset-value-change",
        ):
            self.assertIn(marker, s)

    def test_r22_retires_disproven_confidence_mutation(self):
        s = self.r22
        self.assertIn(
            "calibrationR21Policy=disabled-r22-PersistenceLoad-already-restores-confident", s
        )
        self.assertIn("calibrationR22PersistenceLoadConfidentStoreRva=0x1169674", s)
        self.assertIn("calibrationR22PersistenceLoadFromDictRva=0x11696B8", s)
        self.assertIn("'    MaybeInstallCalibrationR21();\\n'", s)
        self.assertIn("MaybeInstallCalibrationR22();", s)

    def test_no_calibration_value_or_identity_mutation(self):
        s = self.r22
        self.assertNotIn("SetInputOffset", s)
        self.assertNotIn("GetAudioOutputName", s)
        self.assertNotIn("currentPreset.confident =", s)
        self.assertNotIn("*confident =", s)
        self.assertNotIn("outputName =", s)
        self.assertNotIn("inputOffset =", s)

    def test_r22_is_self_fused_and_exact_abi_guarded(self):
        s = self.r22
        for marker in (
            "calibration-r22-persist-install.pending",
            "calibration-r22-persist-call.pending",
            "saveCurrentPreset._isStatic",
            "persistenceSave._isStatic",
            "currentInfo->parameters_count == 0",
            "persistenceInfo->parameters_count == 0",
            "TypeCode(currentInfo->return_type) == 1",
            "TypeCode(persistenceInfo->return_type) == 1",
            "BasicHook(saveCurrentPreset, HookCalibrationR22SaveCurrentPreset",
            "if s.count('BasicHook(') != 13",
        ):
            self.assertIn(marker, s)

    def test_r21_tile_root_fix_remains_active(self):
        s = self.r22
        self.assertIn("MaybeInstallTileR21();", s)
        self.assertNotIn("origin.x =", s)
        self.assertNotIn("origin.y =", s)

    def test_build_asserts_only_final_active_calibration_policy(self):
        b = self.build
        self.assertIn("apply-v240-r22-calibration-persist.py", b)
        self.assertIn("stabilityRevision=22", b)
        self.assertIn(
            "calibrationR21Policy=disabled-r22-PersistenceLoad-already-restores-confident", b
        )
        self.assertIn(
            "calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced", b
        )
        self.assertNotIn("grep -q 'stabilityRevision=19'", b)
        self.assertNotIn(
            "grep -q 'calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero'", b
        )

    def test_channel_tracks_tests_and_publishes_r22(self):
        w = self.workflow
        self.assertIn("scripts/apply-v240-r22-calibration-persist.py", w)
        self.assertIn("tests/test_v240_runtime_r22.py", w)
        self.assertIn("test_v240_runtime_r22.py", w)
        self.assertIn("Build r22 calibration persistence and editor collider sync", w)


if __name__ == "__main__":
    unittest.main()
