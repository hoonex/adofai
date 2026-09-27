from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R22 = ROOT / "scripts/apply-v240-r22-calibration-persist.py"
R23 = ROOT / "scripts/apply-v240-r23-exact-raycast-resolution.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR23ExactRaycastContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r22 = R22.read_text(encoding="utf-8")
        cls.r23 = R23.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r22_chains_r23(self):
        self.assertIn("apply-v240-r23-exact-raycast-resolution.py", self.r22)
        self.assertIn('subprocess").run([sys.executable, str(r23), str(path)]', self.r22)

    def test_r23_resolves_exact_raycast_overload(self):
        s = self.r23
        for marker in (
            'physics.GetMethod("RaycastAll", {',
            "vector2.GetCompileTimeClass(), vector2.GetCompileTimeClass()",
            "floatClass.GetCompileTimeClass(), intClass.GetCompileTimeClass()",
            'MetadataTypeName(raycastInfo->return_type) == "UnityEngine.RaycastHit2D[]"',
            "tileR23Resolution=RaycastAll-Vector2-Vector2-float-int-exact",
            "tileR23ReturnType=UnityEngine.RaycastHit2D[]",
        ):
            self.assertIn(marker, s)
        self.assertEqual(s.count('physics.GetMethod("RaycastAll", 4)'), 1)  # transform anchor only

    def test_r23_changes_resolution_only(self):
        s = self.r23
        self.assertIn("tileR23MutationDelta=0", s)
        self.assertIn("MaybeInstallCalibrationR22();", s)
        self.assertIn("MaybeInstallTileR21();", s)
        self.assertIn("if s.count('BasicHook(') != 13", s)
        self.assertNotIn("origin.x =", s)
        self.assertNotIn("origin.y =", s)
        self.assertNotIn("SelectFloor(", s)

    def test_build_requires_final_r23_contract(self):
        b = self.build
        for marker in (
            "apply-v240-r23-exact-raycast-resolution.py",
            "stabilityRevision=23",
            "activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window",
            "tileR23Resolution=RaycastAll-Vector2-Vector2-float-int-exact",
            "tileR23ReturnType=UnityEngine.RaycastHit2D[]",
            'MetadataTypeName(raycastInfo->return_type) == "UnityEngine.RaycastHit2D[]"',
        ):
            self.assertIn(marker, b)

    def test_channel_runs_full_v240_suite_and_tracks_r23(self):
        w = self.workflow
        self.assertIn("scripts/apply-v240-r23-exact-raycast-resolution.py", w)
        self.assertIn("tests/test_v240_runtime_r23.py", w)
        self.assertIn("python3 -m unittest discover -s tests -p 'test_v240_*.py' -v", w)
        self.assertIn("Build r23 exact editor raycast and calibration persistence", w)


if __name__ == "__main__":
    unittest.main()
