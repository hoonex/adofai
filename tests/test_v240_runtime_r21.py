from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R19 = ROOT / "scripts/apply-v240-r19-exact-calibration.py"
R20 = ROOT / "scripts/apply-v240-r20-playerprefsjson-calibration.py"
R21 = ROOT / "scripts/apply-v240-r21-root-fixes.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR21RootFixContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r19 = R19.read_text(encoding="utf-8")
        cls.r20 = R20.read_text(encoding="utf-8")
        cls.r21 = R21.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_overlay_chain_is_complete(self):
        self.assertIn("apply-v240-r20-playerprefsjson-calibration.py", self.r19)
        self.assertIn("apply-v240-r21-root-fixes.py", self.r20)
        self.assertIn("R21 supersedes the active R20 repair", self.r20)

    def test_calibration_fix_targets_serialization_root_cause(self):
        s = self.r21
        for marker in (
            "CalibrationPreset.ToDict        RVA 0x1F2BF78",
            "CalibrationPreset.FromDict      RVA 0x1F2C0E0",
            "scnSplash.GoToMenu              RVA 0x0C4F5A0",
            "kCalibrationR21ConfidentOffset = 24",
            'preset.GetMethod("FromDict", 1)',
            'preset.GetField("confident")',
            "confident.GetOffset() == kCalibrationR21ConfidentOffset",
            "*confident = 1;",
            "BasicHook(fromDict, HookCalibrationR21FromDict, g_oldCalibrationR21FromDict)",
            "calibrationR21Policy=persisted-CalibrationPreset-FromDict-restore-confident",
            "calibrationR21Mutation=memory-only-persisted-preset-confidence-no-offset-write",
        ):
            self.assertIn(marker, s)

    def test_old_calibration_value_writes_are_not_active(self):
        s = self.r21
        for forbidden in (
            "'    MaybeRepairCalibrationR20();\\n'",
            "'    MaybeRepairCalibrationR19();\\n'",
            "'    MaybeNeutralizeUnsetCalibration();\\n'",
        ):
            self.assertIn(forbidden, s)
        self.assertIn("calibrationR20Policy=disabled-r21-symptom-write-forensic-only", s)
        self.assertIn("MaybeInstallCalibrationR21();", s)

    def test_tile_fix_is_narrow_and_preserves_original_hit_arguments(self):
        s = self.r21
        for marker in (
            "scnEditor.ObjectsAtMouse disassembly (RVA 0x22E8DF4)",
            "Physics2D.RaycastAll (RVA 0x1B9FF44)",
            'editor.GetMethod("ObjectsAtMouse", 0)',
            'physics.GetMethod("RaycastAll", 4)',
            'input.GetMethod("get_touchCount", 0)',
            'resolve("UnityEngine.Physics2D::SyncTransforms()")',
            "g_tileR21SyncTransforms();",
            "g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo)",
            "tileR21CoordinatesModified=0",
            "tileR21RaycastArgumentsModified=0",
            "BasicHook(objects, HookTileR21ObjectsAtMouse, g_oldTileR21ObjectsAtMouse)",
            "BasicHook(raycast, HookTileR21Raycast, g_oldTileR21Raycast)",
        ):
            self.assertIn(marker, s)
        self.assertNotIn("SelectFloor(self", s)
        self.assertNotIn("origin.x =", s)
        self.assertNotIn("origin.y =", s)

    def test_tile_fix_is_touch_and_objects_at_mouse_scoped(self):
        s = self.r21
        self.assertIn("g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive", s)
        self.assertIn("g_tileR21GetTouchCount.Call() > 0", s)
        self.assertIn("g_tileR21SyncedThisObjectsCall = false", s)
        self.assertIn("!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms", s)
        self.assertIn("editor-r21-tile-install.pending", s)
        self.assertIn("editor-r21-objects-call.pending", s)
        self.assertIn("editor-r21-raycast-call.pending", s)

    def test_r21_uses_exact_abi_guards_and_self_fuses(self):
        s = self.r21
        for marker in (
            "preset.GetIl2CppType()->type != IL2CPP::IL2CPP_TYPE_VALUETYPE",
            "TypeCode(info->return_type) == 1",
            "SameClass(confident.GetType(), boolClass)",
            "SameClass(Class(raycastInfo->parameters[0]), vector2)",
            "SameClass(Class(raycastInfo->parameters[2]), floatClass)",
            "SameClass(Class(raycastInfo->parameters[3]), intClass)",
            "calibration-r21-install.pending",
            "calibration-r21-confidence.pending",
            "if s.count('BasicHook(') != 12",
        ):
            self.assertIn(marker, s)

    def test_build_and_channel_publish_r21(self):
        for marker in (
            "apply-v240-r21-root-fixes.py",
            "stabilityRevision=21",
            "calibrationR21Policy=persisted-CalibrationPreset-FromDict-restore-confident",
            "tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll",
            "! grep -q '    MaybeRepairCalibrationR20();'",
        ):
            self.assertIn(marker, self.build)
        self.assertIn("scripts/apply-v240-r21-root-fixes.py", self.workflow)
        self.assertIn("tests/test_v240_runtime_r21.py", self.workflow)
        self.assertIn("test_v240_runtime_r21.py", self.workflow)
        self.assertIn("Build r21 exact calibration confidence and editor collider sync", self.workflow)


    def test_final_binary_assertions_do_not_require_superseded_r19_state(self):
        build = self.build
        self.assertNotIn("stabilityRevision=19", build[build.index('readelf -h'):])
        self.assertNotIn("calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero",
                         build[build.index('readelf -h'):])
        self.assertIn("calibrationR19Policy=disabled-r20-wrong-backend-forensic-only",
                      build[build.index('readelf -h'):])
        self.assertIn("grep -aFq 'stabilityRevision=21'", build)
        self.assertNotIn('strings "${OUT}/libv240fix.so" | grep -q', build)


if __name__ == "__main__":
    unittest.main()
