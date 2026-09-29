from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R27 = ROOT / "scripts/apply-v240-r27-hot-fps-bridge.py"
R28 = ROOT / "scripts/apply-v240-r28-objects-scope-tile-sync.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR28TileSyncContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r27 = R27.read_text(encoding="utf-8")
        cls.r28 = R28.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r27_chains_r28(self):
        self.assertIn("apply-v240-r28-objects-scope-tile-sync.py", self.r27)
        self.assertIn(
            '__import__("subprocess").run([sys.executable, str(r28), str(path)], check=True)',
            self.r27,
        )

    def test_r28_removes_only_touch_gate_from_exact_objects_scope(self):
        s = self.r28
        self.assertIn("if (g_tileR21ObjectsDepth <= 0) {", s)
        self.assertIn(
            "'    if (g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive) {\\n'",
            s,
        )
        self.assertIn(
            "'    if (g_tileR21ObjectsDepth <= 0) {\\n'",
            s,
        )
        self.assertIn("!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms", s)
        self.assertIn("g_tileR21SyncTransforms();", s)
        self.assertIn(
            "g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo)",
            s,
        )
        self.assertIn("tileR28OwnershipBoundary=ObjectsAtMouse", s)
        self.assertIn("tileR28TouchGateRemoved=1", s)
        self.assertIn("tileR28SyncWithoutTouchCalls=", s)

    def test_r28_removes_touch_telemetry_as_install_requirement(self):
        s = self.r28
        self.assertIn("const bool abi = objectsAbi && raycastAbi && sync;", s)
        self.assertIn(
            "'    const bool abi = objectsAbi && raycastAbi && touchAbi && sync;\\n'",
            s,
        )
        self.assertIn(
            "'    const bool abi = objectsAbi && raycastAbi && sync;\\n'",
            s,
        )
        self.assertIn('if (input) g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);', s)
        self.assertIn("const bool touchTelemetryAvailable =", s)

    def test_r28_does_not_add_hooks_or_change_coordinates(self):
        s = self.r28
        self.assertIn('s.count("BasicHook(") != 18', s)
        for forbidden in (
            "origin.x =",
            "origin.y =",
            "SelectFloor(",
            "ScreenToWorldPoint",
            "RaycastHit2D",
        ):
            self.assertNotIn(forbidden, s)

    def test_build_and_channel_require_r28_final_markers(self):
        for marker in (
            "R28_OVERLAY",
            "activeTilePolicy=r28-objects-scope-exact-raycast-sync-plus-full-width-window",
            "tileRepairRevision=28",
            "tileR28OwnershipBoundary=ObjectsAtMouse",
            "tileR28TouchGateRemoved=1",
            "tileR21Policy=ObjectsAtMouse-all-calls-SyncTransforms-before-original-RayCastAll",
        ):
            self.assertIn(marker, self.build)
        self.assertIn("scripts/apply-v240-r28-objects-scope-tile-sync.py", self.workflow)
        self.assertIn("tests/test_v240_runtime_r28.py", self.workflow)
        self.assertIn("Build r28 exact ObjectsAtMouse collider synchronization", self.workflow)


if __name__ == "__main__":
    unittest.main()
