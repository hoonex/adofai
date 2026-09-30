from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
R29 = ROOT / "scripts/apply-v240-r29-il2cpp-resolver-fallback.py"
R30 = ROOT / "scripts/apply-v240-r30-tile-result-observation.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class V240RuntimeR30ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r29 = R29.read_text(encoding="utf-8")
        cls.r30 = R30.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r29_chains_r30(self):
        self.assertIn("apply-v240-r30-tile-result-observation.py", self.r29)

    def test_sync_mutation_is_retired_after_real_device_failure(self):
        self.assertIn("tileR30SyncRetired=1", self.r30)
        self.assertIn("tileR30Mutation=0", self.r30)
        self.assertIn("activeTilePolicy=r30-read-only-original-raycast-and-objects-results", self.r30)
        self.assertIn("tileR21Policy=ObjectsAtMouse-original-results-observe-only-r30", self.r30)
        self.assertIn('if "g_tileR21SyncTransforms();" in s:', self.r30)

    def test_existing_two_hooks_are_observation_only(self):
        self.assertIn('if s.count("BasicHook(") != 18:', self.r30)
        self.assertIn("auto* result = g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);", self.r30)
        self.assertIn("EditorObjectsArray* result = g_oldTileR21ObjectsAtMouse(self, methodInfo);", self.r30)
        self.assertNotIn("SelectFloor", self.r30)
        self.assertNotIn("origin.x =", self.r30)
        self.assertNotIn("origin.y =", self.r30)

    def test_two_raycast_and_final_object_counts_are_bounded(self):
        for marker in (
            "kTileR30MaxObservedArray = 4096",
            "ObserveTileR30ArrayCount",
            "result->capacity",
            "tileR30FirstRaycastLastCount=",
            "tileR30SecondRaycastLastCount=",
            "tileR30ObjectsLastCount=",
            "tileR30CountGuardFailures=",
        ):
            self.assertIn(marker, self.r30)

    def test_exact_query_arguments_are_observed_not_modified(self):
        for marker in (
            "tileR30FirstOriginX1000=",
            "tileR30FirstOriginY1000=",
            "tileR30SecondOriginX1000=",
            "tileR30SecondOriginY1000=",
            "tileR30FirstDirectionX1000=",
            "tileR30FirstDirectionY1000=",
            "tileR30SecondDirectionX1000=",
            "tileR30SecondDirectionY1000=",
            "tileR30FirstDistanceX1000=",
            "tileR30SecondDistanceX1000=",
            "tileR30FirstLayerMask=",
            "tileR30SecondLayerMask=",
        ):
            self.assertIn(marker, self.r30)

    def test_install_no_longer_depends_on_disproven_sync(self):
        self.assertIn("'    const bool abi = objectsAbi && raycastAbi;\\n'", self.r30)
        self.assertIn("g_oldTileR21Raycast != nullptr;", self.r30)

    def test_build_and_channel_require_r30(self):
        for marker in (
            'R30_OVERLAY="${ROOT}/scripts/apply-v240-r30-tile-result-observation.py"',
            'test -f "${R30_OVERLAY}"',
            "tileResultRevision=30",
            "tileR30Policy=read-only-two-raycasts-plus-final-objects-result",
            "tileR30Mutation=0",
            "tileR30SyncRetired=1",
        ):
            self.assertIn(marker, self.build)
        self.assertIn("'scripts/apply-v240-r30-tile-result-observation.py'", self.workflow)
        self.assertIn("'tests/test_v240_runtime_r30.py'", self.workflow)


if __name__ == "__main__":
    unittest.main()
