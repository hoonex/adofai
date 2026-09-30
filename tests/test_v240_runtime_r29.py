from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
R28 = ROOT / "scripts/apply-v240-r28-objects-scope-tile-sync.py"
R29 = ROOT / "scripts/apply-v240-r29-il2cpp-resolver-fallback.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class V240RuntimeR29ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r28 = R28.read_text(encoding="utf-8")
        cls.r29 = R29.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r28_chains_r29_without_adding_a_new_hook_family(self):
        self.assertIn("apply-v240-r29-il2cpp-resolver-fallback.py", self.r28)
        self.assertIn('if s.count("BasicHook(") != 18:', self.r29)

    def test_resolver_falls_back_to_loaded_libil2cpp(self):
        for marker in (
            'dlsym(RTLD_DEFAULT, "il2cpp_resolve_icall")',
            'dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD)',
            'dladdr(managedMethodPointer, &owner)',
            'dlopen(owner.dli_fname, RTLD_NOW | RTLD_NOLOAD)',
            'dlsym(handle, "il2cpp_resolve_icall")',
            'resolve("UnityEngine.Physics2D::SyncTransforms()")',
        ):
            self.assertIn(marker, self.r29)
        self.assertNotIn("RTLD_NOW | RTLD_LOCAL", self.r29)

    def test_exact_tile_gates_are_reported_individually(self):
        for marker in (
            "tileResolverRevision=29",
            "tileR29EditorClassReady=",
            "tileR29PhysicsClassReady=",
            "tileR29PrimitiveTypesReady=",
            "tileR29ObjectsAbiReady=",
            "tileR29RaycastAbiReady=",
            "tileR29ResolveSymbolPath=",
            "tileR29ResolveIcallResult=",
        ):
            self.assertIn(marker, self.r29)

    def test_r28_ownership_and_touch_independence_are_preserved(self):
        self.assertNotIn(
            "g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive",
            self.r29,
        )
        self.assertNotIn(
            "const bool abi = objectsAbi && raycastAbi && touchAbi && sync;",
            self.r29,
        )
        self.assertIn("g_tileR21IcallReady.store(sync ? 1 : 0);", self.r29)

    def test_native_build_contract_requires_r29(self):
        for marker in (
            'R29_OVERLAY="${ROOT}/scripts/apply-v240-r29-il2cpp-resolver-fallback.py"',
            'test -f "${R29_OVERLAY}"',
            "tileResolverRevision=29",
            "tileR29ResolverPolicy=RTLD_DEFAULT-then-method-owner-NOLOAD-then-SONAME-NOLOAD",
            "tileR29ResolveSymbolPath=",
            "tileR29ResolveIcallResult=",
        ):
            self.assertIn(marker, self.build)

    def test_runtime_channel_tracks_r29_source_and_contract(self):
        self.assertIn("'scripts/apply-v240-r29-il2cpp-resolver-fallback.py'", self.workflow)
        self.assertIn("'tests/test_v240_runtime_r29.py'", self.workflow)


if __name__ == "__main__":
    unittest.main()
