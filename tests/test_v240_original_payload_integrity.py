from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "android/game-patcher/app/src/main/java/dev/hoonex/adofai/gamepatcher/V240PatchPipeline.java"
HOST_TEST = ROOT / "android/game-patcher/app/src/test/java/dev/hoonex/adofai/gamepatcher/V240HostPatchTest.java"
FINAL_WORKFLOW = ROOT / ".github/workflows/v240-final-release.yml"
PATCHER_WORKFLOW = ROOT / ".github/workflows/v240-custom-patcher.yml"
HOST_TOOLKIT = ROOT / "tools/v240-host-toolkit/src/main/java/dev/hoonex/adofai/v240tool/V240HostPatchCli.java"
HOST_TOOLKIT_WORKFLOW = ROOT / ".github/workflows/v240-host-toolkit.yml"

CRITICAL = {
    "lib/arm64-v8a/libil2cpp.so":
        "c86d7ff549eeef7ecef2c8471019f771e609b3ce7cccf3f775625531b43f3494",
    "lib/arm64-v8a/libunity.so":
        "b18718a2452441d05d9a6eb39bd5cb8bb38603ac0fc740fcb04348bb8f36dfd0",
    "lib/arm64-v8a/libmain.so":
        "8588e1be5a574f2a276696651713e39483904b245c56f91aec970c3170b710bc",
    "assets/bin/Data/Managed/Metadata/global-metadata.dat":
        "26b6c4c711eb43815092925cad2fad99c9de12ba9665d880b7f3e6792823d6f9",
}


class V240OriginalPayloadIntegrityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.host_test = HOST_TEST.read_text(encoding="utf-8")
        cls.final_workflow = FINAL_WORKFLOW.read_text(encoding="utf-8")
        cls.patcher_workflow = PATCHER_WORKFLOW.read_text(encoding="utf-8")
        cls.host_toolkit = HOST_TOOLKIT.read_text(encoding="utf-8")
        cls.host_toolkit_workflow = HOST_TOOLKIT_WORKFLOW.read_text(encoding="utf-8")

    def test_authoritative_critical_entry_hashes_are_pinned_in_patcher(self):
        for entry, digest in CRITICAL.items():
            self.assertIn(entry, self.pipeline)
            self.assertIn(digest, self.pipeline)

    def test_on_device_pipeline_proves_source_unsigned_and_signed_preservation(self):
        self.assertIn(
            "Map<String, String> originalCritical = snapshotCriticalOriginalEntries(source);",
            self.pipeline,
        )
        self.assertIn(
            "assertCriticalOriginalEntriesPreserved(apk, originalCritical);",
            self.pipeline,
        )
        self.assertIn(
            "assertCriticalOriginalEntriesPreserved(signed, originalCritical);",
            self.pipeline,
        )
        self.assertIn("sha256ZipEntry", self.pipeline)

    def test_host_exact_source_path_reuses_same_sha256_contract(self):
        self.assertIn(
            "V240PatchPipeline.snapshotCriticalOriginalEntries(source)",
            self.host_test,
        )
        self.assertIn(
            "V240PatchPipeline.assertCriticalOriginalEntriesPreserved(output, criticalBefore)",
            self.host_test,
        )

    def test_final_release_triggers_on_pipeline_and_rehashes_signed_apk(self):
        self.assertIn(
            "'android/game-patcher/app/src/main/java/dev/hoonex/adofai/gamepatcher/"
            "V240PatchPipeline.java'",
            self.final_workflow,
        )
        self.assertIn("verify_preserved_entry()", self.final_workflow)
        for entry, digest in CRITICAL.items():
            self.assertIn(entry, self.final_workflow)
            self.assertIn(digest, self.final_workflow)

    def test_one_click_patcher_ci_locks_integrity_helpers_and_hashes(self):
        self.assertIn("snapshotCriticalOriginalEntries", self.patcher_workflow)
        self.assertIn("assertCriticalOriginalEntriesPreserved", self.patcher_workflow)
        for digest in CRITICAL.values():
            self.assertIn(digest, self.patcher_workflow)

    def test_standalone_host_toolkit_has_same_critical_entry_contract(self):
        self.assertIn(
            "Map<String, String> criticalBefore = snapshotCriticalOriginalEntries(source);",
            self.host_toolkit,
        )
        self.assertIn(
            "assertCriticalOriginalEntriesPreserved(output, criticalBefore);",
            self.host_toolkit,
        )
        for entry, digest in CRITICAL.items():
            self.assertIn(entry, self.host_toolkit)
            self.assertIn(digest, self.host_toolkit)

    def test_host_toolkit_ci_compiles_and_inspects_integrity_contract(self):
        self.assertIn(
            "'tests/test_v240_original_payload_integrity.py'",
            self.host_toolkit_workflow,
        )
        self.assertIn("Verify authoritative original payload integrity contract",
                      self.host_toolkit_workflow)
        self.assertIn("V240HostPatchCli.class", self.host_toolkit_workflow)
        self.assertIn("snapshotCriticalOriginalEntries", self.host_toolkit_workflow)
        self.assertIn("assertCriticalOriginalEntriesPreserved", self.host_toolkit_workflow)
        for digest in CRITICAL.values():
            self.assertIn(digest, self.host_toolkit_workflow)


if __name__ == "__main__":
    unittest.main()
