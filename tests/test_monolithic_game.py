"""Source contracts for both monolithic and split ADOFAI game APKs."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/game-patcher/app/src/main/java/dev/hoonex/adofai/gamepatcher"


class GameContainerCompatibility(unittest.TestCase):
    def test_monolithic_detection_uses_file_identity(self):
        source = (JAVA / "PatchPipeline.java").read_text(encoding="utf-8")
        self.assertIn("boolean monolithic = game.baseApk.equals(game.arm64Apk)", source)
        self.assertIn("if (monolithic) {", source)
        self.assertIn(
            "ApkMutator.injectArm64IntoExisting(unsignedBase, payload.libOctober)",
            source,
        )
        self.assertIn("File unsignedArm64 = monolithic", source)

    def test_split_mode_still_patches_distinct_arm64_source(self):
        source = (JAVA / "PatchPipeline.java").read_text(encoding="utf-8")
        self.assertIn(
            "ApkMutator.mutateArm64(game.arm64Apk, unsignedArm64, payload.libOctober)",
            source,
        )
        self.assertIn("for (File source : game.allApks)", source)
        self.assertIn("if (source.equals(game.baseApk)) signingInput = unsignedBase", source)
        self.assertIn("else if (source.equals(game.arm64Apk)) signingInput = unsignedArm64", source)

    def test_monolithic_native_library_injected_into_same_modified_archive(self):
        source = (JAVA / "ApkMutator.java").read_text(encoding="utf-8")
        self.assertIn("static void injectArm64IntoExisting(File modifiedApk", source)
        self.assertIn('assertEntryAbsent(modifiedApk, "lib/arm64-v8a/libOctober.so")', source)
        self.assertIn("new ZipArchive(modifiedApk.toPath())", source)
        self.assertIn("deleteSignatureEntries(zip)", source)

    def test_report_loader_accepts_single_signed_apk_and_validates_hash(self):
        source = (JAVA / "PatchPipeline.java").read_text(encoding="utf-8")
        self.assertIn("list.length < 1", source)
        self.assertIn('expectedNames.contains("base.apk")', source)
        self.assertIn("prepared game split integrity mismatch", source)
        self.assertIn("prepared game signer differs from device keystore", source)

    def test_no_untrusted_version_bypass(self):
        source = (JAVA / "InstalledGame.java").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_VERSION_NAME = "3.3.1"', source)
        self.assertIn("EXPECTED_VERSION_CODE = 300382L", source)
        self.assertIn('hasEntry(base, "classes.dex")', source)
        self.assertIn('hasEntry(apk, "lib/arm64-v8a/libil2cpp.so")', source)


if __name__ == "__main__":
    unittest.main()
