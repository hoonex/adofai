from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "android" / "game-patcher"
MAIN = PATCHER / "app" / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "gamepatcher"
WORKFLOW = ROOT / ".github" / "workflows" / "game-patcher.yml"


class GamePatcherContractTests(unittest.TestCase):
    def test_runtime_target_is_fail_closed_to_proven_331_build(self):
        text = (MAIN / "InstalledGame.java").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_VERSION_NAME = "3.3.1"', text)
        self.assertIn("EXPECTED_VERSION_CODE = 300382L", text)
        self.assertIn("검증되지 않은 ADOFAI 버전", text)

    def test_patcher_never_silently_uninstalls_or_clears_game(self):
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in MAIN.glob("*.java")
        )
        self.assertNotIn("deletePackage(", corpus)
        self.assertNotIn("pm clear", corpus)
        self.assertNotIn("Runtime.getRuntime().exec", corpus)
        self.assertIn("Intent.ACTION_DELETE", corpus)
        self.assertIn("시스템 삭제 화면", corpus)

    def test_split_install_is_blocked_while_play_signed_original_exists(self):
        text = (MAIN / "PreparedInstaller.java").read_text(encoding="utf-8")
        check = text.index("InstalledGame.isInstalled(context)")
        create = text.index("installer.createSession(params)")
        self.assertLess(check, create)
        self.assertIn("서명이 달라", text)

    def test_bootstrap_patch_is_zero_register_and_payload_class_is_copied_to_main_dex(self):
        text = (MAIN / "DexBootstrapPatcher.java").read_text(encoding="utf-8")
        self.assertIn("Opcode.INVOKE_STATIC, 0, 0, 0, 0, 0, 0", text)
        self.assertIn("MobileEditorBootstrap", text)
        self.assertIn("classes.add(ImmutableClassDef.of(bootstrap))", text)
        self.assertIn("containsBootstrapInvoke", text)

    def test_native_payload_is_uncompressed_and_16k_aligned(self):
        text = (MAIN / "ApkMutator.java").read_text(encoding="utf-8")
        self.assertIn('"lib/arm64-v8a/libOctober.so", Deflater.NO_COMPRESSION', text)
        self.assertIn("library.align(16 * 1024)", text)

    def test_every_split_is_resigned_and_verified_with_one_identity(self):
        pipeline = (MAIN / "PatchPipeline.java").read_text(encoding="utf-8")
        signer = (MAIN / "SplitSigner.java").read_text(encoding="utf-8")
        self.assertIn("for (File source : game.allApks)", pipeline)
        self.assertIn("SplitSigner.signAndVerify", pipeline)
        self.assertIn("signer drift detected", pipeline)
        self.assertIn("ApkVerifier.Builder", signer)
        self.assertIn("identity.sha256", signer)

    def test_workflow_embeds_canonical_payload_without_committing_binaries(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("bash scripts/build-payload.sh .work/game-patcher-payload", text)
        self.assertIn("app/src/main/assets/payload", text)
        self.assertIn("classes2.dex", text)
        self.assertIn("libOctober.so", text)
        self.assertIn(":app:testDebugUnitTest :app:assembleDebug", text)


if __name__ == "__main__":
    unittest.main()
