"""Acceptance-boundary contracts for preparing the complete installed game.

The user-facing app must not be confused with a separately installed editor.
Native/gameplay acceptance is intentionally left to real-device verification.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "android/game-patcher/app/src/main/java/dev/hoonex/adofai/gamepatcher"
MANIFEST = ROOT / "android/game-patcher/app/src/main/AndroidManifest.xml"
WORKFLOW = ROOT / ".github/workflows/game-patcher.yml"


class ModernGameUpgradeContract(unittest.TestCase):
    def test_product_path_uses_actual_installed_331_game_not_v240_file_picker(self):
        main = (SRC / "MainActivity.java").read_text(encoding="utf-8")
        self.assertIn("PatchPipeline.prepare(", main)
        self.assertIn("PatchPipeline.loadPrepared(this)", main)
        self.assertIn("PreparedInstaller.install(this, checked)", main)
        self.assertIn("InstalledGame.isInstalled(this)", main)
        self.assertNotIn("V240PatchPipeline.patch(", main)
        self.assertNotIn("V2.4.0 Custom APK", main)

    def test_install_permissions_declared_and_system_confirmation_used(self):
        manifest = MANIFEST.read_text(encoding="utf-8")
        main = (SRC / "MainActivity.java").read_text(encoding="utf-8")
        self.assertIn("android.permission.REQUEST_INSTALL_PACKAGES", manifest)
        self.assertIn('android:name="com.fizzd.connectedworlds"', manifest)
        self.assertIn('android:name=".InstallResultReceiver"', manifest)
        self.assertIn('android:exported="false"', manifest)
        self.assertIn("backupConsent.isChecked()", main)
        self.assertIn("Intent.ACTION_DELETE", main)
        self.assertIn("Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES", main)
        self.assertNotIn("deletePackage(", main)

    def test_verified_payload_and_split_integrity(self):
        assets = (SRC / "PayloadAssets.java").read_text(encoding="utf-8")
        pipeline = (SRC / "PatchPipeline.java").read_text(encoding="utf-8")
        mutant = (SRC / "ApkMutator.java").read_text(encoding="utf-8")
        self.assertIn("native libOctober payload is missing", assets)
        self.assertIn('assertEntryAbsent(sourceApk, "classes2.dex")', mutant)
        self.assertIn('assertEntryAbsent(sourceApk, "lib/arm64-v8a/libOctober.so")', mutant)
        self.assertIn("verifyOriginalGameEntries(source, signed)", pipeline)
        self.assertIn("prepared game split integrity mismatch", pipeline)
        self.assertIn("prepared game signer differs", pipeline)
        self.assertIn("!sha256(apk).equals", pipeline)

    def test_ci_builds_embedded_editor_into_game_builder(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("bash scripts/build-payload.sh .work/game-patcher-payload", workflow)
        self.assertIn('cp .work/game-patcher-payload/classes2.dex "$ASSETS/classes2.dex"', workflow)
        self.assertIn('cp .work/game-patcher-payload/libOctober.so "$ASSETS/libOctober.so"', workflow)
        self.assertIn(":app:testDebugUnitTest :app:assembleDebug", workflow)
        self.assertIn("'ndk;29.0.14206865'", workflow)
        self.assertNotIn("android-actions/setup-android@v3", workflow)


if __name__ == "__main__":
    unittest.main()
