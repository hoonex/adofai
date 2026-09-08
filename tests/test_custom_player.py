from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "editor-harness" / "app"
BRIDGE = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "OfficialGameBridge.java"
PROVIDER = APP / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "companion" / "OfficialChartProvider.java"
MANIFEST = APP / "src" / "main" / "AndroidManifest.xml"
PREPARE = ROOT / "scripts" / "prepare-editor-harness.sh"


class CanonicalCompanionProductTests(unittest.TestCase):
    def test_official_bridge_targets_exact_unmodified_play_build(self):
        text = BRIDGE.read_text(encoding="utf-8")
        self.assertIn('TARGET_PACKAGE = "com.fizzd.connectedworlds"', text)
        self.assertIn('TARGET_ACTIVITY = "com.unity3d.player.UnityPlayerActivity"', text)
        self.assertIn('EXPECTED_VERSION_NAME = "3.3.1"', text)
        self.assertIn('EXPECTED_VERSION_CODE = 300382L', text)
        self.assertIn('Intent.FLAG_GRANT_READ_URI_PERMISSION', text)
        self.assertIn('OfficialChartProvider.publish', text)
        self.assertNotIn('PlayerActivity.class', text)
        self.assertNotIn('System.loadLibrary', text)

    def test_chart_provider_is_read_only_and_app_scoped(self):
        text = PROVIDER.read_text(encoding="utf-8")
        self.assertIn('MODE_READ_ONLY', text)
        self.assertIn('Read-only provider', text)
        self.assertIn('getFilesDir()', text)
        self.assertNotIn('MANAGE_EXTERNAL_STORAGE', text)

    def test_manifest_has_no_bundled_custom_player(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn('android:label="ADOFAI Companion Editor"', text)
        self.assertIn('OfficialChartProvider', text)
        self.assertIn('android:grantUriPermissions="true"', text)
        self.assertNotIn('android:name=".PlayerActivity"', text)
        self.assertNotIn('ADOFAI Custom', text)

    def test_prepare_pipeline_does_not_apply_custom_game_mode(self):
        text = PREPARE.read_text(encoding="utf-8")
        self.assertIn('apply_companion_editor_mode.py', text)
        self.assertIn('OfficialGameBridge.open', text)
        self.assertNotIn('apply_custom_game_mode.py', text)
        self.assertNotIn('CustomPlayerBridge.open', text)


if __name__ == "__main__":
    unittest.main()
