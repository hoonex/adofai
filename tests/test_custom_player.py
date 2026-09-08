from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "android/editor-harness/app/src/main/java/dev/hoonex/adofai/companion/PlayerActivity.java"
BRIDGE = ROOT / "android/editor-harness/app/src/main/java/com/unity3d/player/CustomPlayerBridge.java"
MANIFEST = ROOT / "android/editor-harness/app/src/main/AndroidManifest.xml"
PREPARE = ROOT / "scripts/prepare-editor-harness.sh"
TRANSFORM = ROOT / "tools/apply_custom_game_mode.py"


class CustomPlayerTests(unittest.TestCase):
    def test_player_is_standalone_and_contains_no_official_runtime_hooking(self):
        text = PLAYER.read_text(encoding="utf-8")
        self.assertIn("class PlayerActivity", text)
        self.assertIn("MediaPlayer", text)
        self.assertIn("parsePathData", text)
        self.assertIn('"SetSpeed"', text)
        self.assertIn('"Twirl"', text)
        self.assertIn('"Pause"', text)
        self.assertIn('"Hold"', text)
        self.assertNotIn("System.loadLibrary", text)
        self.assertNotIn("com.fizzd.connectedworlds", text)
        self.assertNotIn("libil2cpp", text)

    def test_editor_play_action_targets_bundled_player(self):
        bridge = BRIDGE.read_text(encoding="utf-8")
        transform = TRANSFORM.read_text(encoding="utf-8")
        self.assertIn("PlayerActivity.class", bridge)
        self.assertIn("EXTRA_CHART_PATH", bridge)
        self.assertIn('makeAction("Play"', transform)
        self.assertIn("CustomPlayerBridge.open(currentPath)", transform)

    def test_manifest_exposes_nonexported_landscape_player(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn('android:name=".PlayerActivity"', text)
        self.assertIn('android:exported="false"', text)
        self.assertIn('android:screenOrientation="landscape"', text)
        self.assertIn('android:label="ADOFAI Custom"', text)

    def test_prepare_pipeline_compiles_custom_mode_after_lossless_safety(self):
        text = PREPARE.read_text(encoding="utf-8")
        safety = text.index("apply_mobile_editor_document_safety.py")
        custom = text.index("apply_custom_game_mode.py")
        self.assertLess(safety, custom)
        self.assertIn("CustomPlayerBridge.open", text)
        self.assertIn("Unexpected trailing content after JSON value", text)


if __name__ == "__main__":
    unittest.main()
