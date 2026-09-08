from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "editor-harness" / "app"
MANIFEST = APP / "src" / "main" / "AndroidManifest.xml"
MAIN = APP / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "editorharness" / "MainActivity.java"
SELECTOR = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "FileSelector.java"
BUILD = APP / "build.gradle"
PREPARE = ROOT / "scripts" / "prepare-editor-harness.sh"
COMPANION = ROOT / "tools" / "apply_companion_editor_mode.py"
CUSTOM = ROOT / "tools" / "apply_custom_game_mode.py"


class CompanionEditorTests(unittest.TestCase):
    def test_app_is_independent_custom_runtime_not_game_patcher(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn("package dev.hoonex.adofai.companion;", text)
        self.assertIn("ADOFAI Custom", text)
        self.assertIn("MobileEditorShell.openStandalone();", text)
        self.assertIn("FileSelector.handleActivityResult", text)
        self.assertIn("MobileEditorShell.openStandalonePath(path);", text)
        self.assertNotIn("System.loadLibrary", text)
        self.assertNotIn("MobileEditorBootstrap", text)
        self.assertNotIn("nativeQueuePreview", text)
        self.assertNotIn("com.fizzd.connectedworlds", text)

    def test_manifest_uses_saf_without_all_files_permissions(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn('android:label="ADOFAI Custom"', text)
        self.assertIn('android.intent.action.VIEW', text)
        self.assertIn('android:name=".PlayerActivity"', text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("READ_EXTERNAL_STORAGE", text)
        self.assertNotIn("WRITE_EXTERNAL_STORAGE", text)
        self.assertNotIn("requestLegacyExternalStorage", text)
        self.assertNotIn("com.fizzd.connectedworlds", text)

    def test_saf_bridge_mirrors_and_syncs_documents(self):
        text = SELECTOR.read_text(encoding="utf-8")
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn("Intent.ACTION_CREATE_DOCUMENT", text)
        self.assertIn("takePersistableUriPermission", text)
        self.assertIn("syncSavedPath", text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("CustomFileChooser", text)

    def test_gradle_uses_durable_companion_identity(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("java.srcDirs += ['generated/java']", text)
        self.assertIn("applicationId 'dev.hoonex.adofai.companion'", text)
        self.assertIn("namespace 'dev.hoonex.adofai.companion'", text)
        self.assertIn("compileSdk 35", text)

    def test_prepare_script_keeps_safety_chain_then_applies_custom_mode(self):
        text = PREPARE.read_text(encoding="utf-8")
        lossless = text.index("apply_mobile_editor_document_safety.py")
        strict = text.index("apply_mobile_editor_json_strictness.py")
        picker = text.index("apply_mobile_editor_picker_serialization.py")
        dirty = text.index("apply_mobile_editor_dirty_close_guard.py")
        companion = text.index("apply_companion_editor_mode.py")
        custom = text.index("apply_custom_game_mode.py")
        self.assertLess(lossless, strict)
        self.assertLess(strict, picker)
        self.assertLess(picker, dirty)
        self.assertLess(dirty, companion)
        self.assertLess(companion, custom)
        self.assertNotIn("prepare-upstream.sh", text)
        self.assertNotIn("CustomFileChooser", text)
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn("CustomPlayerBridge.open", text)

    def test_companion_layer_preserves_new_and_saf_sync_before_custom_play_layer(self):
        companion = COMPANION.read_text(encoding="utf-8")
        custom = CUSTOM.read_text(encoding="utf-8")
        self.assertIn('makeAction("New"', companion)
        self.assertIn("FileSelector.syncSavedPath", companion)
        self.assertIn("FileSelector.displayNameForPath", companion)
        self.assertIn("openStandalonePath", companion)
        self.assertIn('makeAction("Play"', custom)
        self.assertIn("CustomPlayerBridge.open", custom)


if __name__ == "__main__":
    unittest.main()
