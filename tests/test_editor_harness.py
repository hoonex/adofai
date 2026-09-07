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


class CompanionEditorTests(unittest.TestCase):
    def test_app_is_independent_companion_not_game_patcher(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn("package dev.hoonex.adofai.companion;", text)
        self.assertIn("ADOFAI Companion Editor", text)
        self.assertIn("MobileEditorShell.openStandalone();", text)
        self.assertIn("FileSelector.handleActivityResult", text)
        self.assertIn("MobileEditorShell.openStandalonePath(path);", text)
        self.assertNotIn("System.loadLibrary", text)
        self.assertNotIn("MobileEditorBootstrap", text)
        self.assertNotIn("nativeQueuePreview", text)

    def test_manifest_uses_saf_without_all_files_permissions(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn('android:label="ADOFAI Companion Editor"', text)
        self.assertIn('android:name="com.fizzd.connectedworlds"', text)
        self.assertIn('android.intent.action.VIEW', text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("READ_EXTERNAL_STORAGE", text)
        self.assertNotIn("WRITE_EXTERNAL_STORAGE", text)
        self.assertNotIn("requestLegacyExternalStorage", text)

    def test_saf_bridge_mirrors_and_syncs_documents(self):
        text = SELECTOR.read_text(encoding="utf-8")
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn("Intent.ACTION_CREATE_DOCUMENT", text)
        self.assertIn("takePersistableUriPermission", text)
        self.assertIn("syncSavedPath", text)
        self.assertIn("openInAdofaiOrShare", text)
        self.assertIn('setPackage("com.fizzd.connectedworlds")', text)
        self.assertIn("Intent.ACTION_SEND", text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("CustomFileChooser", text)

    def test_gradle_uses_durable_companion_identity(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("java.srcDirs += ['generated/java']", text)
        self.assertIn("applicationId 'dev.hoonex.adofai.companion'", text)
        self.assertIn("namespace 'dev.hoonex.adofai.companion'", text)
        self.assertIn("compileSdk 35", text)

    def test_prepare_script_keeps_safety_chain_then_applies_companion_mode(self):
        text = PREPARE.read_text(encoding="utf-8")
        lossless = text.index("apply_mobile_editor_document_safety.py")
        strict = text.index("apply_mobile_editor_json_strictness.py")
        picker = text.index("apply_mobile_editor_picker_serialization.py")
        dirty = text.index("apply_mobile_editor_dirty_close_guard.py")
        companion = text.index("apply_companion_editor_mode.py")
        self.assertLess(lossless, strict)
        self.assertLess(strict, picker)
        self.assertLess(picker, dirty)
        self.assertLess(dirty, companion)
        self.assertNotIn("prepare-upstream.sh", text)
        self.assertNotIn("CustomFileChooser", text)
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn("openInAdofaiOrShare", text)

    def test_companion_transform_owns_new_share_and_saf_sync(self):
        text = COMPANION.read_text(encoding="utf-8")
        self.assertIn('makeAction("New"', text)
        self.assertIn('makeAction("ADOFAI / 공유"', text)
        self.assertIn("FileSelector.syncSavedPath", text)
        self.assertIn("FileSelector.displayNameForPath", text)
        self.assertIn("FileSelector.openInAdofaiOrShare", text)
        self.assertIn("openStandalonePath", text)


if __name__ == "__main__":
    unittest.main()
