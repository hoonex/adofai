from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "editor-harness" / "app"
MANIFEST = APP / "src" / "main" / "AndroidManifest.xml"
MAIN = APP / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "editorharness" / "MainActivity.java"
BUILD = APP / "build.gradle"
PREPARE = ROOT / "scripts" / "prepare-editor-harness.sh"


class EditorHarnessTests(unittest.TestCase):
    def test_harness_hosts_exact_mobile_editor_shell_without_game_native_runtime(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn("FileSelector.context = this;", text)
        self.assertIn("MobileEditorShell.installLauncher();", text)
        self.assertIn("Preview는 여기서는 의도적으로 게임 런타임에 연결되지 않습니다", text)
        self.assertNotIn("System.loadLibrary", text)
        self.assertNotIn("MobileEditorBootstrap", text)
        self.assertNotIn("nativeQueuePreview", text)

    def test_harness_manifest_requests_same_raw_path_storage_contract(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn("android.permission.MANAGE_EXTERNAL_STORAGE", text)
        self.assertIn("android.permission.READ_EXTERNAL_STORAGE", text)
        self.assertIn("android.permission.WRITE_EXTERNAL_STORAGE", text)
        self.assertIn('android:requestLegacyExternalStorage="true"', text)

    def test_gradle_compiles_generated_sources(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("java.srcDirs += ['generated/java']", text)
        self.assertIn("applicationId 'dev.hoonex.adofai.editorharness'", text)
        self.assertIn("compileSdk 35", text)

    def test_prepare_script_reuses_canonical_transform_chain_and_hardened_picker(self):
        text = PREPARE.read_text(encoding="utf-8")
        lossless = text.index("apply_mobile_editor_document_safety.py")
        strict = text.index("apply_mobile_editor_json_strictness.py")
        picker = text.index("apply_mobile_editor_picker_serialization.py")
        self.assertLess(lossless, strict)
        self.assertLess(strict, picker)
        self.assertIn('bash "${ROOT}/scripts/prepare-upstream.sh"', text)
        self.assertIn('cp "${UPSTREAM_JAVA}/FileSelector.java"', text)
        self.assertIn('cp "${UPSTREAM_JAVA}/CustomFileChooser.java"', text)
        self.assertIn("setOnCancelListener", text)
        self.assertIn("Environment.isExternalStorageManager", text)
        self.assertNotIn("MobileEditorBootstrap.java", text)
        self.assertNotIn("libOctober.so", text)


if __name__ == "__main__":
    unittest.main()
