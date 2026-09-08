from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "android" / "zygisk-runtime" / "module"
CPP = MODULE / "jni" / "module.cpp"
PROP = MODULE / "module.prop"
SELECTOR = ROOT / "android" / "zygisk-runtime" / "java" / "com" / "unity3d" / "player" / "FileSelector.java"
BOOTSTRAP = ROOT / "android" / "zygisk-runtime" / "java" / "com" / "unity3d" / "player" / "ZygiskEditorBootstrap.java"
BUILD = ROOT / "scripts" / "build-zygisk-runtime-module.sh"
NATIVE_TRANSFORM = ROOT / "tools" / "apply_zygisk_native_mode.py"
EDITOR_TRANSFORM = ROOT / "tools" / "apply_zygisk_editor_mode.py"
WORKFLOW = ROOT / ".github" / "workflows" / "zygisk-runtime-module.yml"


class ZygiskRuntimeModuleTests(unittest.TestCase):
    def test_module_targets_only_official_adofai_and_preserves_apk_identity(self):
        text = CPP.read_text(encoding="utf-8")
        self.assertIn('kTargetProcess = "com.fizzd.connectedworlds"', text)
        self.assertIn("getModuleDir()", text)
        self.assertIn("exemptFd", text)
        self.assertIn("ANDROID_DLEXT_USE_LIBRARY_FD", text)
        self.assertIn("InMemoryDexClassLoader", text)
        self.assertIn("RegisterNatives", text)
        self.assertNotIn("PackageInstaller", text)
        self.assertNotIn("apksig", text.lower())
        self.assertNotIn("uninstall", text.lower())

    def test_java_bootstrap_fails_closed_to_proven_runtime(self):
        text = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('VERSION_NAME = "3.3.1"', text)
        self.assertIn("VERSION_CODE = 300382L", text)
        self.assertIn("Unsupported ADOFAI build", text)
        self.assertIn("MobileEditorShell.installLauncher()", text)

    def test_file_selector_uses_permissionless_saf_folder_mirror(self):
        text = SELECTOR.read_text(encoding="utf-8")
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT_TREE", text)
        self.assertIn("PickerFragment extends Fragment", text)
        self.assertIn("DocumentsContract.buildChildDocumentsUriUsingTree", text)
        self.assertIn("syncSavedPath", text)
        self.assertIn("MAX_BYTES", text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("/storage/emulated/0", text)

    def test_build_reuses_canonical_editor_safety_chain_and_pinned_zygisk_api(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("scripts/prepare-upstream.sh", text)
        self.assertIn("apply_mobile_editor_document_safety.py", text)
        self.assertIn("apply_mobile_editor_json_strictness.py", text)
        self.assertIn("apply_mobile_editor_picker_serialization.py", text)
        self.assertIn("apply_mobile_editor_dirty_close_guard.py", text)
        self.assertIn("apply_zygisk_editor_mode.py", text)
        self.assertIn("apply_zygisk_native_mode.py", text)
        self.assertIn("7bb941ac8edfcffd1d23761e401c45ca95409dc1", text)
        self.assertIn("ADOFAI-3.3.1-Zygisk-Editor.zip", text)

    def test_native_transform_removes_only_java_apk_bootstrap_from_modern_profile(self):
        text = NATIVE_TRANSFORM.read_text(encoding="utf-8")
        self.assertIn("InstallAndroidFilePickerBridge(true);", text)
        self.assertIn("InstallMobileEditorPreviewBridge();", text)
        self.assertIn("Zygisk owns Java editor bootstrap", text)

    def test_editor_transform_keeps_preview_but_adds_saf_sync_and_new_chart(self):
        text = EDITOR_TRANSFORM.read_text(encoding="utf-8")
        self.assertIn('makeAction("New"', text)
        self.assertIn('makeAction("Open Map"', text)
        self.assertIn("FileSelector.syncSavedPath", text)
        self.assertNotIn("shareCurrent", text)
        self.assertNotIn('makeAction("ADOFAI / 공유"', text)

    def test_module_metadata_requires_magisk_26(self):
        text = PROP.read_text(encoding="utf-8")
        self.assertIn("id=adofai_editor_runtime", text)
        self.assertIn("minMagisk=26000", text)

    def test_workflow_builds_only_runtime_module_artifact(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("build-zygisk-runtime-module.sh", text)
        self.assertIn("adofai-zygisk-runtime-module", text)
        self.assertNotIn("game-patcher", text.lower())


if __name__ == "__main__":
    unittest.main()
