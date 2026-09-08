from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android" / "editor-harness" / "app"
MANIFEST = APP / "src" / "main" / "AndroidManifest.xml"
MAIN = APP / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "editorharness" / "MainActivity.java"
SELECTOR = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "FileSelector.java"
BUNDLE = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "BundleWorkspace.java"
SERVER = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "LoopbackZipServer.java"
BRIDGE = APP / "src" / "main" / "java" / "com" / "unity3d" / "player" / "OfficialGameBridge.java"
PROVIDER = APP / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "companion" / "OfficialChartProvider.java"
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
        self.assertNotIn("ADOFAI Custom", text)

    def test_manifest_supports_url_and_zip_bundle_inputs_without_raw_storage(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn('android.permission.INTERNET', text)
        self.assertIn('android:label="ADOFAI Companion Editor"', text)
        self.assertIn('android.intent.action.VIEW', text)
        self.assertIn('application/zip', text)
        self.assertIn('application/x-zip-compressed', text)
        self.assertIn('OfficialChartProvider', text)
        self.assertIn('com.fizzd.connectedworlds', text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("READ_EXTERNAL_STORAGE", text)
        self.assertNotIn("WRITE_EXTERNAL_STORAGE", text)
        self.assertNotIn("requestLegacyExternalStorage", text)
        self.assertNotIn('android:name=".PlayerActivity"', text)

    def test_saf_bridge_imports_zip_or_direct_chart_and_repackages_bundle(self):
        text = SELECTOR.read_text(encoding="utf-8")
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn("Intent.ACTION_CREATE_DOCUMENT", text)
        self.assertIn("takePersistableUriPermission", text)
        self.assertIn("BundleWorkspace.importZip", text)
        self.assertIn("BundleWorkspace.importZipUrl", text)
        self.assertIn("BundleWorkspace.packageBundle", text)
        self.assertIn("BUNDLE_URI_BY_PATH", text)
        self.assertIn("selectUrlBundle", text)
        self.assertIn("syncSavedPath", text)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", text)
        self.assertNotIn("CustomFileChooser", text)

    def test_bundle_workspace_preserves_sibling_assets_and_rejects_zip_slip(self):
        text = BUNDLE.read_text(encoding="utf-8")
        self.assertIn("ZipInputStream", text)
        self.assertIn("ZipOutputStream", text)
        self.assertIn('name.equals("main.adofai")', text)
        self.assertIn("MAX_ENTRIES", text)
        self.assertIn("MAX_DOWNLOAD_BYTES", text)
        self.assertIn("MAX_EXTRACTED_BYTES", text)
        self.assertIn("getCanonicalFile", text)
        self.assertIn("ZIP path traversal rejected", text)
        self.assertIn("zipDirectory(root, output)", text)

    def test_loopback_zip_server_is_local_only_and_serves_zip(self):
        text = SERVER.read_text(encoding="utf-8")
        self.assertIn('InetAddress.getByName("127.0.0.1")', text)
        self.assertIn("new ServerSocket(0", text)
        self.assertIn('"application/zip"', text)
        self.assertIn('"GET".equals(method)', text)
        self.assertIn('"HEAD".equals(method)', text)
        self.assertIn("Cache-Control: no-store", text)

    def test_official_bridge_prefers_historical_zip_url_shape_and_keeps_chart_uri_fallback(self):
        bridge = BRIDGE.read_text(encoding="utf-8")
        provider = PROVIDER.read_text(encoding="utf-8")
        self.assertIn('new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY)', bridge)
        self.assertIn('Intent.FLAG_GRANT_READ_URI_PERMISSION', bridge)
        self.assertIn('EXPECTED_VERSION_CODE = 300382L', bridge)
        self.assertIn('BundleWorkspace.packageBundle', bridge)
        self.assertIn('LoopbackZipServer.publish', bridge)
        self.assertIn('setDataAndType(bundleUri, "application/zip")', bridge)
        self.assertIn('intent.putExtra("url", bundleUrl)', bridge)
        self.assertIn('intent.putExtra("levelUrl", bundleUrl)', bridge)
        self.assertIn('Intent.EXTRA_TEXT', bridge)
        self.assertIn('OfficialChartProvider.publish', bridge)
        self.assertIn('Intent.EXTRA_STREAM', bridge)
        self.assertIn('MODE_READ_ONLY', provider)

    def test_gradle_uses_durable_companion_identity(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("java.srcDirs += ['generated/java']", text)
        self.assertIn("applicationId 'dev.hoonex.adofai.companion'", text)
        self.assertIn("namespace 'dev.hoonex.adofai.companion'", text)
        self.assertIn("compileSdk 35", text)

    def test_prepare_script_keeps_safety_chain_then_bundle_companion_only(self):
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
        self.assertNotIn("apply_custom_game_mode.py", text)
        self.assertNotIn("prepare-upstream.sh", text)
        self.assertNotIn("CustomFileChooser", text)
        self.assertIn("Intent.ACTION_OPEN_DOCUMENT", text)
        self.assertIn('makeAction("ZIP URL"', text)
        self.assertIn("BundleWorkspace.importZip", text)
        self.assertIn("LoopbackZipServer.publish", text)
        self.assertIn("OfficialGameBridge.open", text)

    def test_companion_layer_has_zip_url_new_save_sync_and_official_handoff(self):
        companion = COMPANION.read_text(encoding="utf-8")
        self.assertIn('makeAction("New"', companion)
        self.assertIn('makeAction("ZIP URL"', companion)
        self.assertIn('FileSelector.selectUrlBundle()', companion)
        self.assertIn('makeAction("공식 ADOFAI"', companion)
        self.assertIn("FileSelector.syncSavedPath", companion)
        self.assertIn("FileSelector.displayNameForPath", companion)
        self.assertIn("openStandalonePath", companion)
        self.assertIn("OfficialGameBridge.open(currentPath)", companion)
        self.assertNotIn("CustomPlayerBridge", companion)


if __name__ == "__main__":
    unittest.main()
