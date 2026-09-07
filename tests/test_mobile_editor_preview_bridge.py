from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "native" / "MobileEditorBridge.cpp"


class MobileEditorPreviewBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = BRIDGE.read_text(encoding="utf-8")

    def test_required_runtime_surface_is_resolved_before_gcs_mutation(self):
        text = self.text
        mutation = text.index("customLevelIndex.cast<int>().Set(0);")

        required_before_mutation = (
            'auto controllerClass = Class("", "scrController");',
            'auto getInstance = controllerClass.GetMethod("get_instance", 0);',
            'auto loadCustomLevel = controllerClass.GetMethod("LoadCustomLevel", 3);',
            'auto controller = getInstance.cast<IL2CPP::Il2CppObject*>().Call();',
            'String* levelPath = CreateMonoString(path);',
            'String* gameScene = CreateMonoString("scnGame");',
            'if (!levelPath || !gameScene)',
        )
        for marker in required_before_mutation:
            with self.subTest(marker=marker):
                self.assertLess(text.index(marker), mutation)

    def test_validation_failures_return_before_any_global_write(self):
        text = self.text
        mutation = text.index("customLevelIndex.cast<int>().Set(0);")
        validation_prefix = text[:mutation]
        self.assertNotIn(".Set(", validation_prefix)
        self.assertIn("required GCS fields are missing", validation_prefix)
        self.assertIn("current custom-level methods are missing", validation_prefix)
        self.assertIn("scrController instance is null", validation_prefix)
        self.assertIn("required managed strings could not be created", validation_prefix)

    def test_scene_string_is_prebuilt_and_reused_for_mutation(self):
        text = self.text
        self.assertIn('String* gameScene = CreateMonoString("scnGame");', text)
        self.assertIn("sceneToLoad.cast<String*>().Set(gameScene);", text)
        self.assertNotIn('sceneToLoad.cast<String*>().Set(CreateMonoString("scnGame"));', text)

    def test_preview_queue_rejects_pending_or_dispatching_overlap(self):
        text = self.text
        self.assertIn("bool g_previewDispatchInFlight = false;", text)
        self.assertIn(
            "if (g_previewDispatchInFlight || !g_pendingPreviewPath.empty()) {",
            text,
        )
        self.assertIn(
            'LOGW("Mobile editor preview rejected: another preview is pending or dispatching");',
            text,
        )
        take = text.index("bool TakePendingPreview")
        finish = text.index("void FinishPreviewDispatch")
        self.assertIn("g_previewDispatchInFlight = true;", text[take:finish])
        self.assertIn("g_previewDispatchInFlight = false;", text[finish:])

    def test_dispatch_owner_is_released_after_runtime_attempt(self):
        text = self.text
        drain = text[text.index("void DrainPreviewQueueOnGameThread") : text.index("int HookedTouchCount")]
        attempt = drain.index("bool succeeded = SetRequiredPreviewState(path);")
        release = drain.index("FinishPreviewDispatch();")
        failure = drain.index("if (!succeeded)")
        self.assertLess(attempt, release)
        self.assertLess(release, failure)

    def test_preview_queue_still_drains_on_game_thread_hook(self):
        text = self.text
        self.assertIn("DrainPreviewQueueOnGameThread();", text)
        self.assertIn("BasicHook(touchCount, HookedTouchCount, g_oldTouchCount);", text)
        self.assertIn("Mobile editor preview request failed closed", text)


if __name__ == "__main__":
    unittest.main()
