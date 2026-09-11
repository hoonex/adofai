from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240SettingsOverlay.java"
NATIVE = ROOT / "android/v240-fixed-runtime/native/V240Fix.cpp"


class V240PerformanceRuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java = JAVA.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")

    def test_android_refresh_policy_is_explicit_and_reversible(self):
        self.assertIn("preferredDisplayModeId = mode.getModeId()", self.java)
        self.assertIn("preferredRefreshRate = mode.getRefreshRate()", self.java)
        self.assertIn("preferredDisplayModeId = 0", self.java)
        self.assertIn("preferredRefreshRate = 0f", self.java)
        self.assertIn("60, 90, 120, 144, 165, 240", self.java)

    def test_native_fps_policy_intercepts_only_render_pacing(self):
        self.assertIn('Class application("UnityEngine", "Application")', self.native)
        self.assertIn('GetMethod("set_targetFrameRate", 1)', self.native)
        self.assertIn('Class qualitySettings("UnityEngine", "QualitySettings")', self.native)
        self.assertIn('GetMethod("set_vSyncCount", 1)', self.native)
        self.assertIn("g_framePolicyConfigured", self.native)

    def test_frame_policy_fails_open_until_settings_arrive(self):
        self.assertIn("if (!g_framePolicyConfigured.load", self.native)
        self.assertIn("g_oldSetTargetFrameRate(fps)", self.native)
        self.assertIn("g_oldSetVSyncCount(count)", self.native)
        self.assertIn("g_framePolicyConfigured.store(true", self.native)

    def test_editor_scene_checks_are_cached(self):
        self.assertIn("kEditorSceneCacheNs", self.native)
        self.assertIn("g_editorSceneCacheAtNs", self.native)
        self.assertIn("IsDragAxis(axis) && IsEditorScene()", self.native)

    def test_rhythm_timing_is_not_modified(self):
        forbidden = (
            "timeScale",
            "fixedDeltaTime",
            "captureFramerate",
            "maximumDeltaTime",
            "DSPTime",
        )
        for marker in forbidden:
            self.assertNotIn(marker, self.native, marker)


if __name__ == "__main__":
    unittest.main()
