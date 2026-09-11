from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240SettingsOverlay.java"
BOOTSTRAP = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240Bootstrap.java"
BRIDGE = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240AndroidBridge.java"
NATIVE = ROOT / "android/v240-fixed-runtime/native/V240Fix.cpp"


class V240PerformanceRuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java = JAVA.read_text(encoding="utf-8")
        cls.bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")

    def test_android_refresh_policy_is_explicit_reversible_and_low_latency(self):
        self.assertIn("private static void applyWindowPolicy", self.java)
        self.assertIn("int targetModeId = mode != null ? mode.getModeId() : 0", self.java)
        self.assertIn("float targetRefreshRate = mode != null ? mode.getRefreshRate() : 0f", self.java)
        self.assertIn("params.preferredDisplayModeId = targetModeId", self.java)
        self.assertIn("params.preferredRefreshRate = targetRefreshRate", self.java)
        self.assertIn("Build.VERSION.SDK_INT >= 30", self.java)
        self.assertIn("params.preferMinimalPostProcessing = lowLatency", self.java)
        self.assertIn("if (changed) owner.getWindow().setAttributes(params)", self.java)
        self.assertIn("60, 90, 120, 144, 165, 240", self.java)

    def test_overlay_bootstrap_stops_retrying_after_success(self):
        self.assertIn("private static volatile boolean installed", self.java)
        self.assertIn("public static boolean isInstalled()", self.java)
        self.assertIn("if (installed) return;", self.java)
        self.assertIn("installed = true;", self.java)
        self.assertIn("if (V240SettingsOverlay.isInstalled()) return;", self.bootstrap)
        self.assertIn("if (attempts < 24) main.postDelayed(this, 250L);", self.bootstrap)
        self.assertLess(
            self.java.index("pushNative(owner, owner.getSharedPreferences"),
            self.java.index("installed = true;", self.java.index("pushNative(owner, owner.getSharedPreferences")),
        )

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

    def test_native_input_string_checks_do_not_allocate_utf8_strings(self):
        self.assertIn("MonoStringEqualsAscii", self.native)
        self.assertIn("MonoStringStartsWithAscii", self.native)
        self.assertIn('MonoStringEqualsAscii(axis, "Mouse X", 7)', self.native)
        self.assertIn('MonoStringEqualsAscii(axis, "Mouse Y", 7)', self.native)
        self.assertIn('MonoStringEqualsAscii(axis, "Mouse ScrollWheel", 17)', self.native)
        self.assertIn('MonoStringStartsWithAscii(g_getSceneName.Call(), "scnEditor", 9)', self.native)
        self.assertNotIn("const std::string value = axis->str()", self.native)
        self.assertNotIn("const std::string value = name->str()", self.native)

    def test_touch_assist_reuses_raycast_work_objects(self):
        self.assertIn("Method<void> g_listClear", self.native)
        self.assertIn("RaycastUiWithContext", self.native)
        self.assertIn("g_listClear[results].Call()", self.native)
        self.assertIn('g_listRaycastResultClass.GetMethod("Clear", 0)', self.native)
        self.assertEqual(self.native.count("CreateNewObjectParameters(eventSystem)"), 1)
        self.assertEqual(self.native.count("g_listRaycastResultClass.CreateNewObjectParameters()"), 1)
        self.assertNotIn("bool RaycastUi(Vector2 point)", self.native)

    def test_save_sync_uses_single_debounced_task(self):
        self.assertIn("final Runnable syncTask", self.bridge)
        self.assertIn("handler.removeCallbacks(binding.syncTask)", self.bridge)
        self.assertIn("handler.postDelayed(binding.syncTask, 180L)", self.bridge)
        self.assertIn("stopBinding(existing)", self.bridge)
        self.assertIn("binding.observer.stopWatching()", self.bridge)
        self.assertNotIn("volatile long generation", self.bridge)

    def test_storage_worker_is_lazy_and_background_priority(self):
        self.assertIn("private static volatile HandlerThread IO_THREAD", self.bridge)
        self.assertIn("private static volatile Handler IO", self.bridge)
        self.assertIn("private static Handler io()", self.bridge)
        self.assertIn('"adofai-v240-storage", Process.THREAD_PRIORITY_BACKGROUND', self.bridge)
        self.assertNotIn(
            'private static final HandlerThread IO_THREAD = new HandlerThread("adofai-v240-storage")',
            self.bridge,
        )
        self.assertNotIn("static {\n        IO_THREAD.start();", self.bridge)

    def test_storage_copy_buffer_is_reused_per_thread(self):
        self.assertIn("private static final ThreadLocal<byte[]> COPY_BUFFER", self.bridge)
        self.assertIn("return new byte[COPY_BUFFER_BYTES]", self.bridge)
        self.assertIn("byte[] buffer = COPY_BUFFER.get()", self.bridge)
        self.assertEqual(self.bridge.count("new byte[COPY_BUFFER_BYTES]"), 1)
        self.assertNotIn("new byte[256 * 1024]", self.bridge)

    def test_large_copy_path_does_not_double_buffer(self):
        self.assertNotIn("BufferedInputStream", self.bridge)
        self.assertNotIn("BufferedOutputStream", self.bridge)
        self.assertIn("InputStream in = requireInput", self.bridge)
        self.assertIn("OutputStream out = requireOutput", self.bridge)

    def test_explicit_flush_cancels_pending_background_write(self):
        self.assertIn("if (IO != null) IO.removeCallbacks(binding.syncTask);\n                syncNow(binding);", self.bridge)

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
