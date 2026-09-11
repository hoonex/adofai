from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240SettingsOverlay.java"
BOOTSTRAP = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240Bootstrap.java"
BRIDGE = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240AndroidBridge.java"
PICKER = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240PickerActivity.java"
MOBILE_ACTIVITY = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240MobileXActivity.java"
SELECTOR = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/FileSelector.java"
NATIVE = ROOT / "android/v240-fixed-runtime/native/V240Fix.cpp"
TOUCH_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240TouchAssist.cpp"
NATIVE_BUILD = ROOT / "scripts/build-v240-fixed-native.sh"


class V240PerformanceRuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java = JAVA.read_text(encoding="utf-8")
        cls.bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.picker = PICKER.read_text(encoding="utf-8")
        cls.mobile_activity = MOBILE_ACTIVITY.read_text(encoding="utf-8")
        cls.selector = SELECTOR.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")
        cls.touch_native = TOUCH_NATIVE.read_text(encoding="utf-8")
        cls.native_build = NATIVE_BUILD.read_text(encoding="utf-8")

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

    def test_explicit_fps_target_is_not_clamped_to_display_refresh(self):
        self.assertIn(
            "if (requestedFps > 0) return Math.max(30, Math.min(240, requestedFps))",
            self.java,
        )
        self.assertIn("Display.Mode atLeastRequested = null", self.java)
        self.assertIn("rate + 1.0f >= requestedFps", self.java)
        self.assertIn("return atLeastRequested != null ? atLeastRequested : highest", self.java)
        self.assertNotIn("Math.min(requestedFps, max)", self.java)

    def test_overlay_bootstrap_stops_retrying_after_success_but_reinstalls_on_new_activity(self):
        self.assertIn("private static volatile boolean installed", self.java)
        self.assertIn("public static boolean isInstalled()", self.java)
        self.assertIn("View existing = root.findViewWithTag(TAG)", self.java)
        self.assertIn("public static void refresh()", self.java)
        self.assertIn("installed = true;", self.java)
        self.assertNotIn("public static void install() {\n        if (installed) return;", self.java)
        self.assertIn("if (V240SettingsOverlay.isInstalled()) return;", self.bootstrap)
        self.assertIn("if (attempts < 24) main.postDelayed(this, 250L);", self.bootstrap)

    def test_phone_and_tablet_ui_touch_scaling_is_device_adaptive(self):
        self.assertIn("deviceAdjustedUiScale", self.java)
        self.assertIn("deviceAdjustedTouchScale", self.java)
        self.assertIn("deviceTouchRadiusPx", self.java)
        self.assertIn("smallestWidthDp", self.java)
        self.assertIn("metrics.density", self.java)
        self.assertIn("extra * density", self.java)
        self.assertIn("if (smallest <= 360) boost = 1.18f", self.java)
        self.assertIn("else if (smallest < 600) boost = 1.03f", self.java)
        self.assertIn("else boost = 1.00f", self.java)

    def test_enhanced_touch_hook_expands_actual_eventsystem_hit_test_with_safe_fallback(self):
        self.assertIn("nativeApplyTouchAssist(boolean enabled, float radiusPx)", self.java)
        self.assertIn("enhancedTouch ? 1.0f : legacyTouchScale", self.java)
        self.assertIn('Class eventSystem("UnityEngine.EventSystems", "EventSystem")', self.touch_native)
        self.assertIn('eventSystem.GetMethod("RaycastAll")', self.touch_native)
        self.assertIn('input.GetMethod("get_touchCount", 0)', self.touch_native)
        self.assertIn("g_listCount[results].Get() == 0", self.touch_native)
        self.assertIn("HasActiveTouch()", self.touch_native)
        self.assertIn("IsEditorScene()", self.touch_native)
        self.assertIn("RaycastAt", self.touch_native)
        self.assertIn("g_pointerPosition[eventData].Set(original)", self.touch_native)
        self.assertIn("V240TouchAssist.cpp", self.native_build)
        self.assertIn("nativeApplyTouchAssist'", self.native_build)

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

    def test_save_sync_uses_one_debounced_task_and_immediate_terminal_events(self):
        self.assertIn("final Runnable syncTask", self.bridge)
        self.assertIn("private static final long MODIFY_DEBOUNCE_MS = 180L", self.bridge)
        self.assertIn("handler.removeCallbacks(binding.syncTask)", self.bridge)
        self.assertIn("handler.postDelayed(binding.syncTask, delayMs)", self.bridge)
        self.assertIn("scheduleSync(SaveBinding.this, MODIFY_DEBOUNCE_MS)", self.bridge)
        self.assertIn("scheduleSync(SaveBinding.this, 0L)", self.bridge)
        self.assertIn("stopBinding(existing, true)", self.bridge)
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
        self.assertIn("new byte[COPY_BUFFER_BYTES]", self.bridge)
        self.assertIn("byte[] buffer = COPY_BUFFER.get()", self.bridge)
        self.assertEqual(self.bridge.count("new byte[COPY_BUFFER_BYTES]"), 1)
        self.assertNotIn("new byte[256 * 1024]", self.bridge)

    def test_large_copy_path_does_not_double_buffer(self):
        self.assertNotIn("BufferedInputStream", self.bridge)
        self.assertNotIn("BufferedOutputStream", self.bridge)
        self.assertIn("InputStream in = requireInput", self.bridge)
        self.assertIn("OutputStream out = requireOutput", self.bridge)

    def test_picker_result_io_leaves_every_activity_main_thread(self):
        self.assertIn("static void handleResultAsync", self.bridge)
        self.assertIn("static void handleOpenResultsAsync", self.bridge)
        self.assertIn("io().post(new Runnable()", self.bridge)
        self.assertIn("V240AndroidBridge.handleOpenResultsAsync(", self.picker)
        self.assertIn("V240AndroidBridge.handleResultAsync(", self.picker)
        self.assertIn("V240AndroidBridge.handleOpenResultsAsync(", self.mobile_activity)
        self.assertIn("V240AndroidBridge.handleResultAsync(", self.mobile_activity)
        for source in (self.picker, self.mobile_activity):
            self.assertNotIn("V240AndroidBridge.handleOpen(this", source)
            self.assertNotIn("V240AndroidBridge.handleSave(this", source)
            self.assertNotIn("V240AndroidBridge.handleFolder(this", source)

    def test_multiselect_and_filter_contract_is_preserved_end_to_end(self):
        self.assertIn("Intent.EXTRA_ALLOW_MULTIPLE", self.picker)
        self.assertIn("Intent.EXTRA_ALLOW_MULTIPLE", self.mobile_activity)
        self.assertIn("getClipData()", self.picker)
        self.assertIn("getClipData()", self.mobile_activity)
        self.assertIn("selectFile(String extensions, boolean multiselect)", self.selector)
        self.assertIn("(Ljava/lang/String;Z)V", self.native)
        self.assertIn("ExtensionFilterValue", self.native)
        self.assertIn("ReadFilterExtensions", self.native)
        self.assertIn("SplitPickerPaths", self.native)
        self.assertIn("bool multiselect", self.native)

    def test_imported_assets_cannot_become_level_save_targets(self):
        self.assertIn("isLevelDocument(workingFiles.get(0))", self.bridge)
        self.assertIn('endsWith(".adofai")', self.bridge)
        self.assertIn("workingFiles.size() == 1", self.bridge)
        self.assertIn("uris.length == 1", self.bridge)

    def test_launcher_picker_survives_rotation_and_reapplies_device_policy(self):
        self.assertIn("onSaveInstanceState", self.mobile_activity)
        self.assertIn("STATE_REQUEST_ID", self.mobile_activity)
        self.assertIn("pendingRequestId = state.getInt", self.mobile_activity)
        self.assertIn("STATE_MULTI", self.mobile_activity)
        self.assertIn("pendingMulti = state.getBoolean", self.mobile_activity)
        self.assertIn("onConfigurationChanged", self.mobile_activity)
        self.assertIn("V240SettingsOverlay.refresh()", self.mobile_activity)
        self.assertIn("onWindowFocusChanged", self.mobile_activity)
        self.assertIn("if (isFinishing() && pendingRequestId > 0)", self.mobile_activity)

    def test_file_selector_waits_for_signal_instead_of_polling(self):
        self.assertIn("final CountDownLatch done = new CountDownLatch(1)", self.bridge)
        self.assertIn("result.done.await(waitMs, TimeUnit.MILLISECONDS)", self.bridge)
        self.assertIn("result.done.countDown()", self.bridge)
        self.assertIn("V240AndroidBridge.await(requestId, 600_000L)", self.selector)
        self.assertNotIn("V240AndroidBridge.poll(requestId)", self.selector)
        self.assertNotIn("Thread.sleep(80L)", self.selector)

    def test_picker_completion_is_first_terminal_result_wins(self):
        self.assertIn("if (result.state != Result.PENDING) return false", self.bridge)
        self.assertIn("final int previousRequestId = activeRequestId", self.selector)
        self.assertIn("V240AndroidBridge.cancel(previousRequestId)", self.selector)
        self.assertIn("if (generation != myGeneration) return", self.selector)

    def test_cancelled_copy_stops_and_partial_files_are_cleaned(self):
        self.assertIn("private static void ensurePending", self.bridge)
        self.assertIn("if (requestId > 0) ensurePending(requestId)", self.bridge)
        self.assertIn("deleteRecursively(working.getParentFile())", self.bridge)
        self.assertIn("deleteRecursively(mirror)", self.bridge)
        self.assertIn("MAX_TREE_DEPTH", self.bridge)
        self.assertIn("budget.bytes += copy(in, out, requestId, remaining)", self.bridge)

    def test_explicit_flush_cancels_pending_background_write_before_sync(self):
        self.assertIn("if (IO != null) IO.removeCallbacks(binding.syncTask)", self.bridge)
        self.assertIn("if (!binding.active) return true", self.bridge)
        self.assertIn("syncNow(binding)", self.bridge)
        self.assertIn("final save sync failed", self.bridge)

    def test_provider_write_mode_has_compatibility_fallbacks(self):
        self.assertIn('openOutputStream(uri, "wt")', self.bridge)
        self.assertIn('openOutputStream(uri, "w")', self.bridge)
        self.assertIn("resolver.openOutputStream(uri)", self.bridge)

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
            self.assertNotIn(marker, self.touch_native, marker)


if __name__ == "__main__":
    unittest.main()
