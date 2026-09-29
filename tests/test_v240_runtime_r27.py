from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R26 = ROOT / "scripts/apply-v240-r26-sfb-save-folder.py"
R27 = ROOT / "scripts/apply-v240-r27-hot-fps-bridge.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"
SETTINGS = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240SettingsOverlay.java"


class RuntimeR27HotFpsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r26 = R26.read_text(encoding="utf-8")
        cls.r27 = R27.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.settings = SETTINGS.read_text(encoding="utf-8")

    def test_r26_chains_r27(self):
        self.assertIn("apply-v240-r27-hot-fps-bridge.py", self.r26)
        self.assertIn(
            '__import__("subprocess").run([sys.executable, str(r27), str(path)], check=True)',
            self.r26,
        )

    def test_parent_java_contract_matches_hot_runtime_jni_exports(self):
        self.assertIn("private static native void nativeApply(", self.settings)
        self.assertIn("private static native boolean nativeApplyTouchAssist(", self.settings)
        self.assertIn("Java_com_unity3d_player_V240SettingsOverlay_nativeApply(", self.r27)
        self.assertIn("Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(", self.r27)
        self.assertIn("return JNI_FALSE;", self.r27)
        self.assertIn("enhanced-touch-unavailable-return-false-no-false-claim", self.r27)

    def test_fps_hooks_are_exact_int_static_setters_and_reversible(self):
        s = self.r27
        for marker in (
            'application.GetMethod("set_targetFrameRate", 1)',
            'quality.GetMethod("set_vSyncCount", 1)',
            "setTarget._isStatic",
            "setVsync._isStatic",
            "SameClass(Class(targetInfo->parameters[0]), intClass)",
            "SameClass(Class(vsyncInfo->parameters[0]), intClass)",
            "TypeCode(targetInfo->return_type) == 1",
            "TypeCode(vsyncInfo->return_type) == 1",
            "BasicHook(setTarget, HookFpsR27SetTarget, g_oldFpsR27SetTarget)",
            "BasicHook(setVsync, HookFpsR27SetVsync, g_oldFpsR27SetVsync)",
            "g_fpsR27LastRequestedTarget.store(fps)",
            "g_fpsR27LastRequestedVsync.store(count)",
            "? desiredTarget : requestedTarget",
            "g_fpsR27DesiredLowLatency.load() ? 0 : requestedVsync",
        ):
            self.assertIn(marker, s)

    def test_fps_hooks_are_self_fused_and_fail_closed(self):
        s = self.r27
        for marker in (
            "fps-r27-install.pending",
            "fps-r27-target-call.pending",
            "fps-r27-vsync-call.pending",
            "fps-r27-apply.pending",
            "PrepareFpsR27Fuse()",
            "g_fpsR27RecoveryState",
            "WriteMarker(g_fpsR27InstallMarker)",
            "WriteMarker(g_fpsR27TargetCallMarker)",
            "WriteMarker(g_fpsR27VsyncCallMarker)",
            "WriteMarker(g_fpsR27ApplyMarker)",
        ):
            self.assertIn(marker, s)
        self.assertIn('s.count("BasicHook(") != 18', s)

    def test_final_diagnostics_count_only_active_hook_sites(self):
        s = self.r27
        self.assertIn("(g_fpsR27HookInstalled.load() ? 2 : 0)", s)
        self.assertIn(
            "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2",
            s,
        )
        self.assertIn("fpsR27Revision=27", s)
        self.assertIn("fpsR27PolicyApplyCalls=", s)

    def test_build_and_channel_require_r27_symbols_and_markers(self):
        for marker in (
            "R27_OVERLAY",
            "fpsR27Revision=27",
            "fpsR27Policy=hot-cache-Application-targetFrameRate-QualitySettings-vSync-reversible",
            "Java_com_unity3d_player_V240SettingsOverlay_nativeApply",
            "Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist",
            "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2",
        ):
            self.assertIn(marker, self.build)
        self.assertIn("scripts/apply-v240-r27-hot-fps-bridge.py", self.workflow)
        self.assertIn("tests/test_v240_runtime_r27.py", self.workflow)
        self.assertIn("Build r23 exact editor raycast and calibration persistence", self.workflow)


if __name__ == "__main__":
    unittest.main()
