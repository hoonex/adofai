from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240WindowCompat.java"
ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
R17 = ROOT / "scripts/apply-v240-r17-editor-physics-sync.py"
R18 = ROOT / "scripts/apply-v240-r18-window-audio-baseline.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR18StabilityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = WINDOW.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.r17 = R17.read_text(encoding="utf-8")
        cls.r18 = R18.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_parent_window_policy_preserves_full_width_on_cutout_phones(self):
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.window)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.window)
        self.assertIn("Input.mousePosition/Screen coordinates", self.window)

    def test_hot_runtime_reasserts_full_width_after_parent_bootstrap_callbacks(self):
        s = self.entry
        for marker in (
            "scheduleLegacyWindowNormalization();",
            "normalizeLegacyWindowViewport()",
            "LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES",
            "1750L",
            "3000L",
            "decor.requestLayout()",
        ):
            self.assertIn(marker, s)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", s)

    def test_r18_restores_original_editor_selection_path(self):
        s = self.r18
        for marker in (
            "activeTilePolicy=original-v240-editor-path-plus-full-width-window",
            "windowViewportPolicy=short-edges-full-width",
            "editorProbePolicy=disabled-r18-original-editor-path-window-viewport-fix",
            "editorObjectsPolicy=disabled-r18-original-editor-path-window-viewport-fix",
            "editorPhysicsPolicy=disabled-r18-window-viewport-fix",
        ):
            self.assertIn(marker, s)
        for forbidden in (
            "'    MaybeInstallEditorProbe();\\n'",
            "'    MaybeInstallEditorObjectsProbe();\\n'",
            "'    MaybeInstallEditorPhysicsSync();\\n'",
            "'    MaybeNeutralizeUnsetCalibration();\\n'",
        ):
            self.assertIn(forbidden, s)
        self.assertIn("if forbidden_call in s:", s)

    def test_startup_audio_fix_is_exact_and_does_not_write_calibration_preferences(self):
        s = self.r18
        for marker in (
            'controller.GetMethod("Start", 0)',
            'controller.GetMethod("CheckForAudioOutputChange", 0)',
            'conductor.GetMethod("UpdateCurrentAudioOutput", 0)',
            "g_controllerStartDepth > 0",
            "g_updateCurrentAudioOutput.Call();",
            "g_oldCheckForAudioOutputChange(self, methodInfo);",
            "startupAudioMutation=runtime-baseline-only-no-PlayerPrefs",
        ):
            self.assertIn(marker, s)
        self.assertNotIn('SetInputOffset', s)
        self.assertNotIn('Class playerPrefs', s)
        self.assertNotIn('playerPrefs.GetMethod', s)

    def test_startup_audio_hook_is_self_fused_and_fail_open(self):
        s = self.r18
        for marker in (
            "startup-audio-r18-install.pending",
            "startup-audio-r18-call.pending",
            "startup-audio-r18-probe.tmp",
            "WriteMarker(g_startupAudioInstallMarker)",
            "WriteMarker(g_startupAudioCallMarker)",
            "BasicHook(check, HookCheckForAudioOutputChange, g_oldCheckForAudioOutputChange)",
            "BasicHook(start, HookControllerStart, g_oldControllerStart)",
            "TypeCode(startInfo->return_type) == 1",
            "TypeCode(checkInfo->return_type) == 1",
            "TypeCode(updateInfo->return_type) == 1",
        ):
            self.assertIn(marker, s)

    def test_r17_chains_r18_and_build_validates_final_active_policy(self):
        self.assertIn("apply-v240-r18-window-audio-baseline.py", self.r17)
        for marker in (
            "R18_OVERLAY=",
            "stabilityRevision=18",
            "activeTilePolicy=original-v240-editor-path-plus-full-width-window",
            "startupAudioPolicy=first-check-inside-scrController-Start-baseline-silent-then-original",
            "calibrationExecution=disabled-r16-after-r15-self-fuse-recovery",
        ):
            self.assertIn(marker, self.build)

    def test_channel_runs_r18_contract(self):
        self.assertIn("scripts/apply-v240-r18-window-audio-baseline.py", self.workflow)
        self.assertIn("test_v240_runtime_r18.py", self.workflow)
        self.assertIn("Build r18 window viewport and startup audio baseline", self.workflow)


if __name__ == "__main__":
    unittest.main()
