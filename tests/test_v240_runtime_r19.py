from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240WindowCompat.java"
ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
R18 = ROOT / "scripts/apply-v240-r18-window-audio-baseline.py"
R19 = ROOT / "scripts/apply-v240-r19-exact-calibration.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
FIXED_JAVA = ROOT / "scripts/build-v240-fixed-java.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


class RuntimeR19ExactCalibrationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = WINDOW.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.r18 = R18.read_text(encoding="utf-8")
        cls.r19 = R19.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.fixed_java = FIXED_JAVA.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_window_owner_is_full_width_short_edges(self):
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.window)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.window)
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.entry)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.entry)
        self.assertIn("6500L", self.entry)
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.fixed_java)
        self.assertIn("! grep -Fq 'LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER'", self.fixed_java)

    def test_repair_is_exact_game_sentinel_only(self):
        s = self.r19
        for marker in (
            'kCalibrationR19Unset = 999.0f',
            'kCalibrationR19Neutral = 0.0f',
            'playerPrefs.GetMethod("GetFloat", 2)',
            'playerPrefs.GetMethod("SetFloat", 2)',
            'playerPrefs.GetMethod("Save", 0)',
            'CreateMonoString("offset")',
            'if (before != kCalibrationR19Unset) return;',
            'setFloat.Call(key, kCalibrationR19Neutral);',
            'save.Call();',
            'if (after == kCalibrationR19Neutral)',
            'calibration-r19-playerprefs-write.pending',
            'calibrationR19Mutation=only-if-exact-999-preserve-existing',
        ):
            self.assertIn(marker, s)

    def test_r19_does_not_use_failed_r15_persistence_path(self):
        s = self.r19
        self.assertNotIn('GetField("inputOffsetNotSet")', s)
        self.assertNotIn('GetMethod("SetInputOffset"', s)
        self.assertNotIn('MaybeNeutralizeUnsetCalibration();', s)

    def test_unrelated_r18_audio_hook_is_not_installed(self):
        s = self.r19
        self.assertIn("'    MaybeInstallStartupAudioBaselineHook();\\n'", s)
        self.assertIn("'    MaybeRepairCalibrationR19();\\n'", s)
        self.assertIn("startupAudioPolicy=disabled-r19-not-device-calibration-root", s)
        self.assertIn("startupAudioMutation=disabled-r19", s)

    def test_exact_authoritative_apk_rvas_replace_old_report_values(self):
        s = self.r19
        for marker in (
            "editorProbeMetadataHandleRva=0x22E3EB0",
            "editorProbeMetadataSelectRva=0x22E7DD0",
            "editorProbeMetadataObjectsRva=0x22E8DF4",
            "editorProbeMetadataSource=authoritative-v240-apk",
        ):
            self.assertIn(marker, s)

    def test_r18_chains_r19_and_native_build_asserts_final_policy(self):
        self.assertIn("apply-v240-r19-exact-calibration.py", self.r18)
        for marker in (
            "R19_OVERLAY=",
            "stabilityRevision=19",
            "calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero",
            "startupAudioPolicy=disabled-r19-not-device-calibration-root",
            "editorProbeMetadataObjectsRva=0x22E8DF4",
        ):
            self.assertIn(marker, self.build)

    def test_channel_tracks_and_runs_r19_contract(self):
        self.assertIn("scripts/apply-v240-r19-exact-calibration.py", self.workflow)
        self.assertIn("tests/test_v240_runtime_r19.py", self.workflow)
        self.assertIn("test_v240_runtime_r19.py", self.workflow)
        self.assertIn("Build r19 full-width viewport and exact calibration persistence", self.workflow)


if __name__ == "__main__":
    unittest.main()
