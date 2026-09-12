from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOUCH_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240TouchAssist.cpp"


class V240EventRuntimeProbeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TOUCH_NATIVE.read_text(encoding="utf-8")
        start = cls.source.index("void LogPostV240CompatibilitySurface()")
        end = cls.source.index("bool RaycastAt(", start)
        cls.probe = cls.source[start:end]

    def test_probe_covers_lossless_unknown_event_bridge_surfaces(self):
        expected = (
            'Class levelData("ADOFAI", "LevelData")',
            'Class levelEvent("ADOFAI", "LevelEvent")',
            'Class levelEventInfo("ADOFAI", "LevelEventInfo")',
            'Class gcs("", "GCS")',
            'levelData.GetMethod("Decode")',
            'levelEvent.GetMethod("Decode")',
            'levelEvent.GetMethod("Encode")',
            'gcs.GetField("levelEventsInfo")',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_probe_covers_frame_event_scheduler_candidates(self):
        self.assertIn('Class scnGame("", "scnGame")', self.probe)
        self.assertIn('scnGame.GetMethod("ApplyEvent")', self.probe)
        self.assertIn('Class scrCamera("", "scrCamera")', self.probe)
        self.assertIn('"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}', self.probe)
        self.assertIn('"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<float>()}', self.probe)
        self.assertIn('setCustomFrameRateTyped = setCustomFrameRateBoolInt || setCustomFrameRateBoolFloat', self.probe)

    def test_frame_rate_probe_requires_exact_primitive_abi(self):
        self.assertNotIn('scrCamera.GetMethod("SetCustomFrameRate").IsValid()', self.probe)
        self.assertIn('bool-int=%d bool-float=%d', self.probe)
        self.assertIn('setCustomFrameRateBoolInt ? 1 : 0', self.probe)
        self.assertIn('setCustomFrameRateBoolFloat ? 1 : 0', self.probe)

    def test_probe_is_read_only(self):
        self.assertNotIn("BasicHook", self.probe)
        self.assertNotIn("CreateNewObject", self.probe)
        self.assertNotIn(".Call(", self.probe)
        self.assertNotIn(".Set(", self.probe)
        self.assertNotIn("operator[]", self.probe)

    def test_probe_runs_only_after_bnm_loaded_event(self):
        callback = self.source.index("Loading::AddOnLoadedEvent")
        probe_call = self.source.index("LogPostV240CompatibilitySurface();", callback)
        touch_install = self.source.index("InstallTouchAssistHook()", probe_call)
        self.assertLess(callback, probe_call, touch_install)

    def test_runtime_log_has_stable_machine_searchable_marker(self):
        self.assertIn("V240: compatibility surface LevelData=%d", self.probe)
        self.assertIn("scrCamera.SetCustomFrameRate=%d", self.probe)
        self.assertIn("bool-int=%d bool-float=%d", self.probe)


if __name__ == "__main__":
    unittest.main()
