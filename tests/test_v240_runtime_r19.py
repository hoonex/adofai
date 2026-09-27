from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
R18 = ROOT / "scripts/apply-v240-r18-window-audio-baseline.py"
R19 = ROOT / "scripts/apply-v240-r19-exact-calibration.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"

class RuntimeR19SupersededContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r18 = R18.read_text(encoding="utf-8")
        cls.r19 = R19.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_r19_is_explicitly_superseded_by_r20(self):
        self.assertIn("R20 supersedes this file", self.r19)
        self.assertIn("Persistence -> PlayerPrefsJson, not UnityEngine.PlayerPrefs", self.r19)
        self.assertIn("apply-v240-r20-playerprefsjson-calibration.py", self.r19)

    def test_r19_retains_authoritative_editor_rvas_as_history(self):
        for marker in (
            "editorProbeMetadataHandleRva=0x22E3EB0",
            "editorProbeMetadataSelectRva=0x22E7DD0",
            "editorProbeMetadataObjectsRva=0x22E8DF4",
            "editorProbeMetadataSource=authoritative-v240-apk",
        ):
            self.assertIn(marker, self.r19)

    def test_r18_chains_r19(self):
        self.assertIn("apply-v240-r19-exact-calibration.py", self.r18)

    def test_build_and_channel_track_successor(self):
        self.assertIn("R20_OVERLAY=", self.build)
        self.assertIn("scripts/apply-v240-r20-playerprefsjson-calibration.py", self.workflow)

if __name__ == "__main__":
    unittest.main()
