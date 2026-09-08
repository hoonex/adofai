from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import adofai_compat as compat


class AdoFaiCompatTests(unittest.TestCase):
    def test_tolerates_bom_trailing_commas_and_raw_newline(self):
        raw = '\ufeff{"angleData":[0,90,],"settings":{"artist":"a\nb",},"actions":[],}'
        level = compat.parse_adofai_text(raw)
        self.assertEqual(level["angleData"], [0, 90])
        self.assertEqual(level["settings"]["artist"], "a\nb")

    def test_path_data_conversion_matches_relative_symbols(self):
        self.assertEqual(
            compat.path_data_to_angle_data("R5!6"),
            [0, 72, 999, 927],
        )

    def test_normalization_preserves_unknown_payloads(self):
        level = {
            "angleData": [0],
            "settings": {"futureSetting": {"x": 1}},
            "actions": [{"floor": 0, "eventType": "FutureEvent", "futureField": [1, 2, 3]}],
            "futureTopLevel": {"enabled": True},
        }
        normalized = compat.normalize_level(level)
        self.assertEqual(normalized["futureTopLevel"], {"enabled": True})
        self.assertEqual(normalized["actions"][0]["futureField"], [1, 2, 3])
        self.assertEqual(compat.inspect_level(normalized)["unknownModernEventTypes"], ["FutureEvent"])

    def test_fixture_loads_and_reports_modern_events(self):
        fixture = ROOT / "tests" / "fixtures" / "modern-minimal.adofai"
        level = compat.load_adofai(fixture)
        report = compat.inspect_level(level)
        self.assertEqual(report["pathEncoding"], "angleData")
        self.assertIn("SetInputEvent", report["actionTypes"])
        self.assertIn("AddParticle", report["decorationTypes"])
        self.assertEqual(report["unknownModernEventTypes"], [])


if __name__ == "__main__":
    unittest.main()
