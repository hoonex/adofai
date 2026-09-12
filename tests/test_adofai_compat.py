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
        v240 = report["v240Compatibility"]
        self.assertTrue(v240["requiresRuntimeReview"])
        self.assertIn("SetInputEvent", v240["preserveOnlyEventTypes"])
        self.assertIn("AddParticle", v240["preserveOnlyEventTypes"])
        self.assertIn("SetInputEvent", v240["gameplayMeaningRiskEventTypes"])
        self.assertFalse(v240["unknownEventsAreDeleted"])

    def test_set_frame_rate_is_candidate_not_claimed_supported(self):
        level = {
            "angleData": [0, 90],
            "settings": {},
            "actions": [
                {"floor": 1, "eventType": "SetFrameRate", "frameRate": 144},
            ],
            "decorations": [],
        }
        v240 = compat.v240_compatibility_report(level)
        self.assertEqual(v240["nativeEmulationCandidates"], ["SetFrameRate"])
        event = v240["postV240Events"][0]
        self.assertEqual(event["introduced"], "2.8.0")
        self.assertEqual(event["domain"], "rendering")
        self.assertFalse(event["runtimeBackportImplemented"])
        self.assertNotIn("SetFrameRate", v240["preserveOnlyEventTypes"])

    def test_gameplay_events_are_preserved_not_auto_downgraded(self):
        level = {
            "angleData": [0],
            "settings": {},
            "actions": [
                {
                    "floor": 0,
                    "eventType": "SetInputEvent",
                    "inputAction": "Probe",
                    "inputEventState": "Subscribe",
                    "inputEventTarget": "Pressed",
                },
                {"floor": 0, "eventType": "TileDimensions", "futureGeometry": [2, 1]},
            ],
            "decorations": [],
        }
        normalized = compat.normalize_level(level)
        self.assertEqual(normalized["actions"], level["actions"])
        v240 = compat.v240_compatibility_report(normalized)
        self.assertEqual(
            v240["gameplayMeaningRiskEventTypes"],
            ["SetInputEvent", "TileDimensions"],
        )
        self.assertEqual(
            v240["preserveOnlyEventTypes"],
            ["SetInputEvent", "TileDimensions"],
        )

    def test_later_semantics_are_reported_without_guessing_fields(self):
        level = {
            "angleData": [0],
            "settings": {},
            "actions": [
                {"floor": 0, "eventType": "FreeRoam", "duration": 4},
                {"floor": 0, "eventType": "RepeatEvents", "opaqueFutureGap": 3},
                {"floor": 0, "eventType": "RecolorTrack", "opaqueFutureTexture": "x"},
            ],
            "decorations": [],
        }
        normalized = compat.normalize_level(level)
        self.assertEqual(normalized["actions"], level["actions"])
        v240 = compat.v240_compatibility_report(normalized)
        drift = {item["eventType"]: item for item in v240["semanticDriftEvents"]}
        self.assertIn("FreeRoam", drift)
        self.assertIn("RepeatEvents", drift)
        self.assertIn("RecolorTrack", drift)
        self.assertEqual(drift["RepeatEvents"]["policy"], "preserve_and_verify")

    def test_first_floor_events_are_reported_without_being_moved(self):
        level = {
            "angleData": [0, 90],
            "settings": {},
            "actions": [
                {"floor": 0, "eventType": "MoveCamera", "position": [4, 5]},
                {"floor": 1, "eventType": "Twirl"},
            ],
            "decorations": [
                {"floor": 0, "eventType": "AddDecoration", "tag": "start"},
            ],
        }
        normalized = compat.normalize_level(level)
        self.assertEqual(normalized["actions"][0]["floor"], 0)
        self.assertEqual(normalized["decorations"][0]["floor"], 0)

        v240 = compat.v240_compatibility_report(normalized)
        self.assertEqual(v240["firstFloorEventCount"], 2)
        self.assertTrue(v240["requiresRuntimeReview"])
        self.assertEqual(
            [(item["section"], item["index"], item["eventType"]) for item in v240["structuralRisks"]],
            [
                ("actions", 0, "MoveCamera"),
                ("decorations", 0, "AddDecoration"),
            ],
        )
        self.assertTrue(all(item["policy"] == "preserve_and_verify" for item in v240["structuralRisks"]))

    def test_post_v240_fixture_survives_normalization_losslessly(self):
        fixture = ROOT / "tests" / "fixtures" / "post-v240-semantics.adofai"
        level = compat.load_adofai(fixture)
        normalized = compat.normalize_level(level)
        self.assertEqual(normalized, level)

        report = compat.inspect_level(normalized)
        v240 = report["v240Compatibility"]
        self.assertEqual(v240["nativeEmulationCandidates"], ["SetFrameRate"])
        self.assertIn("SetInputEvent", v240["preserveOnlyEventTypes"])
        self.assertIn("AddParticle", v240["preserveOnlyEventTypes"])
        self.assertEqual(v240["firstFloorEventCount"], 2)

        actions = {item["eventType"]: item for item in normalized["actions"]}
        self.assertEqual(actions["SetFrameRate"]["frameRate"], 144)
        self.assertTrue(actions["SetFrameRate"]["editorOnly"])
        self.assertEqual(actions["RepeatEvents"]["gapLength"], 2)
        self.assertEqual(actions["RepeatEvents"]["futureRepeatField"], "preserve")
        self.assertEqual(actions["RecolorTrack"]["texture"], "fixture.png")
        self.assertEqual(actions["RecolorTrack"]["futureRecolorField"], [1, 2, 3])
        self.assertEqual(normalized["futureTopLevel"], {"mustSurvive": True})


if __name__ == "__main__":
    unittest.main()
