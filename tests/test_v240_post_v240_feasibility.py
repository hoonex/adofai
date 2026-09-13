from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"
SCANNER = JAVA / "V240ChartCompatibilityScanner.java"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"
EVENT_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"


class V240PostV240FeasibilityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.event_native = EVENT_NATIVE.read_text(encoding="utf-8")

    def test_unproven_execution_families_are_explicitly_preserve_only(self):
        start = self.scanner.index("PRESERVE_ONLY_POST_V240")
        end = self.scanner.index("));", start)
        preserve_only = self.scanner[start:end]
        for event_type in (
            "SetInputEvent",
            "SetFilterAdvanced",
            "AddParticle",
            "SetParticle",
            "EmitParticle",
            "TileDimensions",
        ):
            self.assertIn('"' + event_type + '"', preserve_only)
        self.assertNotIn('"SetFrameRate"', preserve_only)

    def test_preserve_only_classification_is_diagnostic_not_destructive(self):
        self.assertIn("final Set<String> preserveOnlyPostV240", self.scanner)
        self.assertIn("result.preserveOnlyPostV240.add(eventType)", self.scanner)
        self.assertIn('" preserve-only=" + result.preserveOnlyPostV240', self.scanner)
        self.assertIn("Set<String> targetTypes = scan.postV240", self.opaque)
        self.assertIn('writeSmallFile(new File(session, token + ".json"), eventObject)', self.opaque)
        self.assertIn("V240SetFrameRateBackport.maybePlaceholder", self.opaque)

    def test_no_guessed_active_backport_exists_for_unproven_families(self):
        combined = self.opaque + "\n" + self.event_native
        for guessed_surface in (
            "V240SetInputEventBackport",
            "V240SetFilterAdvancedBackport",
            "V240ParticleBackport",
            "__V240_SET_INPUT_EVENT__:",
            "__V240_SET_FILTER_ADVANCED__:",
            "__V240_PARTICLE__:",
            'Class ffxSetInputEvent(',
            'Class ffxSetFilterAdvanced(',
            'Class ffxAddParticle(',
            'Class ffxSetParticle(',
            'Class ffxEmitParticle(',
        ):
            self.assertNotIn(guessed_surface, combined)

    def test_set_frame_rate_remains_the_only_proven_post_v240_execution_exception(self):
        self.assertIn("V240SetFrameRateBackport.maybePlaceholder", self.opaque)
        self.assertIn("__V240_SET_FRAME_RATE__:", self.event_native)
        self.assertIn("g_setFrameRateBackportReady", self.event_native)


if __name__ == "__main__":
    unittest.main()
