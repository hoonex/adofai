from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "android/v240-fixed-runtime/native/V240PostV240Probe.cpp"
EVENT_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"
EVENT_JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240EventCompat.java"
NATIVE_BUILD = ROOT / "scripts/build-v240-fixed-native.sh"


class V240PostV240RuntimeProbeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = PROBE.read_text(encoding="utf-8")
        cls.event_native = EVENT_NATIVE.read_text(encoding="utf-8")
        cls.event_java = EVENT_JAVA.read_text(encoding="utf-8")
        cls.native_build = NATIVE_BUILD.read_text(encoding="utf-8")

    def test_probe_is_strictly_read_only(self):
        for forbidden in (".Call(", ".Set(", "BasicHook", "CreateNewObject"):
            self.assertNotIn(forbidden, self.probe)
        self.assertNotIn("__V240_SET_INPUT_EVENT__:", self.probe)
        self.assertNotIn("__V240_PARTICLE__:", self.probe)
        self.assertNotIn("__V240_SET_FILTER_ADVANCED__:", self.probe)

    def test_set_input_event_probe_tracks_scheduler_without_claiming_execution(self):
        expected = (
            'Class ffxPlusBase("", "ffxPlusBase")',
            'Class scrController("", "scrController")',
            'Class inputEventTarget("", "InputEventTarget")',
            'Class inputEventState("", "InputEventState")',
            'Class ffxSetInputEventPlus("", "ffxSetInputEventPlus")',
            '"StartEffectWithOffset", {scrPlanet.GetCompileTimeClass()}',
            'scrController.GetMethod("ResetInputEventFfx", 0)',
            'scrController.GetField("inputEventFfx")',
            'SameManagedType(startEffectWithOffset.GetReturnType(), voidClass)',
            'SameManagedType(resetInputEventFfx.GetReturnType(), voidClass)',
            'V240: SetInputEvent feasibility',
            'active=0',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_emit_particle_probe_requires_exact_tag_lookup_particle_field_and_emit_abi(self):
        expected = (
            'Class scrDecorationManager("", "scrDecorationManager")',
            'Class scrDecoration("", "scrDecoration")',
            'Class scrParticleDecoration("", "scrParticleDecoration")',
            'Class particleSystem("UnityEngine", "ParticleSystem")',
            '"GetTaggedDecorations", {enumerableString.GetCompileTimeClass()}',
            'toArrayDefinition.GetGeneric({scrDecoration.GetCompileTimeClass()})',
            'scrParticleDecoration.GetField("particleSystem")',
            'particleSystem.GetMethod("Emit", {Defaults::Get<int>()})',
            'SameManagedType(particleSystemField.GetType(), particleSystem)',
            'SameManagedType(emitCount.GetReturnType(), voidClass)',
            'emitParticleSubstrate = scrParticleDecoration',
            'V240: EmitParticle ABI',
            'active=0',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_set_particle_remains_preserve_only(self):
        self.assertIn('Class("ADOFAI.FloorFX", "ffxSetParticlePlus")', self.probe)
        self.assertIn('V240: SetParticle feasibility', self.probe)
        self.assertIn('preserveOnly=1', self.probe)

    def test_advanced_filter_probe_only_proves_generic_reflection_substrate(self):
        expected = (
            'Class ffxSetFilterPlus("", "ffxSetFilterPlus")',
            'Class ffxSetFilterAdvancedPlus("", "ffxSetFilterAdvancedPlus")',
            'Class systemType("System", "Type")',
            'Class gameObject("UnityEngine", "GameObject")',
            'Class component("UnityEngine", "Component")',
            'Class behaviour("UnityEngine", "Behaviour")',
            'systemType.GetMethod("GetType", {stringClass.GetCompileTimeClass()})',
            'gameObject.GetMethod("GetComponent", {systemType.GetCompileTimeClass()})',
            'gameObject.GetMethod("AddComponent", {systemType.GetCompileTimeClass()})',
            'behaviour.GetProperty("enabled")',
            'advancedFilterReflectionBase = systemType',
            'V240: SetFilterAdvanced feasibility',
            'active=0',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_probe_has_no_self_registration_and_runs_only_from_post_bnm_callback(self):
        self.assertNotIn("nativeRegisterFeasibilityProbe", self.event_java)
        self.assertNotIn("JNIEXPORT", self.probe)
        self.assertNotIn("Loading::AddOnLoadedEvent", self.probe)
        self.assertIn('extern "C" void V240RunPostV240FeasibilityProbe()', self.probe)
        self.assertIn("std::once_flag g_postV240ProbeOnce;", self.probe)

        declaration = 'extern "C" void V240RunPostV240FeasibilityProbe();'
        self.assertIn(declaration, self.event_native)
        callback_start = self.event_native.index("Loading::AddOnLoadedEvent([]() {")
        callback_end = self.event_native.index("});", callback_start)
        callback = self.event_native[callback_start:callback_end]
        self.assertIn("V240RunPostV240FeasibilityProbe();", callback)
        self.assertIn("ProbeAndInstallEventCompat();", callback)
        self.assertLess(
            callback.index("V240RunPostV240FeasibilityProbe();"),
            callback.index("ProbeAndInstallEventCompat();"),
            "evidence-only ABI probe must run after BNM load and before active event hook install",
        )
        self.assertEqual(
            2,
            self.event_native.count("V240RunPostV240FeasibilityProbe();"),
            "only the declaration and one post-BNM call site are allowed",
        )

    def test_native_payload_cannot_silently_drop_probe(self):
        self.assertIn("V240PostV240Probe.cpp", self.native_build)
        self.assertIn("V240RunPostV240FeasibilityProbe", self.native_build)
        self.assertIn("V240: EmitParticle ABI", self.native_build)


if __name__ == "__main__":
    unittest.main()
