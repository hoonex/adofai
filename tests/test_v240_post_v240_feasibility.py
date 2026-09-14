from pathlib import Path
import base64
import json
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"
SCANNER = JAVA / "V240ChartCompatibilityScanner.java"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"
EVENT_COMPAT = JAVA / "V240EventCompat.java"
SET_FRAME_BACKPORT = JAVA / "V240SetFrameRateBackport.java"
EVENT_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"
POST_V240_PROBE = ROOT / "android/v240-fixed-runtime/native/V240PostV240Probe.cpp"
EVIDENCE = ROOT / "android/v240-fixed-runtime/evidence/post-v240-feasibility.json"


class V240PostV240FeasibilityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.event_native = EVENT_NATIVE.read_text(encoding="utf-8")
        cls.post_v240_probe = POST_V240_PROBE.read_text(encoding="utf-8")
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

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

    def test_feasibility_probe_remains_read_only_and_caller_gated(self):
        for forbidden in (
            ".Call(",
            ".Set(",
            "BasicHook",
            "CreateNewObject",
            "Loading::AddOnLoadedEvent",
            "JNIEXPORT",
            "nativeRegisterFeasibilityProbe",
        ):
            self.assertNotIn(forbidden, self.post_v240_probe)
        self.assertIn("V240RunPostV240FeasibilityProbe", self.post_v240_probe)
        self.assertIn("std::call_once(g_postV240ProbeOnce", self.post_v240_probe)
        self.assertIn("active=0", self.post_v240_probe)
        self.assertNotIn("__V240_SET_INPUT_EVENT__:", self.post_v240_probe)
        self.assertNotIn("__V240_SET_FILTER_ADVANCED__:", self.post_v240_probe)
        self.assertNotIn("__V240_PARTICLE__:", self.post_v240_probe)

    def test_authoritative_source_fingerprint_is_pinned_but_exact_abi_audit_is_pending(self):
        source = self.evidence["authoritative_v240_source"]
        self.assertEqual("V2.4.0 Custom.apk", source["name"])
        self.assertEqual(370092054, source["size_bytes"])
        self.assertEqual(
            "630f519ae1ab3391aad95da90ebc296f4f0f8ae4ea41024ace7349d93926ef30",
            source["sha256"],
        )
        self.assertNotEqual("proven", source["exact_abi_audit"])

    def test_evidence_matrix_fail_closes_every_unproven_family(self):
        self.assertEqual(
            "fail_closed_until_exact_v240_abi_is_proven",
            self.evidence["policy"],
        )
        self.assertEqual("preserve_only", self.evidence["promotion_rule"]["default"])
        required = set(self.evidence["promotion_rule"]["all_required"])
        self.assertEqual(
            {
                "exact_v240_abi_proven",
                "scheduler_or_lifecycle_semantics_proven",
                "lossless_fallback_preserved",
                "production_tests_green",
            },
            required,
        )
        for event_type in (
            "SetInputEvent",
            "EmitParticle",
            "SetParticle",
            "SetFilterAdvanced",
        ):
            family = self.evidence["families"][event_type]
            if not family["exact_v240_abi_proven"]:
                self.assertFalse(
                    family["active_backport_allowed"],
                    event_type + " cannot execute before exact v2.4 ABI proof",
                )

    def test_emit_particle_is_ranked_candidate_not_active_backport(self):
        emit = self.evidence["families"]["EmitParticle"]
        self.assertEqual("candidate_preserve_only", emit["status"])
        self.assertEqual(1, emit["candidate_rank"])
        self.assertFalse(emit["exact_v240_abi_proven"])
        self.assertFalse(emit["active_backport_allowed"])
        self.assertEqual(
            "ADOFAI.FloorFX.ffxEmitParticlePlus",
            emit["latest_runtime_class"],
        )
        surfaces = set(emit["required_v240_surfaces"])
        self.assertIn("scrDecorationManager singleton", surfaces)
        self.assertIn(
            "scrParticleDecoration.particleSystem : UnityEngine.ParticleSystem",
            surfaces,
        )
        self.assertIn("UnityEngine.ParticleSystem.Emit(int) -> void", surfaces)

        historical = emit["historical_nearby_evidence"]
        self.assertEqual("indirect_only", historical["grade"])
        self.assertEqual("v2.5.0", historical["adofai_version"])
        self.assertEqual("adofaiex/Iridium", historical["repository"])
        self.assertEqual(
            "fe7ff015c6e28285d288d6bf58f447d5a1883aa5",
            historical["commit"],
        )
        self.assertIn("v2.5.0", historical["commit_declares_support_for"])
        self.assertIn(
            "scrParticleDecoration.particleSystem",
            historical["same_commit_particle_code_uses"],
        )

    def test_wider_modern_families_remain_preserve_only_in_evidence(self):
        families = self.evidence["families"]
        for event_type in ("SetInputEvent", "SetParticle", "SetFilterAdvanced"):
            self.assertEqual("preserve_only", families[event_type]["status"])
            self.assertFalse(families[event_type]["exact_v240_abi_proven"])
            self.assertFalse(families[event_type]["active_backport_allowed"])

    def test_production_bridge_round_trips_representative_unproven_events_byte_exact(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac, "JDK javac is required for preserve-only behavior proof")
        self.assertIsNotNone(java, "JDK java is required for preserve-only behavior proof")

        original_text = (
            '{\r\n'
            '  "pathData":"R!R",\r\n'
            '  "actions":[\r\n'
            '    {"floor":1,"eventType":"SetInputEvent","inputAction":"Set",'
            '"inputEventState":"Down","inputEventTarget":"Any","active":true,'
            '"eventTag":"input:gate","futureInputField":{"mode":"모바일"}},\r\n'
            '    {"floor":2,"eventType":"SetFilterAdvanced","filterType":"Glitch",'
            '"enabled":true,"intensity":73.5,"duration":1.25,"easing":"InOutSine",'
            '"disableOthers":false,"plane":"Foreground","angleOffset":2.5,'
            '"futureFilterField":[1,{"nested":true}]},\r\n'
            '    {"floor":3,"eventType":"SetParticle","tag":"fx:main","duration":2,'
            '"positionOffset":[1.5,-2],"rotationOffset":45,"scale":[120,80],'
            '"opacity":67,"color":"ff00ffaa","colorTo":"00ffffff",'
            '"colorToDuration":0.5,"colorToEasing":"OutQuad","easing":"InOutSine",'
            '"futureParticleField":{"seed":7}},\r\n'
            '    {"floor":4,"eventType":"EmitParticle","tag":"fx:main","amount":12,'
            '"angleOffset":-1.25,"futureEmitField":"그대로"}\r\n'
            '  ],\r\n'
            '  "decorations":[\r\n'
            '    {"floor":5,"eventType":"AddParticle","particle":"spark✨",'
            '"angleOffset":0.125,"futureAddField":{"literal":"}{","items":[1,2,3]}}\r\n'
            '  ]\r\n'
            '}\r\n'
        )
        original = b"\xef\xbb\xbf" + original_text.encode("utf-8")
        original_b64 = base64.b64encode(original).decode("ascii")

        log_source = textwrap.dedent(
            """
            package android.util;
            public final class Log {
                public static int d(String tag, String msg) { return 0; }
                public static int w(String tag, String msg) { return 0; }
                public static int w(String tag, String msg, Throwable error) { return 0; }
                public static int e(String tag, String msg) { return 0; }
                public static int e(String tag, String msg, Throwable error) { return 0; }
            }
            """
        )

        harness_source = textwrap.dedent(
            f"""
            package com.unity3d.player;

            import java.nio.charset.StandardCharsets;
            import java.nio.file.Files;
            import java.nio.file.Path;
            import java.util.Arrays;
            import java.util.Base64;

            public final class V240PostV240PreserveOnlyHostTest {{
                private static void check(boolean value, String message) {{
                    if (!value) throw new AssertionError(message);
                }}

                public static void main(String[] args) throws Exception {{
                    Path root = Files.createTempDirectory("v240-post-v240-preserve-");
                    Path chart = root.resolve("modern.adofai");
                    byte[] original = Base64.getDecoder().decode("{original_b64}");
                    Files.write(chart, original);

                    check(V240OpaqueEventBridge.prepareForV240(chart.toFile()),
                            "modern chart was not prepared");
                    Path session = root.resolve("modern.adofai.v240-opaque-events");
                    check(Files.isDirectory(session), "opaque sidecar directory missing");
                    check("5".equals(Files.readString(session.resolve(".ready"),
                                    StandardCharsets.UTF_8)),
                            "unexpected preserve-only replacement count");

                    String prepared = Files.readString(chart, StandardCharsets.UTF_8);
                    check(prepared.contains("__V240_OPAQUE__:"), "opaque marker missing");
                    check(!prepared.contains("__V240_SET_INPUT_EVENT__:"),
                            "SetInputEvent unexpectedly gained an execution carrier");
                    check(!prepared.contains("__V240_SET_FILTER_ADVANCED__:"),
                            "SetFilterAdvanced unexpectedly gained an execution carrier");
                    check(!prepared.contains("__V240_PARTICLE__:"),
                            "particle family unexpectedly gained an execution carrier");

                    java.io.File restored = V240OpaqueEventBridge.buildRestoredExport(chart.toFile());
                    check(restored != null && restored.isFile(), "restored export missing");
                    check(Arrays.equals(original, Files.readAllBytes(restored.toPath())),
                            "unproven post-v2.4 events were not restored byte-for-byte");
                    V240OpaqueEventBridge.releaseRestoredExport(restored);
                    check(!restored.exists(), "restored export temp was not released");
                }}
            }}
            """
        )

        with tempfile.TemporaryDirectory(prefix="v240-post-v240-java-") as temp:
            temp_root = Path(temp)
            classes = temp_root / "classes"
            log_java = temp_root / "android/util/Log.java"
            harness_java = temp_root / "com/unity3d/player/V240PostV240PreserveOnlyHostTest.java"
            log_java.parent.mkdir(parents=True)
            harness_java.parent.mkdir(parents=True)
            classes.mkdir()
            log_java.write_text(log_source, encoding="utf-8")
            harness_java.write_text(harness_source, encoding="utf-8")

            compile_result = subprocess.run(
                [
                    javac,
                    "-encoding", "UTF-8",
                    "-d", str(classes),
                    str(log_java),
                    str(SCANNER),
                    str(EVENT_COMPAT),
                    str(SET_FRAME_BACKPORT),
                    str(OPAQUE),
                    str(harness_java),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                0,
                compile_result.returncode,
                "production preserve-only host compile failed:\n"
                + compile_result.stdout + compile_result.stderr,
            )

            run_result = subprocess.run(
                [java, "-cp", str(classes), "com.unity3d.player.V240PostV240PreserveOnlyHostTest"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                0,
                run_result.returncode,
                "production preserve-only behavior contract failed:\n"
                + run_result.stdout + run_result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
