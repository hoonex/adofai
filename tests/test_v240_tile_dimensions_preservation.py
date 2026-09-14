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
EVENT_COMPAT = JAVA / "V240EventCompat.java"
SET_FRAME = JAVA / "V240SetFrameRateBackport.java"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"
EVENT_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"
EVIDENCE = ROOT / "android/v240-fixed-runtime/evidence/post-v240-feasibility.json"


class V240TileDimensionsPreservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.native = EVENT_NATIVE.read_text(encoding="utf-8")
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_tile_dimensions_is_post_v240_but_has_no_active_backport(self):
        self.assertIn('"TileDimensions"', self.scanner)
        self.assertNotIn("V240TileDimensionsBackport", self.opaque)
        self.assertNotIn("__V240_TILE_DIMENSIONS__", self.opaque)
        self.assertIn('scrFloor.GetField("lengthMult")', self.native)
        self.assertIn('scrFloor.GetField("widthMult")', self.native)

        start = self.native.index('auto floorLengthMult =')
        end = self.native.index('LOGD("V240: TileDimensions ABI', start)
        probe = self.native[start:end]
        self.assertNotIn(".Set(", probe)
        self.assertNotIn("BasicHook", probe)
        self.assertNotIn("CreateNewObject", probe)

    def test_tile_dimensions_evidence_fail_closes_schema_and_runtime_gaps(self):
        tile = self.evidence["families"]["TileDimensions"]
        self.assertEqual("candidate_preserve_only", tile["status"])
        self.assertFalse(tile["exact_v240_abi_proven"])
        self.assertFalse(tile["scheduler_or_lifecycle_semantics_proven"])
        self.assertFalse(tile["active_backport_allowed"])

        surfaces = set(tile["required_v240_surfaces"])
        self.assertIn("scrFloor.lengthMult : float", surfaces)
        self.assertIn("scrFloor.widthMult : float", surfaces)

        schema = tile["serialized_schema_evidence"]
        width_length = schema["width_length"]
        self.assertEqual("strong_third_party_parser_evidence", width_length["grade"])
        self.assertEqual("Cocoa2219/AdofaiBin", width_length["repository"])
        self.assertIn("Width = 100", width_length["properties"])
        self.assertIn("Length = 100", width_length["properties"])
        self.assertIn("PascalCase", width_length["parser_mapping"])

        conflict = schema["conflicting_interface"]
        self.assertEqual("conflicting_third_party_interface", conflict["grade"])
        self.assertEqual("low_unverified_interface", conflict["weight"])
        self.assertEqual("adofaiex/ADOFAI-JS", conflict["repository"])
        self.assertEqual(
            "b0a922606b0dc138a66c119d5de5482e4c0c9aff",
            conflict["introduced_in_commit"],
        )
        self.assertIn("scale", conflict["properties"])
        self.assertIn("scaleTo", conflict["properties"])
        self.assertIn("Do not transform or execute", conflict["resolution"])

        runtime = tile["runtime_evidence"]["current_multiplier_usage"]
        self.assertEqual("strong_current_runtime_usage_evidence", runtime["grade"])
        self.assertEqual("kkorenn/adomeji", runtime["repository"])
        self.assertEqual(
            "a4f22ffeed0571c35d1615f3ac86f18ed81783eb",
            runtime["commit"],
        )
        self.assertEqual(
            "95147c78be862bf17518e08f8675fd6c074f085d",
            runtime["source_blob_sha"],
        )
        formulae = "\n".join(runtime["observed_formulae"])
        self.assertIn("baseFloorDimensions.x * scrFloor.lengthMult", formulae)
        self.assertIn("baseFloorDimensions.y * scrFloor.widthMult", formulae)
        self.assertIn("dimensionless runtime multipliers", runtime["proves"])
        self.assertIn("does not prove how serialized", runtime["does_not_prove"])

        blockers = set(tile["remaining_blockers"])
        self.assertIn("resolve serialized schema conflict", blockers)
        self.assertIn(
            "prove serialized width/length 100-to-1.0 conversion into runtime multipliers",
            blockers,
        )
        self.assertIn("prove floor propagation/application ordering", blockers)
        self.assertIn("prove exact v2.4 runtime fields on authoritative source or device", blockers)

    def test_production_bridge_preserves_tile_dimensions_byte_exactly(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac)
        self.assertIsNotNone(java)

        chart = (
            '{\r\n'
            '  "angleData":[0,90,180],\r\n'
            '  "settings":{"songFilename":"song.ogg"},\r\n'
            '  "actions":[\r\n'
            '    {"floor":1,"eventType":"TileDimensions","scale":125,'
            '"scaleTo":80,"scaleToDuration":2,"scaleToEasing":"InOutSine",'
            '"angleOffset":15,"futureField":{"keep":true}},\r\n'
            '    {"floor":2,"eventType":"MoveCamera","duration":1}\r\n'
            '  ],\r\n'
            '  "decorations":[]\r\n'
            '}\r\n'
        )
        original = b"\xef\xbb\xbf" + chart.encode("utf-8")
        encoded = base64.b64encode(original).decode("ascii")

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
        harness = textwrap.dedent(
            f"""
            package com.unity3d.player;

            import java.io.File;
            import java.nio.charset.StandardCharsets;
            import java.nio.file.Files;
            import java.nio.file.Path;
            import java.util.Arrays;
            import java.util.Base64;
            import java.util.stream.Stream;

            public final class V240TileDimensionsPreservationHostTest {{
                private static void check(boolean value, String message) {{
                    if (!value) throw new AssertionError(message);
                }}

                public static void main(String[] args) throws Exception {{
                    Path root = Files.createTempDirectory("v240-tile-dimensions-");
                    byte[] original = Base64.getDecoder().decode("{encoded}");
                    Path chart = root.resolve("level.adofai");
                    Files.write(chart, original);

                    check(V240OpaqueEventBridge.prepareForV240(chart.toFile()),
                            "TileDimensions chart was not prepared");
                    String prepared = Files.readString(chart, StandardCharsets.UTF_8);
                    check(prepared.contains("\\\"eventType\\\":\\\"EditorComment\\\""),
                            "TileDimensions did not become an inactive opaque action");
                    check(prepared.contains("__V240_OPAQUE__:"),
                            "TileDimensions opaque token missing");
                    check(!prepared.contains("__V240_TILE_DIMENSIONS__"),
                            "unproven TileDimensions execution carrier became active");

                    Path session = root.resolve("level.adofai.v240-opaque-events");
                    check(Files.isDirectory(session), "TileDimensions sidecar directory missing");
                    check("1".equals(Files.readString(session.resolve(".ready"),
                                    StandardCharsets.UTF_8)),
                            "unexpected TileDimensions replacement count");
                    try (Stream<Path> files = Files.list(session)) {{
                        check(files.anyMatch(path -> path.getFileName().toString().endsWith(".json")),
                                "TileDimensions original payload sidecar missing");
                    }}

                    File restored = V240OpaqueEventBridge.buildRestoredExport(chart.toFile());
                    check(restored != null && restored.isFile(), "restored chart missing");
                    check(Arrays.equals(original, Files.readAllBytes(restored.toPath())),
                            "TileDimensions chart was not restored byte-for-byte");
                    V240OpaqueEventBridge.releaseRestoredExport(restored);
                }}
            }}
            """
        )

        with tempfile.TemporaryDirectory(prefix="v240-tile-dimensions-java-") as temp:
            root = Path(temp)
            classes = root / "classes"
            log_java = root / "android/util/Log.java"
            harness_java = root / "com/unity3d/player/V240TileDimensionsPreservationHostTest.java"
            log_java.parent.mkdir(parents=True)
            harness_java.parent.mkdir(parents=True)
            classes.mkdir()
            log_java.write_text(log_source, encoding="utf-8")
            harness_java.write_text(harness, encoding="utf-8")

            compiled = subprocess.run(
                [javac, "-encoding", "UTF-8", "-d", str(classes),
                 str(log_java), str(SCANNER), str(EVENT_COMPAT), str(SET_FRAME),
                 str(OPAQUE), str(harness_java)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(0, compiled.returncode, compiled.stdout + compiled.stderr)

            executed = subprocess.run(
                [java, "-cp", str(classes),
                 "com.unity3d.player.V240TileDimensionsPreservationHostTest"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(0, executed.returncode, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
