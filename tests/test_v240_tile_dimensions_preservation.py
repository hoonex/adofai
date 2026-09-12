from pathlib import Path
import base64
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


class V240TileDimensionsPreservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.native = EVENT_NATIVE.read_text(encoding="utf-8")

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
