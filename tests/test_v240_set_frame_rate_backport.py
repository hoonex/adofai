from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"
BACKPORT = JAVA / "V240SetFrameRateBackport.java"
EVENT_COMPAT = JAVA / "V240EventCompat.java"
NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"


class V240SetFrameRateBackportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backport = BACKPORT.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")
        cls.opaque = OPAQUE.read_text(encoding="utf-8")

    def test_java_and_native_share_tokenized_fail_closed_marker(self):
        marker = "__V240_SET_FRAME_RATE__:"
        self.assertIn(marker, self.backport)
        self.assertIn(marker, self.native)
        self.assertIn("safeToken(token)", self.backport)
        self.assertIn("IsLowerUuidToken", self.native)
        self.assertIn("tokenEnd - tokenStart", self.native)

    def test_execution_subset_is_deliberately_conservative(self):
        self.assertIn('findTopLevelBoolean(eventObject, "active")', self.backport)
        self.assertIn("Boolean.FALSE.equals(active)", self.backport)
        self.assertIn('findTopLevelString(eventObject, "eventTag")', self.backport)
        self.assertIn("!eventTag.isEmpty()", self.backport)
        self.assertIn('findTopLevelBoolean(eventObject, "enabled")', self.backport)
        self.assertIn("if (enabled == null) return null;", self.backport)
        self.assertIn('findTopLevelNumber(eventObject, "frameRate")', self.backport)
        self.assertIn("Float.parseFloat(raw)", self.backport)

    def test_opaque_bridge_keeps_exact_original_sidecar_for_executable_carrier(self):
        self.assertIn("V240SetFrameRateBackport.maybePlaceholder", self.opaque)
        self.assertIn("V240SetFrameRateBackport.tokenFromPlaceholder", self.opaque)
        sidecar = self.opaque.index('writeSmallFile(new File(session, token + ".json"), eventObject)')
        carrier = self.opaque.index("V240SetFrameRateBackport.maybePlaceholder", sidecar)
        self.assertLess(sidecar, carrier)

    def test_production_marker_codec_behavior(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac)
        self.assertIsNotNone(java)

        log_source = textwrap.dedent(
            """
            package android.util;
            public final class Log {
                public static int w(String tag, String msg, Throwable error) { return 0; }
            }
            """
        )
        harness = textwrap.dedent(
            r"""
            package com.unity3d.player;

            public final class V240SetFrameRateBackportHostTest {
                private static final String TOKEN = "01234567-89ab-cdef-0123-456789abcdef";

                private static void check(boolean value, String message) {
                    if (!value) throw new AssertionError(message);
                }

                public static void main(String[] args) {
                    String good = "{\"floor\":2,\"eventType\":\"SetFrameRate\","
                            + "\"enabled\":true,\"frameRate\":144,\"angleOffset\":1.25}";
                    String carrier = V240SetFrameRateBackport.maybePlaceholder(good, 2, TOKEN, true);
                    check(carrier != null, "safe SetFrameRate did not get a carrier");
                    check(carrier.contains("\"eventType\":\"CallMethod\""),
                            "carrier is not CallMethod");
                    check(carrier.contains("__V240_SET_FRAME_RATE__:" + TOKEN + ":1:144"),
                            "carrier payload changed");
                    check(carrier.contains("\"angleOffset\":1.25"),
                            "angleOffset was not preserved");
                    check(TOKEN.equals(V240SetFrameRateBackport.tokenFromPlaceholder(carrier)),
                            "carrier token did not round trip");

                    check(V240SetFrameRateBackport.maybePlaceholder(good, 2, TOKEN, false) == null,
                            "runtime capability gate was bypassed");
                    check(V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"enabled\":true,"
                                    + "\"frameRate\":120,\"eventTag\":\"manual\"}",
                            2, TOKEN, true) == null,
                            "tagged event must stay preserve-only");
                    check(V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"active\":false,"
                                    + "\"enabled\":true,\"frameRate\":120}",
                            2, TOKEN, true) == null,
                            "inactive event must stay preserve-only");
                    check(V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"frameRate\":120}",
                            2, TOKEN, true) == null,
                            "implicit enabled default must not be guessed");
                    check(V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"enabled\":true,"
                                    + "\"frameRate\":1e100}",
                            2, TOKEN, true) == null,
                            "float overflow must stay preserve-only");
                    check(V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"nested\":{"
                                    + "\"enabled\":true,\"frameRate\":120}}",
                            2, TOKEN, true) == null,
                            "nested fields must not masquerade as top-level semantics");

                    String disable = V240SetFrameRateBackport.maybePlaceholder(
                            "{\"eventType\":\"SetFrameRate\",\"enabled\":false}",
                            3, TOKEN, true);
                    check(disable != null && disable.contains(TOKEN + ":0:0"),
                            "disable event without frameRate must remain executable");
                    check(V240SetFrameRateBackport.maybePlaceholder(good, 2,
                            "not-a-uuid", true) == null,
                            "invalid sidecar token must not become executable");
                    check(V240SetFrameRateBackport.tokenFromPlaceholder(
                            "{\"eventType\":\"CallMethod\",\"method\":\"ordinary()\"}") == null,
                            "ordinary CallMethod was mistaken for a carrier");
                }
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="v240-set-frame-rate-") as temp:
            root = Path(temp)
            classes = root / "classes"
            log_java = root / "android/util/Log.java"
            harness_java = root / "com/unity3d/player/V240SetFrameRateBackportHostTest.java"
            log_java.parent.mkdir(parents=True)
            harness_java.parent.mkdir(parents=True)
            classes.mkdir()
            log_java.write_text(log_source, encoding="utf-8")
            harness_java.write_text(harness, encoding="utf-8")

            compiled = subprocess.run(
                [javac, "-encoding", "UTF-8", "-d", str(classes),
                 str(log_java), str(EVENT_COMPAT), str(BACKPORT), str(harness_java)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(0, compiled.returncode, compiled.stdout + compiled.stderr)

            executed = subprocess.run(
                [java, "-cp", str(classes),
                 "com.unity3d.player.V240SetFrameRateBackportHostTest"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(0, executed.returncode, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
