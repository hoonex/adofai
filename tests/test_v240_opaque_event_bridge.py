from pathlib import Path
import base64
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"
OPAQUE = JAVA / "V240OpaqueEventBridge.java"
SCANNER = JAVA / "V240ChartCompatibilityScanner.java"
EVENT_COMPAT = JAVA / "V240EventCompat.java"
SET_FRAME_BACKPORT = JAVA / "V240SetFrameRateBackport.java"
ANDROID_BRIDGE = JAVA / "V240AndroidBridge.java"
LEVEL_BRIDGE = JAVA / "V240LevelFolderBridge.java"
SELECTOR = JAVA / "FileSelector.java"
BUILD = ROOT / "scripts/build-v240-fixed-java.sh"


class V240OpaqueEventBridgeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opaque = OPAQUE.read_text(encoding="utf-8")
        cls.scanner = SCANNER.read_text(encoding="utf-8")
        cls.backport = SET_FRAME_BACKPORT.read_text(encoding="utf-8")
        cls.android = ANDROID_BRIDGE.read_text(encoding="utf-8")
        cls.level = LEVEL_BRIDGE.read_text(encoding="utf-8")
        cls.selector = SELECTOR.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")

    def test_fallback_placeholders_stay_inactive_and_sidecar_backed(self):
        self.assertIn('MARKER_PREFIX = "__V240_OPAQUE__:"', self.opaque)
        self.assertIn('"eventType\\":\\"EditorComment\\"', self.opaque)
        self.assertIn('"eventType\\":\\"AddDecoration\\"', self.opaque)
        self.assertIn('\\"active\\":false', self.opaque)
        self.assertIn('writeSmallFile(new File(session, token + ".json"), eventObject)', self.opaque)
        self.assertIn('copyUtf8File(original, writer)', self.opaque)
        self.assertIn('V240SetFrameRateBackport.maybePlaceholder', self.opaque)
        self.assertIn('V240SetFrameRateBackport.tokenFromPlaceholder', self.opaque)
        self.assertNotIn('LevelEventType', self.opaque)
        self.assertNotIn('libil2cpp', self.opaque)

    def test_bridge_reuses_scanner_inventory_instead_of_duplicating_event_types(self):
        self.assertIn('V240ChartCompatibilityScanner.scanAndLog(chart)', self.opaque)
        self.assertIn('Set<String> targetTypes = scan.postV240', self.opaque)
        for event_type in (
            "SetFrameRate", "SetInputEvent", "AddParticle", "SetParticle", "EmitParticle",
            "SetFilterAdvanced", "TileDimensions",
        ):
            self.assertIn('"' + event_type + '"', self.scanner)
            self.assertNotIn('"' + event_type + '"', self.opaque)

    def test_external_save_builds_restored_private_export_before_opening_saf_output(self):
        self.assertIn('buildRestoredExport', self.opaque)
        self.assertIn('releaseRestoredExport', self.opaque)
        for source in (self.android, self.level):
            self.assertIn('V240OpaqueEventBridge.buildRestoredExport', source)
            self.assertIn('V240OpaqueEventBridge.releaseRestoredExport', source)
            self.assertLess(
                source.index('V240OpaqueEventBridge.buildRestoredExport'),
                source.index('requireOutput(resolver, binding.uri)'),
            )

    def test_save_as_clones_sidecar_before_new_binding(self):
        self.assertIn('beginSave(String suggestedName, String mime, String opaqueSourcePath)', self.android)
        self.assertIn('V240OpaqueEventBridge.cloneSession', self.android)
        clone = self.android.index('V240OpaqueEventBridge.cloneSession')
        bind = self.android.index('bindSave(context, uri, working)', clone)
        self.assertLess(clone, bind)
        self.assertIn('mimeForFilename(suggestedName), filePath', self.selector)

    def test_prepare_happens_before_writable_bindings_and_file_selector_does_not_rescan(self):
        direct_prepare = self.android.index('V240OpaqueEventBridge.prepareForV240')
        direct_bind = self.android.index('bindSave(context, uris[0], workingFiles.get(0))')
        self.assertLess(direct_prepare, direct_bind)

        tree_prepare = self.level.index('V240OpaqueEventBridge.prepareForV240')
        tree_bind = self.level.index('installed = new SaveBinding')
        self.assertLess(tree_prepare, tree_bind)

        self.assertIn('V240OpaqueEventBridge.prepareForV240(chart)', self.selector)
        self.assertNotIn('V240ChartCompatibilityScanner.scanAndLog(chart)', self.selector)

    def test_runtime_dex_contract_contains_scanner_opaque_bridge_and_backport_codec(self):
        self.assertIn('Lcom/unity3d/player/V240ChartCompatibilityScanner;', self.build)
        self.assertIn('Lcom/unity3d/player/V240OpaqueEventBridge;', self.build)
        self.assertIn('Lcom/unity3d/player/V240SetFrameRateBackport;', self.build)

    def test_production_java_round_trip_is_byte_exact_and_fail_closed_before_export(self):
        javac = shutil.which("javac")
        java = shutil.which("java")
        self.assertIsNotNone(javac, "JDK javac is required for the opaque bridge behavior contract")
        self.assertIsNotNone(java, "JDK java is required for the opaque bridge behavior contract")

        original_text = (
            '{\r\n'
            '  "pathData":"R!R",\r\n'
            '  "settings":{"songFilename":"song.ogg","artist":"홍길동"},\r\n'
            '  "actions":[\r\n'
            '    {"floor":1,"eventType":"MoveCamera","duration":1},\r\n'
            '    {"floor":2,"eventType":"SetFrameRate","frameRate":144,'
            '"nested":{"brace":"{still-json}","items":[1,{"q":"\\\\\\\""}]},"label":"한글"},\r\n'
            '    {"floor":3,"eventType":"SetInputEvent","eventTag":"x:y","active":true}\r\n'
            '  ],\r\n'
            '  "decorations":[\r\n'
            '    {"floor":4,"eventType":"AddParticle","tag":"fx",'
            '"params":{"seed":7,"literal":"}{"}},\r\n'
            '    {"floor":5,"eventType":"AddDecoration","decorationImage":"bg.png","opacity":50}\r\n'
            '  ]\r\n'
            '}\r\n'
        )
        original = b"\xef\xbb\xbf" + original_text.encode("utf-8")
        known_only = (
            '{"pathData":"R","actions":[{"floor":1,"eventType":"MoveCamera"}],'
            '"decorations":[]}\n'
        ).encode("utf-8")

        original_b64 = base64.b64encode(original).decode("ascii")
        known_b64 = base64.b64encode(known_only).decode("ascii")

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

            import java.io.File;
            import java.io.IOException;
            import java.nio.charset.StandardCharsets;
            import java.nio.file.Files;
            import java.nio.file.Path;
            import java.util.Arrays;
            import java.util.Base64;
            import java.util.stream.Stream;

            public final class V240OpaqueEventBridgeHostTest {{
                private static void check(boolean value, String message) {{
                    if (!value) throw new AssertionError(message);
                }}

                private static void same(byte[] expected, byte[] actual, String message) {{
                    if (!Arrays.equals(expected, actual)) {{
                        throw new AssertionError(message + " expected=" + expected.length
                                + " actual=" + actual.length);
                    }}
                }}

                private static long countExports(Path root, String chartName) throws IOException {{
                    try (Stream<Path> files = Files.list(root)) {{
                        return files.filter(path -> path.getFileName().toString().startsWith(
                                chartName + ".v240-export-"))
                                .count();
                    }}
                }}

                public static void main(String[] args) throws Exception {{
                    Path root = Files.createTempDirectory("v240-opaque-host-");
                    byte[] original = Base64.getDecoder().decode("{original_b64}");
                    byte[] knownOnly = Base64.getDecoder().decode("{known_b64}");

                    Path chart = root.resolve("level.adofai");
                    Files.write(chart, original);
                    check(V240OpaqueEventBridge.prepareForV240(chart.toFile()),
                            "modern chart was not prepared");
                    check(V240OpaqueEventBridge.hasSession(chart.toFile()),
                            "opaque session marker was not created");

                    Path session = root.resolve("level.adofai.v240-opaque-events");
                    check(Files.isDirectory(session), "opaque session directory missing");
                    check("3".equals(Files.readString(session.resolve(".ready"),
                                    StandardCharsets.UTF_8)),
                            "unexpected opaque replacement count");

                    String prepared = Files.readString(chart, StandardCharsets.UTF_8);
                    check(prepared.contains("__V240_OPAQUE__:"), "opaque marker missing");
                    check(prepared.contains("\\\"eventType\\\":\\\"EditorComment\\\""),
                            "action fallback placeholder missing");
                    check(prepared.contains("\\\"eventType\\\":\\\"AddDecoration\\\""),
                            "decoration placeholder missing");
                    check(!prepared.contains("__V240_SET_FRAME_RATE__:"),
                            "host without native capability unexpectedly enabled execution carrier");

                    File restored = V240OpaqueEventBridge.buildRestoredExport(chart.toFile());
                    check(restored != null && restored.isFile(), "restored export missing");
                    same(original, Files.readAllBytes(restored.toPath()),
                            "prepare/restore was not byte exact");
                    V240OpaqueEventBridge.releaseRestoredExport(restored);
                    check(!restored.exists(), "restored export temp was not released");

                    Path saveAs = root.resolve("saved-as.adofai");
                    Files.write(saveAs, new byte[0]);
                    check(V240OpaqueEventBridge.cloneSession(chart.toFile(), saveAs.toFile()),
                            "Save As sidecar clone failed");
                    Files.write(saveAs, Files.readAllBytes(chart));
                    File restoredSaveAs = V240OpaqueEventBridge.buildRestoredExport(saveAs.toFile());
                    same(original, Files.readAllBytes(restoredSaveAs.toPath()),
                            "cloned Save As session did not restore the original chart");
                    V240OpaqueEventBridge.releaseRestoredExport(restoredSaveAs);

                    Path known = root.resolve("known-only.adofai");
                    Files.write(known, knownOnly);
                    check(!V240OpaqueEventBridge.prepareForV240(known.toFile()),
                            "known-only chart should not create opaque state");
                    same(knownOnly, Files.readAllBytes(known),
                            "known-only chart was unexpectedly mutated");
                    check(!V240OpaqueEventBridge.hasSession(known.toFile()),
                            "known-only chart unexpectedly has opaque state");

                    Path corrupt = root.resolve("corrupt.adofai");
                    Files.write(corrupt, Files.readAllBytes(chart));
                    check(V240OpaqueEventBridge.cloneSession(chart.toFile(), corrupt.toFile()),
                            "corrupt-case sidecar clone failed");
                    Path corruptSession = root.resolve("corrupt.adofai.v240-opaque-events");
                    Path payload;
                    try (Stream<Path> files = Files.list(corruptSession)) {{
                        payload = files.filter(path -> path.getFileName().toString().endsWith(".json"))
                                .findFirst()
                                .orElseThrow(() -> new AssertionError("opaque payload missing"));
                    }}
                    Files.delete(payload);
                    long before = countExports(root, "corrupt.adofai");
                    boolean failed = false;
                    try {{
                        V240OpaqueEventBridge.buildRestoredExport(corrupt.toFile());
                    }} catch (IOException expected) {{
                        failed = true;
                    }}
                    check(failed, "missing sidecar payload did not fail the export");
                    check(before == countExports(root, "corrupt.adofai"),
                            "failed export leaked a partially restored temp file");
                }}
            }}
            """
        )

        with tempfile.TemporaryDirectory(prefix="v240-opaque-java-") as temp:
            temp_root = Path(temp)
            classes = temp_root / "classes"
            log_java = temp_root / "android/util/Log.java"
            harness_java = temp_root / "com/unity3d/player/V240OpaqueEventBridgeHostTest.java"
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
                "production opaque bridge host compile failed:\n"
                + compile_result.stdout + compile_result.stderr,
            )

            run_result = subprocess.run(
                [java, "-cp", str(classes), "com.unity3d.player.V240OpaqueEventBridgeHostTest"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                0,
                run_result.returncode,
                "production opaque bridge behavior contract failed:\n"
                + run_result.stdout + run_result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
