from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "android" / "mobile-editor-shell" / "src" / "com" / "unity3d" / "player" / "MobileEditorShell.java"
TOOL = ROOT / "tools" / "apply_mobile_editor_document_safety.py"
BUILD = ROOT / "scripts" / "build-payload.sh"


class MobileEditorDocumentSafetyTests(unittest.TestCase):
    def render(self, source: Path = SOURCE):
        temp = tempfile.TemporaryDirectory()
        output = Path(temp.name) / "MobileEditorShell.java"
        result = subprocess.run(
            ["python3", str(TOOL), str(source), str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return temp, output, result

    def test_transform_removes_open_and_view_time_normalization(self):
        temp, output, result = self.render()
        try:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = output.read_text(encoding="utf-8")

            for forbidden in (
                'if (!(parsed.opt("settings") instanceof JSONObject)) parsed.put("settings", new JSONObject());',
                'if (!(parsed.opt("actions") instanceof JSONArray)) parsed.put("actions", new JSONArray());',
                'if (!(parsed.opt("decorations") instanceof JSONArray)) parsed.put("decorations", new JSONArray());',
                'JSONArray array = getOrCreateArray(name);',
                'private static JSONArray getOrCreateArray(String key)',
            ):
                with self.subTest(forbidden=forbidden):
                    self.assertNotIn(forbidden, text)

            self.assertIn("Opening a chart is read-only with respect to parsed data", text)
            self.assertIn("Structured Settings editing is disabled to preserve it exactly", text)
            self.assertIn("<preserved non-array ", text)
            self.assertIn("private static JSONArray getExistingArray(String key)", text)
            self.assertIn("private static JSONArray getOrCreateArrayForWrite(String key) throws JSONException", text)
        finally:
            temp.cleanup()

    def test_missing_structured_fields_are_created_only_by_explicit_add(self):
        temp, output, result = self.render()
        try:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = output.read_text(encoding="utf-8")
            self.assertIn(
                'if (document.opt("settings") != settings) document.put("settings", settings);',
                text,
            )
            self.assertIn(
                'getOrCreateArrayForWrite(decoration ? "decorations" : "actions").put(event);',
                text,
            )
            self.assertIn(
                'throw new JSONException(key + " is not an array; use Raw tab to replace it explicitly");',
                text,
            )
        finally:
            temp.cleanup()

    def test_transform_fails_closed_if_source_blob_moves(self):
        with tempfile.TemporaryDirectory() as temp:
            drifted = Path(temp) / "MobileEditorShell.java"
            drifted.write_bytes(SOURCE.read_bytes() + b"\n")
            result = subprocess.run(
                ["python3", str(TOOL), str(drifted), str(Path(temp) / "out.java")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("refusing document-safety transform: source blob", result.stderr)

    def test_canonical_payload_compiles_only_generated_safe_shell(self):
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn("apply_mobile_editor_document_safety.py", text)
        self.assertIn('EDITOR_SHELL_TEMPLATE=', text)
        self.assertIn('GENERATED_JAVA_DIR=', text)
        self.assertIn('EDITOR_SHELL="${GENERATED_JAVA_DIR}/MobileEditorShell.java"', text)
        self.assertIn('"${EDITOR_SHELL}"', text)
        self.assertNotIn('"${EDITOR_SHELL_TEMPLATE}"\n', text.split("javac", 1)[1])


if __name__ == "__main__":
    unittest.main()
