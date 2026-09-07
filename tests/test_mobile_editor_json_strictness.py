from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "android" / "mobile-editor-shell" / "src" / "com" / "unity3d" / "player" / "MobileEditorShell.java"
LOSSLESS = ROOT / "tools" / "apply_mobile_editor_document_safety.py"
STRICT = ROOT / "tools" / "apply_mobile_editor_json_strictness.py"
BUILD = ROOT / "scripts" / "build-payload.sh"


class MobileEditorJsonStrictnessTests(unittest.TestCase):
    def render(self):
        temp = tempfile.TemporaryDirectory()
        generated = Path(temp.name) / "MobileEditorShell.java"
        first = subprocess.run(
            ["python3", str(LOSSLESS), str(SOURCE), str(generated)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        second = subprocess.run(
            ["python3", str(STRICT), str(generated), str(generated)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return temp, generated, second

    def test_all_editor_json_entry_points_require_complete_consumption(self):
        temp, generated, result = self.render()
        try:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = generated.read_text(encoding="utf-8")
            self.assertIn("private static void requireJsonEof(JSONTokener tokener) throws JSONException", text)
            self.assertIn('throw new JSONException("Unexpected trailing content after JSON value")', text)
            self.assertIn("JSONObject parsed = parseJsonObject(raw);", text)
            self.assertEqual(text.count("JSONObject replacement = parseJsonObject(raw.getText().toString());"), 2)
            self.assertIn("if (normalized.startsWith(\"[\")) return parseJsonArray(normalized);", text)
            self.assertIn("JSONTokener tokener = new JSONTokener(sanitizeJson(trimmed));", text)
            self.assertNotIn("return new JSONTokener(trimmed).nextValue();", text)
            self.assertNotIn("new JSONObject(sanitizeJson(raw))", text)
        finally:
            temp.cleanup()

    def test_strict_transform_fails_if_expected_parse_surface_moves(self):
        with tempfile.TemporaryDirectory() as temp:
            generated = Path(temp) / "MobileEditorShell.java"
            first = subprocess.run(
                ["python3", str(LOSSLESS), str(SOURCE), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            text = generated.read_text(encoding="utf-8")
            text = text.replace(
                '        if (normalized.startsWith("[")) return new JSONArray(normalized);\n',
                '        if (normalized.startsWith("[")) return new JSONArray(normalized + " ");\n',
                1,
            )
            generated.write_text(text, encoding="utf-8")

            result = subprocess.run(
                ["python3", str(STRICT), str(generated), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("angleData array parsing", result.stderr)

    def test_canonical_build_orders_lossless_then_strict_before_javac(self):
        text = BUILD.read_text(encoding="utf-8")
        lossless = text.index("apply_mobile_editor_document_safety.py")
        strict = text.index("apply_mobile_editor_json_strictness.py")
        javac = text.index("javac")
        self.assertLess(lossless, strict)
        self.assertLess(strict, javac)
        self.assertIn('"${EDITOR_SHELL}" \\\n  "${EDITOR_SHELL}"', text)


if __name__ == "__main__":
    unittest.main()
