from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "android" / "mobile-editor-shell" / "src" / "com" / "unity3d" / "player" / "MobileEditorShell.java"
LOSSLESS = ROOT / "tools" / "apply_mobile_editor_document_safety.py"
STRICT = ROOT / "tools" / "apply_mobile_editor_json_strictness.py"
PICKER = ROOT / "tools" / "apply_mobile_editor_picker_serialization.py"
BUILD = ROOT / "scripts" / "build-payload.sh"


class MobileEditorPickerSerializationTests(unittest.TestCase):
    def render(self):
        temp = tempfile.TemporaryDirectory()
        generated = Path(temp.name) / "MobileEditorShell.java"
        for tool, source in (
            (LOSSLESS, SOURCE),
            (STRICT, generated),
            (PICKER, generated),
        ):
            result = subprocess.run(
                ["python3", str(tool), str(source), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return temp, generated

    def test_open_and_save_as_share_single_picker_owner(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            self.assertIn("private static boolean pickerInFlight;", text)
            self.assertEqual(text.count("if (!beginPickerRequest()) return;"), 2)
            self.assertIn('setStatus("A file picker is already open", false);', text)
            self.assertIn("pickerInFlight = true;", text)
            self.assertIn("pickerInFlight = false;", text)
        finally:
            temp.cleanup()

    def test_picker_owner_is_released_on_completion_timeout_and_invocation_error(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            self.assertIn(
                'if (FileSelector.isDone) {\n                    String value = FileSelector.getFilePath();\n                    finishPickerRequest();',
                text,
            )
            self.assertIn(
                'if (System.currentTimeMillis() >= deadline) {\n                    finishPickerRequest();\n                    setStatus("File picker timed out", true);',
                text,
            )
            self.assertIn('reportError("Could not open file picker", error);', text)
            self.assertIn('reportError("Could not open Save As picker", error);', text)
            self.assertGreaterEqual(text.count("finishPickerRequest();"), 4)
        finally:
            temp.cleanup()

    def test_picker_transform_fails_closed_if_await_surface_moves(self):
        with tempfile.TemporaryDirectory() as temp:
            generated = Path(temp) / "MobileEditorShell.java"
            for tool, source in ((LOSSLESS, SOURCE), (STRICT, generated)):
                result = subprocess.run(
                    ["python3", str(tool), str(source), str(generated)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            text = generated.read_text(encoding="utf-8")
            text = text.replace("if (FileSelector.isDone) {", "if (FileSelector.isDone == true) {", 1)
            generated.write_text(text, encoding="utf-8")
            result = subprocess.run(
                ["python3", str(PICKER), str(generated), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("picker completion release", result.stderr)

    def test_canonical_build_orders_picker_transform_before_actual_javac(self):
        text = BUILD.read_text(encoding="utf-8")
        lossless = text.index("apply_mobile_editor_document_safety.py")
        strict = text.index("apply_mobile_editor_json_strictness.py")
        picker = text.index("apply_mobile_editor_picker_serialization.py")
        javac = text.index("\njavac ")
        self.assertLess(lossless, strict)
        self.assertLess(strict, picker)
        self.assertLess(picker, javac)


if __name__ == "__main__":
    unittest.main()
