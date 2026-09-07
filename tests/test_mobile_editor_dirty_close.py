from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "android" / "mobile-editor-shell" / "src" / "com" / "unity3d" / "player" / "MobileEditorShell.java"
LOSSLESS = ROOT / "tools" / "apply_mobile_editor_document_safety.py"
STRICT = ROOT / "tools" / "apply_mobile_editor_json_strictness.py"
PICKER = ROOT / "tools" / "apply_mobile_editor_picker_serialization.py"
DIRTY_CLOSE = ROOT / "tools" / "apply_mobile_editor_dirty_close_guard.py"
BUILD = ROOT / "scripts" / "build-payload.sh"
HARNESS_PREPARE = ROOT / "scripts" / "prepare-editor-harness.sh"
HARNESS_WORKFLOW = ROOT / ".github" / "workflows" / "editor-harness.yml"


class MobileEditorDirtyCloseTests(unittest.TestCase):
    def render_before_dirty_close(self):
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

    def render(self):
        temp, generated = self.render_before_dirty_close()
        result = subprocess.run(
            ["python3", str(DIRTY_CLOSE), str(generated), str(generated)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return temp, generated

    def test_close_button_and_android_back_share_guarded_owner(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            self.assertIn(
                'close.setOnClickListener(new View.OnClickListener() {\n'
                '            @Override public void onClick(View view) {\n'
                '                requestClose();',
                text,
            )
            self.assertIn("dialog.setOnKeyListener(new android.content.DialogInterface.OnKeyListener()", text)
            self.assertIn("if (keyCode != android.view.KeyEvent.KEYCODE_BACK) return false;", text)
            self.assertIn("if (event.getAction() == android.view.KeyEvent.ACTION_UP) requestClose();", text)
            self.assertIn("return true;", text)
        finally:
            temp.cleanup()

    def test_dirty_close_requires_explicit_discard_and_clears_session(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            start = text.index("    private static void requestClose() {")
            end = text.index("    private static View buildEditorRoot", start)
            method = text[start:end]

            self.assertIn("if (!dirty) {", method)
            self.assertIn("dialog.dismiss();", method)
            self.assertIn('.setTitle("Unsaved changes")', method)
            self.assertIn('.setPositiveButton("Discard & close"', method)
            self.assertIn('.setNegativeButton("Keep editing", null)', method)
            self.assertIn(
                'setStatus("Unsaved changes: cannot close safely without a foreground Activity", true);',
                method,
            )

            document_clear = method.index("document = null;")
            path_clear = method.index("currentPath = null;")
            dirty_clear = method.index("dirty = false;")
            dismiss = method.rindex("dialog.dismiss();")
            self.assertLess(document_clear, path_clear)
            self.assertLess(path_clear, dirty_clear)
            self.assertLess(dirty_clear, dismiss)
        finally:
            temp.cleanup()

    def test_dirty_open_confirms_only_after_a_path_is_selected(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            begin = text.index("    private static void beginOpen() {")
            helper = text.index("    private static void confirmOpenPath", begin)
            open_flow = text[begin:helper]
            self.assertIn('if (path.length() == 0) {\n                    setStatus("Open cancelled", false);\n                    return;', open_flow)
            self.assertIn("confirmOpenPath(path);", open_flow)
            self.assertNotIn("loadPath(path);", open_flow)

            end = text.index("    private static void beginSaveAs()", helper)
            confirm = text[helper:end]
            self.assertIn("if (!dirty) {\n            loadPath(path);\n            return;", confirm)
            self.assertIn('.setTitle("Unsaved changes")', confirm)
            self.assertIn('.setPositiveButton("Discard & open"', confirm)
            self.assertIn('.setNegativeButton("Keep editing", null)', confirm)
            self.assertIn("loadPath(path);", confirm)
            self.assertNotIn("document = null;", confirm)
            self.assertNotIn("currentPath = null;", confirm)
            self.assertNotIn("dirty = false;", confirm)
            self.assertIn(
                'setStatus("Unsaved changes: cannot replace the current chart safely without a foreground Activity", true);',
                confirm,
            )
        finally:
            temp.cleanup()

    def test_dirty_preview_requires_explicit_save_before_preview(self):
        temp, generated = self.render()
        try:
            text = generated.read_text(encoding="utf-8")
            helper = text.index("    private static void confirmSaveAndPreview() {")
            preview = text.index("    private static void previewCurrent() {", helper)
            confirm = text[helper:preview]

            self.assertIn('.setTitle("Save changes before preview?")', confirm)
            self.assertIn('.setPositiveButton("Save & preview"', confirm)
            self.assertIn('.setNegativeButton("Keep editing", null)', confirm)
            self.assertIn("if (saveCurrent(false)) previewCurrent();", confirm)
            self.assertNotIn("dirty = false;", confirm)
            self.assertIn(
                'setStatus("Unsaved changes: cannot confirm preview save without a foreground Activity", true);',
                confirm,
            )

            preview_end = text.index("    private static void setStatus", preview)
            preview_method = text[preview:preview_end]
            self.assertIn("if (dirty) {\n            confirmSaveAndPreview();\n            return;\n        }", preview_method)
            self.assertNotIn("if (dirty && !saveCurrent(false)) return;", preview_method)
        finally:
            temp.cleanup()

    def test_dirty_close_transform_fails_closed_if_close_surface_moves(self):
        temp, generated = self.render_before_dirty_close()
        try:
            text = generated.read_text(encoding="utf-8")
            original = '''        close.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                if (dialog != null) dialog.dismiss();
            }
        });
'''
            self.assertIn(original, text)
            text = text.replace(
                original,
                original.replace("if (dialog != null) dialog.dismiss();", "dialog.dismiss();"),
                1,
            )
            generated.write_text(text, encoding="utf-8")

            result = subprocess.run(
                ["python3", str(DIRTY_CLOSE), str(generated), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("Close button ownership", result.stderr)
        finally:
            temp.cleanup()

    def test_dirty_close_transform_fails_closed_if_open_replacement_surface_moves(self):
        temp, generated = self.render_before_dirty_close()
        try:
            text = generated.read_text(encoding="utf-8")
            self.assertEqual(text.count("                loadPath(path);\n"), 1)
            text = text.replace(
                "                loadPath(path);\n",
                "                loadPath(path.trim());\n",
                1,
            )
            generated.write_text(text, encoding="utf-8")

            result = subprocess.run(
                ["python3", str(DIRTY_CLOSE), str(generated), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("Open replacement ownership", result.stderr)
        finally:
            temp.cleanup()

    def test_dirty_close_transform_fails_closed_if_preview_autosave_surface_moves(self):
        temp, generated = self.render_before_dirty_close()
        try:
            text = generated.read_text(encoding="utf-8")
            original = "        if (dirty && !saveCurrent(false)) return;\n"
            self.assertEqual(text.count(original), 1)
            generated.write_text(
                text.replace(original, "        if (dirty && !saveCurrent(true)) return;\n", 1),
                encoding="utf-8",
            )

            result = subprocess.run(
                ["python3", str(DIRTY_CLOSE), str(generated), str(generated)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("Preview autosave ownership", result.stderr)
        finally:
            temp.cleanup()

    def test_canonical_payload_and_harness_apply_guard_after_picker_transform(self):
        build = BUILD.read_text(encoding="utf-8")
        picker = build.index("apply_mobile_editor_picker_serialization.py")
        dirty = build.index("apply_mobile_editor_dirty_close_guard.py")
        javac = build.index("\njavac ")
        self.assertLess(picker, dirty)
        self.assertLess(dirty, javac)

        harness = HARNESS_PREPARE.read_text(encoding="utf-8")
        picker = harness.index("apply_mobile_editor_picker_serialization.py")
        dirty = harness.index("apply_mobile_editor_dirty_close_guard.py")
        self.assertLess(picker, dirty)
        self.assertIn("requestClose", harness)
        self.assertIn("confirmOpenPath", harness)
        self.assertIn("confirmSaveAndPreview", harness)
        self.assertIn("Unsaved changes", harness)
        self.assertIn("Discard & open", harness)
        self.assertIn("Save & preview", harness)

    def test_harness_rebuilds_when_dirty_close_guard_changes(self):
        workflow = HARNESS_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("'tools/apply_mobile_editor_dirty_close_guard.py'", workflow)


if __name__ == "__main__":
    unittest.main()
