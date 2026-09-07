from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = ROOT / "tools" / "apply_hitmargin_picker_cancel_guard.py"
PREPARE = ROOT / "scripts" / "prepare-upstream.sh"

spec = importlib.util.spec_from_file_location("picker_cancel_guard", TRANSFORM_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class PickerCancelGuardTests(unittest.TestCase):
    def test_android_back_cancel_completes_file_selector(self):
        source = """class C {\n    void show() {\n        dialog = builder.create();\n        dialog.show();\n    }\n}\n"""
        rendered = module.transform(source)
        self.assertIn("dialog.setOnCancelListener(new DialogInterface.OnCancelListener()", rendered)
        self.assertIn("FileSelector.setPath(null);", rendered)
        self.assertLess(rendered.index("setOnCancelListener"), rendered.index("dialog.show();"))

    def test_transform_fails_closed_if_dialog_anchor_moves(self):
        with self.assertRaises(RuntimeError):
            module.transform("dialog = builder.create();\ndialog.show();\n")

    def test_transform_is_pinned_to_exact_upstream_identity(self):
        self.assertEqual(module.EXPECTED_HEAD, "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb")
        self.assertEqual(module.EXPECTED_BLOB, "7aca3ef20ded3eb84b41b55edbbebffd158dc06d")
        self.assertEqual(module.FILE, "app/src/main/java/com/unity3d/player/CustomFileChooser.java")

    def test_canonical_upstream_preparation_applies_cancel_guard(self):
        text = PREPARE.read_text(encoding="utf-8")
        storage = text.index("apply_hitmargin_storage_guard.py")
        cancel = text.index("apply_hitmargin_picker_cancel_guard.py")
        profile = text.index("apply_hitmargin_modern_safe_profile.py")
        self.assertLess(storage, cancel)
        self.assertLess(cancel, profile)


if __name__ == "__main__":
    unittest.main()
