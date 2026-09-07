from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-device-editor-smoke.sh"


class DeviceSmokeHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_harness_remains_non_destructive(self):
        forbidden_commands = (
            '"$ADB_BIN" uninstall',
            '"$ADB_BIN" install',
            'shell pm clear',
            'shell rm -rf /data/data',
        )
        for command in forbidden_commands:
            with self.subTest(command=command):
                self.assertNotIn(command, self.text)

    def test_package_presence_requires_real_pm_path_output(self):
        self.assertIn("PM_PATHS=", self.text)
        self.assertIn("grep -q '^package:'", self.text)

    def test_ui_evidence_covers_editor_open_save_and_reopen(self):
        for marker in (
            "01-launcher",
            "02-editor-open",
            "03-chart-loaded",
            "04-chart-saved",
            "05-chart-reopened",
            "06-after-preview",
            "Loaded successfully",
            "Saved",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_runtime_evidence_is_pid_scoped_when_supported(self):
        self.assertIn('logcat -d --pid "$launch_pid"', self.text)
        self.assertIn("ADOFAI.MobileEditor:*", self.text)
        self.assertIn("IL2CPP_EXPORTS:*", self.text)

    def test_preview_success_and_fail_closed_are_distinguished(self):
        self.assertIn("Mobile editor preview bridge installed on Unity game-thread input poll", self.text)
        self.assertIn("Mobile editor preview queued into current runtime", self.text)
        self.assertIn("Mobile editor preview request failed closed", self.text)
        self.assertIn("PRESENT", self.text)
        self.assertIn("ABSENT", self.text)

    def test_report_preserves_evidence_boundary(self):
        self.assertIn("not by itself proof that every chart event rendered correctly", self.text)
        self.assertIn("UNPROVEN", self.text)
        self.assertIn("REPORT.md", self.text)


if __name__ == "__main__":
    unittest.main()
