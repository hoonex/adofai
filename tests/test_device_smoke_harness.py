import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
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

    def test_guided_flow_generates_green_report_with_fake_adb(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            fake_adb = temp_path / "adb"
            state_file = temp_path / "ui-stage"
            state_file.write_text("0\n", encoding="utf-8")
            evidence = temp_path / "evidence"

            fake_adb.write_text(
                textwrap.dedent(
                    r'''#!/usr/bin/env bash
set -euo pipefail

state_file="${FAKE_ADB_STATE:?}"
command_name="${1:-}"
shift || true

case "$command_name" in
  get-state)
    echo device
    ;;
  get-serialno)
    echo FAKE-ADOFAI-DEVICE
    ;;
  shell)
    shell_command="${1:-}"
    shift || true
    case "$shell_command" in
      pm)
        if [[ "${1:-}" == "path" ]]; then
          echo 'package:/data/app/fake/base.apk'
          echo 'package:/data/app/fake/split_config.arm64_v8a.apk'
        fi
        ;;
      getprop)
        case "${1:-}" in
          ro.product.model) echo 'Fake Phone' ;;
          ro.product.device) echo 'fake_device' ;;
          ro.build.version.release) echo '16' ;;
          ro.build.version.sdk) echo '36' ;;
          *) echo '' ;;
        esac
        ;;
      dumpsys)
        echo "fake dumpsys $*"
        ;;
      am|monkey|rm)
        ;;
      pidof)
        echo 4242
        ;;
      uiautomator)
        stage="$(cat "$state_file")"
        stage=$((stage + 1))
        echo "$stage" > "$state_file"
        ;;
      *)
        echo "unexpected fake adb shell command: $shell_command $*" >&2
        exit 90
        ;;
    esac
    ;;
  exec-out)
    exec_command="${1:-}"
    shift || true
    case "$exec_command" in
      cat)
        stage="$(cat "$state_file")"
        case "$stage" in
          1) echo '<hierarchy><node text="Editor"/></hierarchy>' ;;
          2) echo '<hierarchy><node text="ADOFAI Mobile Editor"/></hierarchy>' ;;
          3) echo '<hierarchy><node text="Loaded successfully"/></hierarchy>' ;;
          4) echo '<hierarchy><node text="Saved"/></hierarchy>' ;;
          5) echo '<hierarchy><node text="Loaded successfully"/></hierarchy>' ;;
          *) echo '<hierarchy><node text="Game preview"/></hierarchy>' ;;
        esac
        ;;
      screencap)
        printf 'FAKEPNG'
        ;;
      *)
        echo "unexpected fake adb exec-out command: $exec_command $*" >&2
        exit 91
        ;;
    esac
    ;;
  logcat)
    cat <<'EOF'
09-07 14:00:00.000 4242 4242 D IL2CPP_EXPORTS: MobileEditorShell launcher installed through injected DEX loader
09-07 14:00:00.010 4242 4242 D IL2CPP_EXPORTS: Mobile editor preview bridge installed on Unity game-thread input poll
09-07 14:00:02.000 4242 4242 D IL2CPP_EXPORTS: Mobile editor preview queued into current runtime: /sdcard/level.adofai
EOF
    ;;
  *)
    echo "unexpected fake adb command: $command_name $*" >&2
    exit 92
    ;;
esac
'''
                ),
                encoding="utf-8",
            )
            fake_adb.chmod(0o755)

            env = os.environ.copy()
            env["ADB_BIN"] = str(fake_adb)
            env["FAKE_ADB_STATE"] = str(state_file)
            result = subprocess.run(
                ["bash", str(SCRIPT), str(evidence)],
                cwd=ROOT,
                env=env,
                input="\n" * 6,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            report = (evidence / "REPORT.md").read_text(encoding="utf-8")
            for boundary in (
                "Injected DEX editor bootstrap executed | PASS",
                "Floating Editor launcher visible | PASS",
                "Android-native editor shell visible | PASS",
                "Modern chart open reports success | PASS",
                "Save reports success | PASS",
                "Saved chart reopens | PASS",
                "Native preview bridge installed | PASS",
                "Preview reached current runtime LoadCustomLevel call | PASS",
                "Preview fail-closed marker | ABSENT",
            ):
                with self.subTest(boundary=boundary):
                    self.assertIn(boundary, report)

            self.assertTrue((evidence / "01-launcher.png").is_file())
            self.assertTrue((evidence / "06-after-preview.xml").is_file())
            self.assertTrue((evidence / "runtime.log").is_file())


if __name__ == "__main__":
    unittest.main()
