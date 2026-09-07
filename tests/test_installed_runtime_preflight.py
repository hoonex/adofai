import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify-installed-runtime.sh"
BUILD = ROOT / "scripts" / "build-from-installed-current.sh"


class InstalledRuntimePreflightTests(unittest.TestCase):
    def make_fake_adb(self, directory: Path) -> tuple[Path, Path]:
        fake = directory / "adb"
        command_log = directory / "adb-commands.log"
        fake.write_text(
            textwrap.dedent(
                r'''#!/usr/bin/env bash
set -euo pipefail

log="${FAKE_ADB_LOG:?}"
printf '%s|%s\n' "${1:-}" "${*:2}" >> "$log"

case "${1:-}" in
  get-state)
    echo device
    ;;
  shell)
    shift
    case "${1:-}" in
      pm)
        if [[ "${2:-}" == "path" ]]; then
          echo 'package:/data/app/fake/base.apk'
          if [[ "${FAKE_ARM64:-1}" == "1" ]]; then
            echo 'package:/data/app/fake/split_config.arm64_v8a.apk'
          fi
          echo 'package:/data/app/fake/split_asset_pack.apk'
        fi
        ;;
      dumpsys)
        cat <<EOF
  Package [com.fizzd.connectedworlds] (fake):
    versionCode=${FAKE_VERSION_CODE:-300382} minSdk=23 targetSdk=35
    versionName=${FAKE_VERSION_NAME:-3.3.1}
    primaryCpuAbi=${FAKE_ABI:-arm64-v8a}
EOF
        ;;
      *)
        echo "unexpected shell command: $*" >&2
        exit 90
        ;;
    esac
    ;;
  pull)
    echo "pull should not be reached by preflight-only tests" >&2
    exit 91
    ;;
  *)
    echo "unexpected adb command: $*" >&2
    exit 92
    ;;
esac
'''
            ),
            encoding="utf-8",
        )
        fake.chmod(0o755)
        command_log.write_text("", encoding="utf-8")
        return fake, command_log

    def make_env(self, fake_adb: Path, command_log: Path, **overrides: str) -> dict[str, str]:
        env = os.environ.copy()
        env["ADB_BIN"] = str(fake_adb)
        env["FAKE_ADB_LOG"] = str(command_log)
        env["PATH"] = str(fake_adb.parent) + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        return env

    def test_exact_331_runtime_is_accepted_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake, log = self.make_fake_adb(root)
            report = root / "runtime.txt"
            result = subprocess.run(
                ["bash", str(VERIFY), str(report)],
                cwd=ROOT,
                env=self.make_env(fake, log),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = report.read_text(encoding="utf-8")
            self.assertIn("status=compatible", text)
            self.assertIn("versionName=3.3.1", text)
            self.assertIn("versionCode=300382", text)
            self.assertIn("arm64SplitCount=1", text)

    def test_future_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake, log = self.make_fake_adb(root)
            report = root / "runtime.txt"
            result = subprocess.run(
                ["bash", str(VERIFY), str(report)],
                cwd=ROOT,
                env=self.make_env(
                    fake,
                    log,
                    FAKE_VERSION_NAME="3.3.2",
                    FAKE_VERSION_CODE="300400",
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("status=blocked", report.read_text(encoding="utf-8"))
            self.assertIn("version drift must be inspected", result.stderr)

    def test_missing_arm64_split_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake, log = self.make_fake_adb(root)
            result = subprocess.run(
                ["bash", str(VERIFY)],
                cwd=ROOT,
                env=self.make_env(fake, log, FAKE_ARM64="0"),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("arm64-v8a", result.stdout + result.stderr)

    def test_one_command_build_stops_before_pull_on_version_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake, log = self.make_fake_adb(root)
            work = root / "work"
            result = subprocess.run(
                ["bash", str(BUILD), str(work)],
                cwd=ROOT,
                env=self.make_env(
                    fake,
                    log,
                    FAKE_VERSION_NAME="3.3.2",
                    FAKE_VERSION_CODE="300400",
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            commands = log.read_text(encoding="utf-8")
            self.assertNotIn("pull|", commands)
            self.assertFalse((work / "installed-splits").exists())
            report = (work / "installed-runtime.txt").read_text(encoding="utf-8")
            self.assertIn("status=blocked", report)


if __name__ == "__main__":
    unittest.main()
