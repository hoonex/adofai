import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPACK = ROOT / "scripts" / "repack-split-apks.sh"
PULL = ROOT / "scripts" / "pull-installed-splits.sh"


class SplitScriptSafetyTests(unittest.TestCase):
    def run_guard_case(self, output_kind: str):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = root / "installed"
            payload = root / "payload"
            installed.mkdir()
            payload.mkdir()

            installed_sentinel = installed / "KEEP-INSTALLED.txt"
            payload_sentinel = payload / "KEEP-PAYLOAD.txt"
            installed_sentinel.write_text("installed\n", encoding="utf-8")
            payload_sentinel.write_text("payload\n", encoding="utf-8")

            if output_kind == "installed":
                output = installed
                expected = "must not overlap installed-splits input"
            elif output_kind == "payload":
                output = payload
                expected = "must not overlap payload input"
            else:
                raise AssertionError(output_kind)

            result = subprocess.run(
                ["bash", str(REPACK), str(installed), str(payload), str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(expected, result.stderr)
            self.assertEqual(installed_sentinel.read_text(encoding="utf-8"), "installed\n")
            self.assertEqual(payload_sentinel.read_text(encoding="utf-8"), "payload\n")

    def test_repack_refuses_output_equal_to_installed_input(self):
        self.run_guard_case("installed")

    def test_repack_refuses_output_equal_to_payload_input(self):
        self.run_guard_case("payload")

    def test_repack_refuses_repository_root_as_output_before_rm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = root / "installed"
            payload = root / "payload"
            installed.mkdir()
            payload.mkdir()

            result = subprocess.run(
                ["bash", str(REPACK), str(installed), str(payload), str(ROOT)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("repository root or its parent", result.stderr)
            self.assertTrue((ROOT / "AGENTS.md").is_file())

    def test_pull_refuses_repository_root_before_device_access_or_rm(self):
        env = os.environ.copy()
        env["ADB_BIN"] = "/bin/true"
        sentinel = ROOT / "AGENTS.md"
        before = sentinel.read_bytes()

        result = subprocess.run(
            ["bash", str(PULL), str(ROOT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("contains protected path", result.stderr)
        self.assertEqual(sentinel.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
