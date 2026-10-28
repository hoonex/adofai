"""Tests for bounded, non-executing local and HTTP Range APK analysis."""
import importlib.util
import io
import pathlib
import tempfile
import unittest
import urllib.request
import zipfile
from unittest.mock import patch


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "apk_remote_audit.py"
spec = importlib.util.spec_from_file_location("apk_remote_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FakeResponse:
    def __init__(self, blob: bytes, start: int, end: int, status=206):
        self.data = blob[start:end + 1]
        self.status = status
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{len(blob)}",
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("unbounded body read prohibited")
        part = self.data[:size]
        self.data = self.data[size:]
        return part


class ApkRangeAuditTest(unittest.TestCase):
    @staticmethod
    def demo_apk():
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(
                "AndroidManifest.xml",
                '<manifest xmlns:android="http://schemas.android.com/apk/res/android" '
                'package="com.fizzd.connectedworlds" android:versionCode="300382" '
                'android:versionName="3.3.1"/>',
            )
            z.writestr("lib/arm64-v8a/libil2cpp.so", b"TEST_BINARY_NOT_EXECUTED")
            z.writestr("assets/bin/Data/globalgamemanagers", b"TEST_ASSET")
            z.writestr("classes.dex", b"NOT_A_DEX")
        return data.getvalue()

    def test_inspect_local_apk_without_running(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "example.apk"
            path.write_bytes(self.demo_apk())
            info = audit.audit(audit.LocalFile(path))
            self.assertEqual(info["manifest"]["package"], "com.fizzd.connectedworlds")
            self.assertEqual(info["manifest"]["versionCode"], "300382")
            self.assertEqual(info["manifest"]["versionName"], "3.3.1")
            self.assertTrue(info["has_arm64_il2cpp"])
            self.assertTrue(info["has_unity_assets"])
            self.assertEqual(info["apk_size"], len(path.read_bytes()))

    def test_strict_range_reader(self):
        blob = self.demo_apk()
        def get(req, timeout=0):
            header = req.headers.get("Range")
            if header.startswith("bytes=-"):
                start = max(0, len(blob) - int(header[7:]))
                end = len(blob) - 1
            else:
                left, right = header.removeprefix("bytes=").split("-")
                start, end = int(left), int(right)
            return FakeResponse(blob, start, end)
        with patch.object(urllib.request, "urlopen", side_effect=get):
            src = audit.RangeFile("https://example.invalid/test.apk")
            report = audit.audit(src)
            self.assertEqual(report["file_count"], 4)
            self.assertEqual(report["manifest"]["versionCode"], "300382")
            self.assertLess(src.bytes_read, 500000)
            self.assertGreaterEqual(src.requests, 3)

    def test_range_ignored_never_fetches_large_body(self):
        with patch.object(
            urllib.request, "urlopen",
            return_value=FakeResponse(b"test", 0, 3, status=200),
        ):
            src = audit.RangeFile("https://example.invalid/ignored")
            with self.assertRaisesRegex(audit.AuditError, "ignored Range"):
                src.slice(0, 4)
            self.assertEqual(src.bytes_read, 0)

    def test_missing_manifest_not_treated_as_valid(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("classes.dex", b"not running")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "missing.apk"
            path.write_bytes(data.getvalue())
            self.assertIsNone(audit.audit(audit.LocalFile(path))["manifest"])

    def test_reject_corrupted_zip(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "corrupted.apk"
            path.write_bytes(b"not-an-apk")
            with self.assertRaisesRegex(audit.AuditError, "ZIP"):
                audit.audit(audit.LocalFile(path))

    def test_utf8_string_length_short_and_long(self):
        self.assertEqual(audit._utf8len(b"\x04abcd", 0), (4, 1))
        self.assertEqual(audit._utf8len(b"\x81\x02", 0), (258, 2))


if __name__ == "__main__":
    unittest.main()
