from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import inject_apk_payload
import patch_android_manifest


class ApkToolingTests(unittest.TestCase):
    def test_manifest_patch_is_idempotent_and_adds_storage_contract(self):
        manifest_text = '''<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.game">
    <uses-permission android:name="android.permission.INTERNET" />
    <application android:label="Game" />
</manifest>
'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "AndroidManifest.xml"
            path.write_text(manifest_text, encoding="utf-8")
            self.assertTrue(patch_android_manifest.patch_manifest(path))
            self.assertFalse(patch_android_manifest.patch_manifest(path))

            tree = ET.parse(path)
            root = tree.getroot()
            android = "{http://schemas.android.com/apk/res/android}"
            permissions = [p.get(android + "name") for p in root.findall("uses-permission")]
            self.assertEqual(permissions.count("android.permission.READ_EXTERNAL_STORAGE"), 1)
            self.assertEqual(permissions.count("android.permission.WRITE_EXTERNAL_STORAGE"), 1)
            self.assertEqual(permissions.count("android.permission.MANAGE_EXTERNAL_STORAGE"), 1)
            write = next(p for p in root.findall("uses-permission") if p.get(android + "name") == "android.permission.WRITE_EXTERNAL_STORAGE")
            self.assertEqual(write.get(android + "maxSdkVersion"), "28")
            self.assertEqual(root.find("application").get(android + "requestLegacyExternalStorage"), "true")

    def test_injector_uses_next_dex_slot_and_replaces_native_library(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_apk = root / "base.apk"
            output_apk = root / "patched.apk"
            dex = root / "payload.dex"
            lib = root / "libOctober.so"
            dex.write_bytes(b"dex\n035\0" + b"x" * 32)
            lib.write_bytes(b"\x7fELF" + b"y" * 32)

            with zipfile.ZipFile(source_apk, "w") as z:
                z.writestr("classes.dex", b"dex\n035\0base")
                z.writestr("classes2.dex", b"dex\n035\0existing")
                z.writestr("lib/arm64-v8a/libOctober.so", b"\x7fELFold")
                z.writestr("assets/keep.txt", b"keep")
                z.writestr("META-INF/CERT.SF", b"stale signature")

            report = inject_apk_payload.inject(source_apk, output_apk, dex, lib)
            self.assertEqual(report["dexEntry"], "classes3.dex")

            with zipfile.ZipFile(output_apk, "r") as z:
                self.assertEqual(z.read("classes2.dex"), b"dex\n035\0existing")
                self.assertEqual(z.read("classes3.dex"), dex.read_bytes())
                self.assertEqual(z.read("lib/arm64-v8a/libOctober.so"), lib.read_bytes())
                self.assertEqual(z.read("assets/keep.txt"), b"keep")
                self.assertNotIn("META-INF/CERT.SF", z.namelist())


if __name__ == "__main__":
    unittest.main()
