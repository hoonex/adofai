from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import inject_split_component


class SplitApkToolingTests(unittest.TestCase):
    def test_base_split_gets_next_dex_without_native_library(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "base.apk"
            output = root / "base-patched.apk"
            dex = root / "picker.dex"
            dex.write_bytes(b"dex\n035\0" + b"d" * 32)

            with zipfile.ZipFile(source, "w") as z:
                z.writestr("classes.dex", b"dex\n035\0base")
                z.writestr("classes2.dex", b"dex\n035\0existing")
                z.writestr("assets/keep.txt", b"keep")
                z.writestr("META-INF/CERT.SF", b"stale")

            report = inject_split_component.inject_components(source, output, dex=dex)
            self.assertEqual(report["dexEntry"], "classes3.dex")

            with zipfile.ZipFile(output, "r") as z:
                self.assertEqual(z.read("classes2.dex"), b"dex\n035\0existing")
                self.assertEqual(z.read("classes3.dex"), dex.read_bytes())
                self.assertNotIn(inject_split_component.LIB_PATH, z.namelist())
                self.assertNotIn("META-INF/CERT.SF", z.namelist())
                self.assertEqual(z.read("assets/keep.txt"), b"keep")

    def test_abi_split_gets_uncompressed_october_without_dex(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "split_config.arm64_v8a.apk"
            output = root / "arm64-patched.apk"
            library = root / "libOctober.so"
            library.write_bytes(b"\x7fELF" + b"o" * 64)

            with zipfile.ZipFile(source, "w") as z:
                il2cpp = zipfile.ZipInfo("lib/arm64-v8a/libil2cpp.so")
                il2cpp.compress_type = zipfile.ZIP_STORED
                z.writestr(il2cpp, b"\x7fELF" + b"i" * 128)
                old_october = zipfile.ZipInfo(inject_split_component.LIB_PATH)
                old_october.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(old_october, b"\x7fELFold")

            report = inject_split_component.inject_components(source, output, library=library)
            self.assertEqual(report["nativeCompression"], "stored")

            with zipfile.ZipFile(output, "r") as z:
                self.assertEqual(z.read(inject_split_component.LIB_PATH), library.read_bytes())
                self.assertEqual(
                    z.getinfo(inject_split_component.LIB_PATH).compress_type,
                    zipfile.ZIP_STORED,
                )
                self.assertNotIn("classes2.dex", z.namelist())
                self.assertEqual(
                    z.getinfo("lib/arm64-v8a/libil2cpp.so").compress_type,
                    zipfile.ZIP_STORED,
                )

    def test_component_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "empty.apk"
            output = root / "out.apk"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("keep.txt", b"keep")
            with self.assertRaises(ValueError):
                inject_split_component.inject_components(source, output)


if __name__ == "__main__":
    unittest.main()
