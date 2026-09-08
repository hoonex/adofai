#!/usr/bin/env python3
"""Inject one mobile-editor component into a caller-supplied split APK.

ADOFAI 3.3.1 is delivered as a split install: managed code lives in base.apk,
large game assets live in an asset split, and arm64 native libraries live in an ABI
split. The single-APK injector cannot safely model that layout. This tool mutates
only the split that owns a requested component:

- --dex: add the picker DEX to the next free classesN.dex slot.
- --library: replace/add lib/arm64-v8a/libOctober.so as an uncompressed ELF.

The caller must zipalign modified APKs and sign every split with the same key before
installation. Unmodified asset splits do not pass through this ZIP rewrite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from inject_apk_payload import LIB_PATH, next_dex_name, sha256

SIGNATURE_SUFFIXES = (".RSA", ".DSA", ".EC", ".SF")


def _validate_dex(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(4) != b"dex\n":
            raise ValueError(f"{path} is not a DEX file")


def _validate_library(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"{path} is not an ELF shared library")


def _is_stale_signature_entry(name: str) -> bool:
    upper = name.upper()
    return upper.startswith("META-INF/") and upper.endswith(SIGNATURE_SUFFIXES)


def _copy_entry(source: zipfile.ZipFile, target: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    # Stream instead of source.read(info) so real ABI splits do not require holding
    # an 80+ MB libil2cpp.so entry in memory while being rewritten.
    with source.open(info, "r") as src, target.open(info, "w") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def inject_components(
    input_apk: Path,
    output_apk: Path,
    *,
    dex: Path | None = None,
    library: Path | None = None,
) -> dict:
    if dex is None and library is None:
        raise ValueError("at least one of dex or library is required")
    if dex is not None:
        _validate_dex(dex)
    if library is not None:
        _validate_library(library)

    output_apk.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_apk, "r") as source:
        source_names = source.namelist()
        dex_name = next_dex_name(source_names) if dex is not None else None

        fd, temp_name = tempfile.mkstemp(
            prefix="adofai-split-inject-", suffix=".apk", dir=str(output_apk.parent)
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_path, "w", allowZip64=True) as target:
                for info in source.infolist():
                    if _is_stale_signature_entry(info.filename):
                        continue
                    if library is not None and info.filename == LIB_PATH:
                        continue
                    _copy_entry(source, target, info)

                if dex is not None and dex_name is not None:
                    dex_info = zipfile.ZipInfo(dex_name)
                    dex_info.compress_type = zipfile.ZIP_DEFLATED
                    dex_info.external_attr = 0o644 << 16
                    target.writestr(dex_info, dex.read_bytes())

                if library is not None:
                    # Modern Play-delivered Unity native libs are uncompressed so
                    # Android can mmap them. zipalign -P 16 is applied afterward.
                    lib_info = zipfile.ZipInfo(LIB_PATH)
                    lib_info.compress_type = zipfile.ZIP_STORED
                    lib_info.external_attr = 0o644 << 16
                    target.writestr(lib_info, library.read_bytes())

            temp_path.replace(output_apk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    report: dict[str, object] = {
        "input": str(input_apk),
        "output": str(output_apk),
        "outputSha256": sha256(output_apk),
    }

    with zipfile.ZipFile(output_apk, "r") as check:
        names = check.namelist()
        if dex is not None and dex_name is not None:
            if dex_name not in names or check.read(dex_name)[:4] != b"dex\n":
                raise RuntimeError("injected DEX failed verification")
            report["dexEntry"] = dex_name
            report["dexSha256"] = sha256(dex)

        if library is not None:
            if LIB_PATH not in names or check.read(LIB_PATH)[:4] != b"\x7fELF":
                raise RuntimeError("injected native library failed verification")
            lib_info = check.getinfo(LIB_PATH)
            if lib_info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("injected native library must remain uncompressed")
            report["nativeEntry"] = LIB_PATH
            report["nativeSha256"] = sha256(library)
            report["nativeCompression"] = "stored"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject ADOFAI editor components into one split APK")
    parser.add_argument("input_apk", type=Path)
    parser.add_argument("output_apk", type=Path)
    parser.add_argument("--dex", type=Path)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()

    report = inject_components(
        args.input_apk,
        args.output_apk,
        dex=args.dex,
        library=args.library,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
