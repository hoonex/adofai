#!/usr/bin/env python3
"""Inject the compiled mobile-editor payload into an unsigned/rebuilt APK.

This does not source or redistribute the game. It operates only on an APK supplied
by the caller, chooses a free classesN.dex slot without overwriting existing dex,
and replaces the arm64 editor hook library. The result still must be zipaligned and
signed before installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

DEX_RE = re.compile(r"^classes(?:(\d+))?\.dex$")
LIB_PATH = "lib/arm64-v8a/libOctober.so"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_payload(dex: Path, library: Path) -> None:
    if dex.read_bytes()[:4] != b"dex\n":
        raise ValueError(f"{dex} is not a DEX file")
    if library.read_bytes()[:4] != b"\x7fELF":
        raise ValueError(f"{library} is not an ELF shared library")


def next_dex_name(names: Iterable[str]) -> str:
    used = set()
    for name in names:
        match = DEX_RE.match(name)
        if not match:
            continue
        number = 1 if match.group(1) is None else int(match.group(1))
        used.add(number)
    candidate = 2
    while candidate in used:
        candidate += 1
    return f"classes{candidate}.dex"


def inject(input_apk: Path, output_apk: Path, dex: Path, library: Path) -> dict:
    validate_payload(dex, library)
    output_apk.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_apk, "r") as source:
        names = source.namelist()
        dex_name = next_dex_name(names)

        fd, temp_name = tempfile.mkstemp(prefix="adofai-inject-", suffix=".apk", dir=str(output_apk.parent))
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_path, "w") as target:
                for info in source.infolist():
                    if info.filename == LIB_PATH:
                        continue
                    # Old JAR signatures are invalid after mutation and can confuse
                    # v1 verification. v2/v3 signing blocks are outside ZIP entries
                    # and disappear when the archive is rewritten.
                    upper = info.filename.upper()
                    if upper.startswith("META-INF/") and upper.endswith((".RSA", ".DSA", ".EC", ".SF")):
                        continue
                    target.writestr(info, source.read(info.filename))

                dex_info = zipfile.ZipInfo(dex_name)
                dex_info.compress_type = zipfile.ZIP_DEFLATED
                dex_info.external_attr = 0o644 << 16
                target.writestr(dex_info, dex.read_bytes())

                lib_info = zipfile.ZipInfo(LIB_PATH)
                lib_info.compress_type = zipfile.ZIP_DEFLATED
                lib_info.external_attr = 0o644 << 16
                target.writestr(lib_info, library.read_bytes())

            temp_path.replace(output_apk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    with zipfile.ZipFile(output_apk, "r") as check:
        out_names = check.namelist()
        if dex_name not in out_names:
            raise RuntimeError("injected DEX missing from output APK")
        if LIB_PATH not in out_names:
            raise RuntimeError("injected native library missing from output APK")
        if check.read(dex_name)[:4] != b"dex\n":
            raise RuntimeError("injected DEX failed magic verification")
        if check.read(LIB_PATH)[:4] != b"\x7fELF":
            raise RuntimeError("injected native library failed ELF verification")

    return {
        "input": str(input_apk),
        "output": str(output_apk),
        "dexEntry": dex_name,
        "nativeEntry": LIB_PATH,
        "dexSha256": sha256(dex),
        "nativeSha256": sha256(library),
        "outputSha256": sha256(output_apk),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject ADOFAI mobile editor DEX/native payload into an APK")
    parser.add_argument("input_apk", type=Path)
    parser.add_argument("output_apk", type=Path)
    parser.add_argument("--dex", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()

    report = inject(args.input_apk, args.output_apk, args.dex, args.library)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
