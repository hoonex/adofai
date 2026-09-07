#!/usr/bin/env python3
"""Ensure decoded UnityPlayerActivity loads libOctober.so from onCreate.

The HitMargin native hook exports JNI_OnLoad, but replacing/injecting the ELF alone
is insufficient unless the game process loads the October library. Release 1.0.3
explicitly required a System.loadLibrary("October") call in UnityPlayerActivity.
This tool performs that bootstrap patch idempotently on apktool-decoded smali.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ACTIVITY_BASENAME = "UnityPlayerActivity.smali"
ONCREATE_RE = re.compile(
    r"(?ms)^\.method(?P<header>[^\n]*\bonCreate\(Landroid/os/Bundle;\)V)\n"
    r"(?P<body>.*?)^\.end method\s*$"
)
LOAD_SNIPPET = (
    '    const-string v0, "October"\n'
    '    invoke-static {v0}, Ljava/lang/System;->loadLibrary(Ljava/lang/String;)V\n'
)


def _ensure_one_local(method_body: str) -> str:
    locals_match = re.search(r"(?m)^(\s*)\.locals\s+(\d+)\s*$", method_body)
    if locals_match:
        current = int(locals_match.group(2))
        if current >= 1:
            return method_body
        start, end = locals_match.span()
        replacement = f"{locals_match.group(1)}.locals 1"
        return method_body[:start] + replacement + method_body[end:]

    registers_match = re.search(r"(?m)^(\s*)\.registers\s+(\d+)\s*$", method_body)
    if registers_match:
        # onCreate is an instance method with p0 + p1, so v0 requires >= 3 total regs.
        current = int(registers_match.group(2))
        if current >= 3:
            return method_body
        start, end = registers_match.span()
        replacement = f"{registers_match.group(1)}.registers 3"
        return method_body[:start] + replacement + method_body[end:]

    raise ValueError("onCreate has neither .locals nor .registers directive")


def patch_smali_text(text: str) -> tuple[str, bool]:
    if 'System;->loadLibrary(Ljava/lang/String;)V' in text and '"October"' in text:
        return text, False

    match = ONCREATE_RE.search(text)
    if not match:
        raise ValueError("UnityPlayerActivity.onCreate(Bundle) not found")

    body = _ensure_one_local(match.group("body"))
    directive = re.search(r"(?m)^\s*\.(?:locals|registers)\s+\d+\s*$", body)
    if not directive:
        raise ValueError("unable to locate register directive after normalization")

    insert_at = directive.end()
    new_body = body[:insert_at] + "\n\n" + LOAD_SNIPPET + body[insert_at:]
    replacement = ".method" + match.group("header") + "\n" + new_body + ".end method"
    return text[: match.start()] + replacement + text[match.end() :], True


def find_activity(decoded_root: Path) -> Path:
    candidates = sorted(decoded_root.glob(f"smali*/**/{ACTIVITY_BASENAME}"))
    if not candidates:
        raise SystemExit(f"{ACTIVITY_BASENAME} not found under {decoded_root}/smali*")
    if len(candidates) != 1:
        joined = "\n  ".join(str(path) for path in candidates)
        raise SystemExit(f"expected exactly one {ACTIVITY_BASENAME}, found {len(candidates)}:\n  {joined}")
    return candidates[0]


def patch_decoded_apk(decoded_root: Path) -> tuple[Path, bool]:
    activity = find_activity(decoded_root)
    original = activity.read_text(encoding="utf-8")
    patched, changed = patch_smali_text(original)
    if changed:
        activity.write_text(patched, encoding="utf-8")
    return activity, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch UnityPlayerActivity to load libOctober.so")
    parser.add_argument("decoded_apk", type=Path, help="apktool-decoded APK directory")
    args = parser.parse_args()

    root = args.decoded_apk.resolve()
    activity, changed = patch_decoded_apk(root)
    action = "patched" if changed else "already configured"
    print(f"October bootstrap {action}: {activity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
