#!/usr/bin/env python3
"""Ensure decoded UnityPlayerActivity starts the injected editor runtime.

The native hook exports JNI_OnLoad, but replacing/injecting libOctober.so alone is
insufficient unless the game process loads it. The injected secondary DEX now owns
that load in MobileEditorBootstrap.init(). UnityPlayerActivity only needs a
zero-argument invoke-static, so this patch consumes no v-registers and never has to
renumber an existing .locals/.registers frame.
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
BOOTSTRAP_CALL = (
    "    invoke-static {}, "
    "Lcom/unity3d/player/MobileEditorBootstrap;->init()V\n"
)
DIRECT_OCTOBER_MARKERS = (
    '"October"',
    "System;->loadLibrary(Ljava/lang/String;)V",
)


def patch_smali_text(text: str) -> tuple[str, bool]:
    if "Lcom/unity3d/player/MobileEditorBootstrap;->init()V" in text:
        return text, False
    if all(marker in text for marker in DIRECT_OCTOBER_MARKERS):
        # Preserve an already-working legacy bootstrap instead of loading twice.
        return text, False

    match = ONCREATE_RE.search(text)
    if not match:
        raise ValueError("UnityPlayerActivity.onCreate(Bundle) not found")

    body = match.group("body")
    directive = re.search(r"(?m)^\s*\.(?:locals|registers)\s+\d+\s*$", body)
    if not directive:
        raise ValueError("onCreate has neither .locals nor .registers directive")

    insert_at = directive.end()
    new_body = body[:insert_at] + "\n\n" + BOOTSTRAP_CALL + body[insert_at:]
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
    parser = argparse.ArgumentParser(
        description="Patch UnityPlayerActivity to invoke the secondary-DEX editor bootstrap"
    )
    parser.add_argument("decoded_apk", type=Path, help="apktool-decoded APK directory")
    args = parser.parse_args()

    root = args.decoded_apk.resolve()
    activity, changed = patch_decoded_apk(root)
    action = "patched" if changed else "already configured"
    print(f"Editor bootstrap {action}: {activity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
