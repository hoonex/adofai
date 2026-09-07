#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"


def transform(path: Path) -> None:
    main = path / "app/src/main/jni/Main.cpp"
    text = main.read_text(encoding="utf-8")
    required = (
        "InstallAndroidFilePickerBridge(true);",
        "InstallMobileEditorPreviewBridge();",
        "Modern runtime platform + mobile editor shell installed",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"zygisk native prerequisites missing: {missing}")

    old = '''        InstallAndroidFilePickerBridge(true);
        InstallMobileEditorPreviewBridge();
        LOGD("Modern runtime platform + mobile editor shell installed");
        return;
'''
    new = '''        // Zygisk owns the Java editor DEX and SAF bootstrap. Keep the native
        // payload limited to the current-runtime preview bridge so it never needs
        // to resolve injected Java classes through the app's default class loader.
        InstallMobileEditorPreviewBridge();
        LOGD("Modern runtime preview bridge installed; Zygisk owns Java editor bootstrap");
        return;
'''
    if text.count(old) != 1:
        raise RuntimeError(f"zygisk modern profile anchor count was {text.count(old)}")
    main.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <prepared-upstream-root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    try:
        transform(root)
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        print(f"zygisk native transform failed: {error}", file=sys.stderr)
        return 3
    print(f"Prepared Zygisk-only native preview payload at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
