#!/usr/bin/env python3
"""Make Android back/cancel complete the pinned raw-path file chooser.

The upstream CustomFileChooser only calls FileSelector.setPath(null) from the
explicit negative button. Pressing Android Back cancels the AlertDialog through
Dialog.cancel(), bypassing that button listener and leaving FileSelector.isDone
false. The mobile editor then waits until its picker timeout.

This transform is pinned to the exact upstream HEAD and CustomFileChooser blob and
adds an OnCancelListener before the dialog is shown so every cancel path completes
with the same empty selection contract.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
FILE = "app/src/main/java/com/unity3d/player/CustomFileChooser.java"
EXPECTED_BLOB = "7aca3ef20ded3eb84b41b55edbbebffd158dc06d"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_identity(source: Path) -> None:
    head = git(source, "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"upstream HEAD mismatch: expected {EXPECTED_HEAD}, got {head}")
    blob = git(source, "hash-object", FILE)
    if blob != EXPECTED_BLOB:
        raise SystemExit(f"upstream blob mismatch for {FILE}: expected {EXPECTED_BLOB}, got {blob}")


def transform(text: str) -> str:
    old = '''        dialog = builder.create();
        dialog.show();
'''
    new = '''        dialog = builder.create();
        dialog.setOnCancelListener(new DialogInterface.OnCancelListener() {
            @Override public void onCancel(DialogInterface d) {
                // Android Back/outside cancellation does not invoke the negative
                // button listener. Complete the static bridge explicitly so native
                // and editor-shell waiters cannot remain stuck with isDone=false.
                FileSelector.setPath(null);
            }
        });
        dialog.show();
'''
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"picker dialog creation: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def apply(source: Path) -> None:
    verify_identity(source)
    path = source / FILE
    try:
        rendered = transform(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        raise SystemExit(f"picker cancel guard failed: {error}")
    path.write_text(rendered, encoding="utf-8")
    print(f"Applied picker cancel completion guard to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
