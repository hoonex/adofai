#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    required = (
        "pickerInFlight",
        "Unexpected trailing content after JSON value",
        "getOrCreateArrayForWrite",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"dirty-close prerequisites missing: {missing}")

    text = replace_exact(
        text,
        '''        close.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                if (dialog != null) dialog.dismiss();
            }
        });
''',
        '''        close.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                requestClose();
            }
        });
''',
        "Close button ownership",
    )

    text = replace_exact(
        text,
        '''        dialog.setContentView(buildEditorRoot(owner));
        dialog.setCanceledOnTouchOutside(false);
        dialog.show();
''',
        '''        dialog.setContentView(buildEditorRoot(owner));
        dialog.setCanceledOnTouchOutside(false);
        dialog.setOnKeyListener(new android.content.DialogInterface.OnKeyListener() {
            @Override public boolean onKey(
                    android.content.DialogInterface ignored,
                    int keyCode,
                    android.view.KeyEvent event) {
                if (keyCode != android.view.KeyEvent.KEYCODE_BACK) return false;
                if (event.getAction() == android.view.KeyEvent.ACTION_UP) requestClose();
                // Consume both DOWN and UP so Dialog's default Back handling cannot
                // dismiss a dirty editor before our confirmation path runs.
                return true;
            }
        });
        dialog.show();
''',
        "Android Back ownership",
    )

    text = replace_exact(
        text,
        '''    private static View buildEditorRoot(Activity owner) {
''',
        '''    private static void requestClose() {
        if (dialog == null || !dialog.isShowing()) return;
        if (!dirty) {
            dialog.dismiss();
            return;
        }

        final Activity owner = activity;
        if (owner == null || owner.isFinishing()) {
            setStatus("Unsaved changes: cannot close safely without a foreground Activity", true);
            return;
        }

        new android.app.AlertDialog.Builder(owner)
                .setTitle("Unsaved changes")
                .setMessage("This chart has unsaved changes. Keep editing, or explicitly discard them before closing.")
                .setPositiveButton("Discard & close", new android.content.DialogInterface.OnClickListener() {
                    @Override public void onClick(android.content.DialogInterface ignored, int which) {
                        // Explicit discard owns the in-memory session too. Otherwise
                        // reopening the shell could make discarded edits appear to
                        // survive even though they were never written to disk.
                        document = null;
                        currentPath = null;
                        dirty = false;
                        if (dialog != null) dialog.dismiss();
                    }
                })
                .setNegativeButton("Keep editing", null)
                .show();
    }

    private static View buildEditorRoot(Activity owner) {
''',
        "dirty close helper",
    )

    text = replace_exact(
        text,
        '''                loadPath(path);
''',
        '''                confirmOpenPath(path);
''',
        "Open replacement ownership",
    )

    text = replace_exact(
        text,
        '''    private static void beginSaveAs() {
''',
        '''    private static void confirmOpenPath(final String path) {
        if (!dirty) {
            loadPath(path);
            return;
        }

        final Activity owner = activity;
        if (owner == null || owner.isFinishing()) {
            setStatus("Unsaved changes: cannot replace the current chart safely without a foreground Activity", true);
            return;
        }

        new android.app.AlertDialog.Builder(owner)
                .setTitle("Unsaved changes")
                .setMessage("Opening another chart will replace the current in-memory edits. Keep editing, or explicitly discard them and open the selected chart.")
                .setPositiveButton("Discard & open", new android.content.DialogInterface.OnClickListener() {
                    @Override public void onClick(android.content.DialogInterface ignored, int which) {
                        // Do not clear the current session before the replacement has
                        // actually loaded. If the selected file is unreadable or
                        // malformed, loadPath fails closed and the dirty document stays.
                        loadPath(path);
                    }
                })
                .setNegativeButton("Keep editing", null)
                .show();
    }

    private static void beginSaveAs() {
''',
        "dirty open helper",
    )

    text = replace_exact(
        text,
        '''        if (dirty && !saveCurrent(false)) return;
''',
        '''        if (dirty) {
            confirmSaveAndPreview();
            return;
        }
''',
        "Preview autosave ownership",
    )

    text = replace_exact(
        text,
        '''    private static void previewCurrent() {
''',
        '''    private static void confirmSaveAndPreview() {
        final Activity owner = activity;
        if (owner == null || owner.isFinishing()) {
            setStatus("Unsaved changes: cannot confirm preview save without a foreground Activity", true);
            return;
        }

        new android.app.AlertDialog.Builder(owner)
                .setTitle("Save changes before preview?")
                .setMessage("Preview uses the chart file on disk. Save the current edits first, or keep editing without changing the file.")
                .setPositiveButton("Save & preview", new android.content.DialogInterface.OnClickListener() {
                    @Override public void onClick(android.content.DialogInterface ignored, int which) {
                        // saveCurrent clears dirty only after the atomic file replace
                        // succeeds. Recursing into previewCurrent is therefore safe and
                        // cannot loop on a failed save.
                        if (saveCurrent(false)) previewCurrent();
                    }
                })
                .setNegativeButton("Keep editing", null)
                .show();
    }

    private static void previewCurrent() {
''',
        "dirty preview helper",
    )

    return text


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.java> <output.java>", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    try:
        rendered = transform(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        print(f"dirty-session transform failed: {error}", file=sys.stderr)
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared dirty-session guarded mobile editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
