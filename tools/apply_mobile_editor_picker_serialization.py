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
    text = replace_exact(
        text,
        '''    private static boolean dirty;
    private static int currentTab;

    private static TextView pathView;
''',
        '''    private static boolean dirty;
    private static int currentTab;
    private static boolean pickerInFlight;

    private static TextView pathView;
''',
        "picker state declaration",
    )

    text = replace_exact(
        text,
        '''    private static void beginOpen() {
        setStatus("Opening file picker…", false);
        FileSelector.selectFile("adofai");
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Open cancelled", false);
                    return;
                }
                loadPath(path);
            }
        });
    }
''',
        '''    private static void beginOpen() {
        if (!beginPickerRequest()) return;
        setStatus("Opening file picker…", false);
        try {
            FileSelector.selectFile("adofai");
        } catch (Throwable error) {
            finishPickerRequest();
            reportError("Could not open file picker", error);
            return;
        }
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Open cancelled", false);
                    return;
                }
                loadPath(path);
            }
        });
    }
''',
        "Open picker serialization",
    )

    text = replace_exact(
        text,
        '''    private static void beginSaveAs() {
        if (document == null) {
            setStatus("No chart to save", true);
            return;
        }
        String name = "level.adofai";
        if (currentPath != null) name = new File(currentPath).getName();
        FileSelector.saveAs(name);
        setStatus("Choose a save path…", false);
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Save As cancelled", false);
                    return;
                }
                if (!path.toLowerCase().endsWith(".adofai")) path += ".adofai";
                saveToPath(path, true);
            }
        });
    }
''',
        '''    private static void beginSaveAs() {
        if (document == null) {
            setStatus("No chart to save", true);
            return;
        }
        if (!beginPickerRequest()) return;
        String name = "level.adofai";
        if (currentPath != null) name = new File(currentPath).getName();
        try {
            FileSelector.saveAs(name);
        } catch (Throwable error) {
            finishPickerRequest();
            reportError("Could not open Save As picker", error);
            return;
        }
        setStatus("Choose a save path…", false);
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Save As cancelled", false);
                    return;
                }
                if (!path.toLowerCase().endsWith(".adofai")) path += ".adofai";
                saveToPath(path, true);
            }
        });
    }
''',
        "Save As picker serialization",
    )

    text = replace_exact(
        text,
        '''    private interface PickerCompletion { void complete(String path); }

    private static void awaitPicker(final PickerCompletion completion) {
''',
        '''    private static boolean beginPickerRequest() {
        if (pickerInFlight) {
            setStatus("A file picker is already open", false);
            return false;
        }
        pickerInFlight = true;
        return true;
    }

    private static void finishPickerRequest() {
        pickerInFlight = false;
    }

    private interface PickerCompletion { void complete(String path); }

    private static void awaitPicker(final PickerCompletion completion) {
''',
        "picker ownership helpers",
    )

    text = replace_exact(
        text,
        '''                if (FileSelector.isDone) {
                    String value = FileSelector.getFilePath();
                    completion.complete(value == null ? "" : value);
                    return;
                }
                if (System.currentTimeMillis() >= deadline) {
                    setStatus("File picker timed out", true);
                    return;
                }
''',
        '''                if (FileSelector.isDone) {
                    String value = FileSelector.getFilePath();
                    finishPickerRequest();
                    completion.complete(value == null ? "" : value);
                    return;
                }
                if (System.currentTimeMillis() >= deadline) {
                    finishPickerRequest();
                    setStatus("File picker timed out", true);
                    return;
                }
''',
        "picker completion release",
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
        print(f"picker serialization transform failed: {error}", file=sys.stderr)
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared serialized-picker mobile editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
