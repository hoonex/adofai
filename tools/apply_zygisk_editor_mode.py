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
        "requestClose",
        "confirmOpenPath",
        "confirmSaveAndPreview",
        "pickerInFlight",
        "Unexpected trailing content after JSON value",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"zygisk editor prerequisites missing: {missing}")

    text = replace_exact(
        text,
        '''        actions.addView(makeAction("Open", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpen(); }
        }));
''',
        '''        actions.addView(makeAction("New", new View.OnClickListener() {
            @Override public void onClick(View v) { newChart(); }
        }));
        actions.addView(makeAction("Open Map", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpen(); }
        }));
''',
        "New/Open action insertion",
    )

    text = replace_exact(
        text,
        '        setStatus("Opening file picker…", false);\n',
        '        setStatus("Choose the map folder containing a .adofai file…", false);\n',
        "Open folder status",
    )

    text = replace_exact(
        text,
        '''            Os.rename(temp.getAbsolutePath(), target.getAbsolutePath());
            currentPath = target.getAbsolutePath();
''',
        '''            Os.rename(temp.getAbsolutePath(), target.getAbsolutePath());
            if (!FileSelector.syncSavedPath(target.getAbsolutePath())) {
                throw new IllegalStateException("Could not synchronize saved chart to the selected Android document");
            }
            currentPath = target.getAbsolutePath();
''',
        "SAF save synchronization",
    )

    text = replace_exact(
        text,
        '''        String value = currentPath == null ? "No chart open" : currentPath;
''',
        '''        String value = currentPath == null ? "No chart open" : FileSelector.displayNameForPath(currentPath);
''',
        "working path display name",
    )

    text = replace_exact(
        text,
        '''    private static void markDirty(String message) {
''',
        '''    private static void newChart() {
        if (dirty) {
            setStatus("Save or discard the current changes before creating a new chart", true);
            return;
        }
        try {
            JSONObject fresh = new JSONObject();
            fresh.put("pathData", "R");
            JSONObject settings = new JSONObject();
            settings.put("version", 15);
            settings.put("bpm", 100);
            settings.put("song", "Untitled");
            fresh.put("settings", settings);
            fresh.put("actions", new JSONArray());
            fresh.put("decorations", new JSONArray());
            document = fresh;
            currentPath = null;
            dirty = true;
            updatePath();
            showTab(0);
            setStatus("New chart created. Use Save As to choose a .adofai document", false);
        } catch (JSONException error) {
            reportError("New chart failed", error);
        }
    }

    private static void markDirty(String message) {
''',
        "New chart helper",
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
        print(f"zygisk editor transform failed: {error}", file=sys.stderr)
        return 3
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared Zygisk in-game editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
