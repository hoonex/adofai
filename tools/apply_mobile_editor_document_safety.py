#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED_SOURCE_BLOB = "629a33f4860873064f59688c8a7ac32931e28de7"


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    text = replace_exact(
        text,
        '''            JSONObject parsed = new JSONObject(sanitizeJson(raw));
            if (!(parsed.opt("settings") instanceof JSONObject)) parsed.put("settings", new JSONObject());
            if (!(parsed.opt("actions") instanceof JSONArray)) parsed.put("actions", new JSONArray());
            if (!(parsed.opt("decorations") instanceof JSONArray)) parsed.put("decorations", new JSONArray());
            document = parsed;
''',
        '''            JSONObject parsed = new JSONObject(sanitizeJson(raw));
            // Opening a chart is read-only with respect to parsed data. Missing or
            // future/malformed structured fields are preserved verbatim; the Raw
            // tab remains available and structured tabs fail closed instead of
            // silently normalizing data merely because the chart was opened.
            document = parsed;
''',
        "loadPath normalization",
    )

    text = replace_exact(
        text,
        '''        final JSONObject settings;
        try {
            JSONObject existing = document.optJSONObject("settings");
            settings = existing != null ? existing : new JSONObject();
            if (existing == null) document.put("settings", settings);
        } catch (JSONException error) {
            reportError("Could not create settings object", error);
            return scroll(body);
        }
''',
        '''        Object storedSettings = document.opt("settings");
        if (storedSettings != null && storedSettings != JSONObject.NULL && !(storedSettings instanceof JSONObject)) {
            body.addView(text("This chart's settings field is not an object. Structured Settings editing is disabled to preserve it exactly; use Raw to inspect or explicitly replace it.", 12, Color.rgb(245, 170, 120)));
            return scroll(body);
        }
        final JSONObject settings = storedSettings instanceof JSONObject ? (JSONObject) storedSettings : new JSONObject();
''',
        "settings read path",
    )

    text = replace_exact(
        text,
        '''                try {
                    settings.put(key, parseJsonValue(newValue.getText().toString()));
                    markDirty("Setting added: " + key);
''',
        '''                try {
                    // Missing/null settings are attached only after an explicit Add.
                    if (document.opt("settings") != settings) document.put("settings", settings);
                    settings.put(key, parseJsonValue(newValue.getText().toString()));
                    markDirty("Setting added: " + key);
''',
        "settings explicit creation",
    )

    text = replace_exact(
        text,
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getOrCreateArray(name);
                List<String> labels = new ArrayList<String>();
                for (int i = 0; i < array.length(); i++) {
                    JSONObject event = array.optJSONObject(i);
                    if (event == null) labels.add(i + ": <non-object>");
                    else labels.add(i + ": " + event.optString("eventType", "<unknown>") + "  floor=" + event.opt("floor"));
                }
''',
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                Object stored = document.opt(name);
                JSONArray array = stored instanceof JSONArray ? (JSONArray) stored : null;
                List<String> labels = new ArrayList<String>();
                if (stored != null && stored != JSONObject.NULL && array == null) {
                    labels.add("<preserved non-array " + name + "; use Raw tab>");
                } else if (array != null) {
                    for (int i = 0; i < array.length(); i++) {
                        JSONObject event = array.optJSONObject(i);
                        if (event == null) labels.add(i + ": <non-object>");
                        else labels.add(i + ": " + event.optString("eventType", "<unknown>") + "  floor=" + event.opt("floor"));
                    }
                }
''',
        "events refresh read path",
    )

    text = replace_exact(
        text,
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getOrCreateArray(name);
                JSONObject event = array.optJSONObject(position);
                selected[0] = position;
                raw.setText(prettyJson(event == null ? array.opt(position) : event));
''',
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getExistingArray(name);
                if (array == null || position < 0 || position >= array.length()) {
                    selected[0] = -1;
                    raw.setText("");
                    return;
                }
                JSONObject event = array.optJSONObject(position);
                selected[0] = position;
                raw.setText(prettyJson(event == null ? array.opt(position) : event));
''',
        "events selection read path",
    )

    text = replace_exact(
        text,
        '''                    String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                    JSONArray array = getOrCreateArray(name);
                    array.put(selected[0], replacement);
''',
        '''                    String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                    JSONArray array = getExistingArray(name);
                    if (array == null || selected[0] >= array.length()) throw new JSONException(name + " is not an editable array");
                    array.put(selected[0], replacement);
''',
        "events explicit apply",
    )

    text = replace_exact(
        text,
        '''                    getOrCreateArray(decoration ? "decorations" : "actions").put(event);
''',
        '''                    getOrCreateArrayForWrite(decoration ? "decorations" : "actions").put(event);
''',
        "events explicit add",
    )

    text = replace_exact(
        text,
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                getOrCreateArray(name).remove(selected[0]);
                markDirty("Event object deleted");
''',
        '''                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getExistingArray(name);
                if (array == null || selected[0] >= array.length()) {
                    setStatus("Event list is not an editable array; use Raw to inspect it", true);
                    return;
                }
                array.remove(selected[0]);
                markDirty("Event object deleted");
''',
        "events explicit delete",
    )

    text = replace_exact(
        text,
        '''    private static JSONArray getOrCreateArray(String key) {
        JSONArray value = document.optJSONArray(key);
        if (value != null) return value;
        value = new JSONArray();
        try { document.put(key, value); }
        catch (JSONException impossible) { throw new IllegalStateException(impossible); }
        return value;
    }
''',
        '''    private static JSONArray getExistingArray(String key) {
        Object value = document.opt(key);
        return value instanceof JSONArray ? (JSONArray) value : null;
    }

    private static JSONArray getOrCreateArrayForWrite(String key) throws JSONException {
        Object existing = document.opt(key);
        if (existing instanceof JSONArray) return (JSONArray) existing;
        if (existing != null && existing != JSONObject.NULL) {
            throw new JSONException(key + " is not an array; use Raw tab to replace it explicitly");
        }
        JSONArray created = new JSONArray();
        document.put(key, created);
        return created;
    }
''',
        "array ownership helper",
    )

    return text


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <MobileEditorShell.java> <output.java>", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    data = source.read_bytes()
    actual_blob = git_blob_sha(data)
    if actual_blob != EXPECTED_SOURCE_BLOB:
        print(
            f"refusing document-safety transform: source blob {actual_blob} != expected {EXPECTED_SOURCE_BLOB}",
            file=sys.stderr,
        )
        return 3

    try:
        rendered = transform(data.decode("utf-8"))
    except (UnicodeDecodeError, RuntimeError) as error:
        print(f"document-safety transform failed: {error}", file=sys.stderr)
        return 4

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared lossless-first mobile editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
