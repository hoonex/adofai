#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(text: str, old: str, new: str, label: str, expected: int = 1) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} source match(es), found {count}")
    return text.replace(old, new)


def transform(text: str) -> str:
    text = replace_exact(
        text,
        '        if (normalized.startsWith("[")) return new JSONArray(normalized);\n',
        '        if (normalized.startsWith("[")) return parseJsonArray(normalized);\n',
        "angleData array parsing",
    )

    text = replace_exact(
        text,
        '                    JSONObject replacement = new JSONObject(sanitizeJson(raw.getText().toString()));\n',
        '                    JSONObject replacement = parseJsonObject(raw.getText().toString());\n',
        "structured/raw object parsing",
        expected=2,
    )

    text = replace_exact(
        text,
        '            JSONObject parsed = new JSONObject(sanitizeJson(raw));\n',
        '            JSONObject parsed = parseJsonObject(raw);\n',
        "chart root parsing",
    )

    text = replace_exact(
        text,
        '''    private static Object parseJsonValue(String text) throws JSONException {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.length() == 0) return "";
        return new JSONTokener(trimmed).nextValue();
    }
''',
        '''    private static Object parseJsonValue(String text) throws JSONException {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.length() == 0) return "";
        JSONTokener tokener = new JSONTokener(sanitizeJson(trimmed));
        Object value = tokener.nextValue();
        requireJsonEof(tokener);
        return value;
    }

    private static JSONObject parseJsonObject(String text) throws JSONException {
        JSONTokener tokener = new JSONTokener(sanitizeJson(text));
        Object value = tokener.nextValue();
        if (!(value instanceof JSONObject)) throw new JSONException("Expected a JSON object");
        requireJsonEof(tokener);
        return (JSONObject) value;
    }

    private static JSONArray parseJsonArray(String text) throws JSONException {
        JSONTokener tokener = new JSONTokener(sanitizeJson(text));
        Object value = tokener.nextValue();
        if (!(value instanceof JSONArray)) throw new JSONException("Expected a JSON array");
        requireJsonEof(tokener);
        return (JSONArray) value;
    }

    private static void requireJsonEof(JSONTokener tokener) throws JSONException {
        if (tokener.nextClean() != '\\0') throw new JSONException("Unexpected trailing content after JSON value");
    }
''',
        "strict JSON helpers",
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
        print(f"strict JSON transform failed: {error}", file=sys.stderr)
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared strict-JSON mobile editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
