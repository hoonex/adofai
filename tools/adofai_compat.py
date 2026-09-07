#!/usr/bin/env python3
"""Lossless-first ADOFAI level parser, inspector and normalizer.

This tool deliberately does not delete event/settings fields it does not understand.
The target game/editor decides whether a recognized event can actually render.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

PATH_DATA_TABLE: Mapping[str, int] = {
    "R": 0, "p": 15, "J": 30, "E": 45, "T": 60, "o": 75,
    "U": 90, "q": 105, "G": 120, "Q": 135, "H": 150, "W": 165,
    "L": 180, "x": 195, "N": 210, "Z": 225, "F": 240, "V": 255,
    "D": 270, "Y": 285, "B": 300, "C": 315, "M": 330, "A": 345,
    "!": 999,
}

PATH_OFFSET_TABLE: Mapping[str, int] = {
    "5": 72,
    "6": -72,
    "7": 52,
    "8": -52,
    "9": -30,
    "h": 120,
    "j": -120,
    "t": 60,
    "y": 300,
}

MODERN_EVENT_TYPES = frozenset({
    "SetSpeed", "Twirl", "Checkpoint", "MoveCamera", "CustomBackground",
    "ChangeTrack", "ColorTrack", "AnimateTrack", "RecolorTrack", "MoveTrack",
    "SetText", "Flash", "SetHitsound", "SetFilter", "SetFilterAdvanced",
    "SetPlanetRotation", "HallOfMirrors", "ShakeScreen", "MoveDecorations",
    "PositionTrack", "RepeatEvents", "Bloom", "Hold", "SetHoldSound",
    "SetConditionalEvents", "ScreenTile", "ScreenScroll", "EditorComment",
    "Bookmark", "CallMethod", "AddComponent", "PlaySound", "MultiPlanet",
    "FreeRoam", "FreeRoamTwirl", "FreeRoamRemove", "Pause", "AutoPlayTiles",
    "Hide", "ScaleMargin", "ScaleRadius", "Multitap", "TileDimensions",
    "KillPlayer", "ScalePlanets", "SetFloorIcon", "AddDecoration", "AddText",
    "AddObject", "SetObject", "SetDefaultText", "SetFrameRate", "AddParticle",
    "SetParticle", "EmitParticle", "SetInputEvent",
})


class AdoFaiFormatError(ValueError):
    pass


def _escape_raw_controls_in_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            code = ord(ch)
            if code < 0x20:
                escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
                out.append(escapes.get(ch, "\\u%04x" % code))
            else:
                out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escape = False

    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def sanitize_json_text(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    text = _escape_raw_controls_in_strings(text)
    return _strip_trailing_commas(text)


def parse_adofai_text(text: str) -> Dict[str, Any]:
    sanitized = sanitize_json_text(text)
    try:
        value = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        raise AdoFaiFormatError(
            f"invalid .adofai JSON after tolerant normalization at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise AdoFaiFormatError(".adofai root must be a JSON object")
    return value


def parse_adofai_bytes(data: bytes) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AdoFaiFormatError(".adofai file is not valid UTF-8") from exc
    return parse_adofai_text(text)


def load_adofai(path: Path) -> Dict[str, Any]:
    return parse_adofai_bytes(path.read_bytes())


def path_data_to_angle_data(path_data: str) -> List[int]:
    result: List[int] = []
    previous = 0
    for symbol in path_data:
        if symbol in PATH_DATA_TABLE:
            previous = PATH_DATA_TABLE[symbol]
        elif symbol in PATH_OFFSET_TABLE:
            previous = previous + PATH_OFFSET_TABLE[symbol]
        # Unknown historical symbols intentionally preserve the current angle.
        result.append(previous)
    return result


def _event_types(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    types = {
        str(item.get("eventType"))
        for item in items
        if isinstance(item, dict) and item.get("eventType") is not None
    }
    return sorted(types)


def inspect_level(level: Mapping[str, Any]) -> Dict[str, Any]:
    actions = level.get("actions", [])
    decorations = level.get("decorations", [])
    action_types = _event_types(actions)
    decoration_types = _event_types(decorations)
    all_types = sorted(set(action_types) | set(decoration_types))
    unknown_types = sorted(set(all_types) - MODERN_EVENT_TYPES)

    angle_data = level.get("angleData")
    path_data = level.get("pathData")
    tile_count = None
    path_encoding = "missing"
    if isinstance(angle_data, list):
        tile_count = len(angle_data)
        path_encoding = "angleData"
    elif isinstance(path_data, str):
        tile_count = len(path_data)
        path_encoding = "pathData"

    return {
        "pathEncoding": path_encoding,
        "tileCount": tile_count,
        "settingsCount": len(level.get("settings", {})) if isinstance(level.get("settings"), dict) else None,
        "actionCount": len(actions) if isinstance(actions, list) else None,
        "decorationCount": len(decorations) if isinstance(decorations, list) else None,
        "actionTypes": action_types,
        "decorationTypes": decoration_types,
        "unknownModernEventTypes": unknown_types,
        "topLevelKeys": sorted(level.keys()),
    }


def normalize_level(level: Mapping[str, Any], *, convert_path_data: bool = False) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(level))

    if convert_path_data and "angleData" not in normalized and isinstance(normalized.get("pathData"), str):
        normalized["angleData"] = path_data_to_angle_data(normalized["pathData"])
        del normalized["pathData"]

    if "settings" not in normalized:
        normalized["settings"] = {}
    if "actions" not in normalized:
        normalized["actions"] = []
    if "decorations" not in normalized:
        normalized["decorations"] = []

    if not isinstance(normalized["settings"], dict):
        raise AdoFaiFormatError("settings must be an object")
    if not isinstance(normalized["actions"], list):
        raise AdoFaiFormatError("actions must be an array")
    if not isinstance(normalized["decorations"], list):
        raise AdoFaiFormatError("decorations must be an array")
    if "angleData" in normalized and not isinstance(normalized["angleData"], list):
        raise AdoFaiFormatError("angleData must be an array")
    if "pathData" in normalized and not isinstance(normalized["pathData"], str):
        raise AdoFaiFormatError("pathData must be a string")

    return normalized


def dump_level(level: Mapping[str, Any]) -> str:
    return json.dumps(level, ensure_ascii=False, indent=4) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and normalize ADOFAI .adofai files without dropping unknown fields.")
    parser.add_argument("input", type=Path, help="input .adofai file")
    parser.add_argument("--normalize", metavar="OUTPUT", type=Path, help="write normalized JSON to this file")
    parser.add_argument("--convert-path-data", action="store_true", help="convert legacy pathData to angleData when angleData is absent")
    parser.add_argument("--fail-on-unknown-event", action="store_true", help="exit non-zero if an event type is not in the pinned modern inventory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = load_adofai(args.input)
    report = inspect_level(level)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.fail_on_unknown_event and report["unknownModernEventTypes"]:
        return 3

    if args.normalize is not None:
        normalized = normalize_level(level, convert_path_data=args.convert_path_data)
        args.normalize.parent.mkdir(parents=True, exist_ok=True)
        args.normalize.write_text(dump_level(normalized), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
