#!/usr/bin/env python3
"""Lossless-first ADOFAI level parser, inspector and normalizer.

This tool deliberately does not delete event/settings fields it does not understand.
The target game/editor decides whether a recognized event can actually render or execute.

The v2.4 compatibility report is intentionally conservative. It distinguishes serialization
backports that can be applied without changing chart meaning from newer event semantics that
must be preserved for later runtime emulation or manual/device verification.
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

# Events confirmed to have been introduced after the v2.4 baseline, or in a later-era feature
# family. We preserve these objects exactly. `policy` describes what the v2.4 compatibility
# layer is currently allowed to do; it is not a claim that v2.4 executes the event natively.
V240_POST_BASELINE_EVENT_POLICY: Mapping[str, Mapping[str, Any]] = {
    "SetFilterAdvanced": {
        "introduced": "2.8.0",
        "domain": "visual",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "advanced filter semantics have no proven lossless v2.4 mapping",
    },
    "SetFrameRate": {
        "introduced": "2.8.0",
        "domain": "rendering",
        "policy": "native_emulation_candidate",
        "runtimeBackportImplemented": False,
        "reason": "render-frame pacing is separable from judgement timing, but chart-trigger execution is not implemented yet",
    },
    "AddParticle": {
        "introduced": "2.8-era",
        "domain": "visual",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "particle-event family introduction is later than the v2.4 baseline; exact per-event build is not pinned",
    },
    "SetParticle": {
        "introduced": "2.8-era",
        "domain": "visual",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "particle-event family introduction is later than the v2.4 baseline; exact per-event build is not pinned",
    },
    "EmitParticle": {
        "introduced": "2.8-era",
        "domain": "visual",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "particle-event family introduction is later than the v2.4 baseline; exact per-event build is not pinned",
    },
    "SetInputEvent": {
        "introduced": "2.9.0",
        "domain": "gameplay_trigger",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "input-trigger semantics affect gameplay and must never be stripped or guessed",
    },
    "TileDimensions": {
        "introduced": "2.9.7-era",
        "domain": "gameplay_geometry",
        "policy": "preserve_only",
        "runtimeBackportImplemented": False,
        "reason": "tile geometry can change gameplay; exact introduction/build semantics are not pinned",
    },
}

# These event types exist in/around the v2.4 feature set, but later releases changed their
# runtime or partial-update semantics. Presence is therefore a warning, not an automatic rewrite.
V240_SEMANTIC_DRIFT_POLICY: Mapping[str, Mapping[str, Any]] = {
    "FreeRoam": {
        "changed": "2.9.7",
        "domain": "timing",
        "policy": "preserve_and_verify",
        "reason": "duration semantics were unified with ordinary level events after v2.4",
    },
    "Pause": {
        "changed": "2.9.3/2.9.7",
        "domain": "timing_runtime_state",
        "policy": "preserve_and_verify",
        "reason": "restart reset and duration/beat behavior changed in later releases",
    },
    "SetConditionalEvents": {
        "changed": "2.9.3",
        "domain": "runtime_state",
        "policy": "preserve_and_verify",
        "reason": "failure/restart transient-state reset behavior changed after v2.4",
    },
    "RepeatEvents": {
        "changed": "2.6.0/3.1.0",
        "domain": "event_dispatch",
        "policy": "preserve_and_verify",
        "reason": "repeat behavior was improved and later gained gap-length semantics; exact serialized gap key is intentionally not guessed",
    },
    "ColorTrack": {
        "changed": "3.3.1",
        "domain": "partial_update",
        "policy": "preserve_and_verify",
        "reason": "later optional-property semantics may differ from v2.4 full-state updates",
    },
    "RecolorTrack": {
        "changed": "3.3.1",
        "domain": "partial_update_asset",
        "policy": "preserve_and_verify",
        "reason": "later optional properties and texture state have no proven lossless v2.4 mapping",
    },
    "MoveDecorations": {
        "changed": "2.6.0/2.7.0",
        "domain": "partial_update",
        "policy": "preserve_and_verify",
        "reason": "relative-to-last-position and axis/scale partial updates were added after the baseline",
    },
    "MoveTrack": {
        "changed": "2.6.0/2.7.0",
        "domain": "partial_update",
        "policy": "preserve_and_verify",
        "reason": "axis/scale partial-update semantics were expanded after the baseline",
    },
    "MoveCamera": {
        "changed": "2.6.0",
        "domain": "partial_update",
        "policy": "preserve_and_verify",
        "reason": "independent X/Y updates were added after the baseline",
    },
}


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


def _policy_records(event_types: Iterable[str], policy: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for event_type in sorted(set(event_types)):
        data = policy.get(event_type)
        if data is None:
            continue
        record = {"eventType": event_type}
        record.update(dict(data))
        records.append(record)
    return records


def v240_compatibility_report(level: Mapping[str, Any]) -> Dict[str, Any]:
    action_types = _event_types(level.get("actions", []))
    decoration_types = _event_types(level.get("decorations", []))
    all_types = sorted(set(action_types) | set(decoration_types))

    post_baseline = _policy_records(all_types, V240_POST_BASELINE_EVENT_POLICY)
    semantic_drift = _policy_records(all_types, V240_SEMANTIC_DRIFT_POLICY)
    preserve_only = sorted(
        record["eventType"] for record in post_baseline
        if record.get("policy") == "preserve_only"
    )
    native_candidates = sorted(
        record["eventType"] for record in post_baseline
        if record.get("policy") == "native_emulation_candidate"
    )
    gameplay_risks = sorted(
        record["eventType"] for record in post_baseline
        if record.get("domain") in {"gameplay_trigger", "gameplay_geometry"}
    )

    return {
        "target": "ADOFAI v2.4.0 Custom Android",
        "serializationBackport": "known_v2.4_togglebool_fields_only",
        "unknownEventsAreDeleted": False,
        "postV240Events": post_baseline,
        "semanticDriftEvents": semantic_drift,
        "preserveOnlyEventTypes": preserve_only,
        "nativeEmulationCandidates": native_candidates,
        "gameplayMeaningRiskEventTypes": gameplay_risks,
        "requiresRuntimeReview": bool(post_baseline or semantic_drift),
    }


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
        "v240Compatibility": v240_compatibility_report(level),
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
    parser.add_argument("--fail-on-v240-runtime-review", action="store_true", help="exit non-zero when the chart contains post-v2.4 or later-semantics events that require runtime review")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = load_adofai(args.input)
    report = inspect_level(level)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.fail_on_unknown_event and report["unknownModernEventTypes"]:
        return 3
    if args.fail_on_v240_runtime_review and report["v240Compatibility"]["requiresRuntimeReview"]:
        return 4

    if args.normalize is not None:
        normalized = normalize_level(level, convert_path_data=args.convert_path_data)
        args.normalize.parent.mkdir(parents=True, exist_ok=True)
        args.normalize.write_text(dump_level(normalized), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
