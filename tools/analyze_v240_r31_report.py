#!/usr/bin/env python3
"""Analyze bounded r31 editor pointer transactions from a v2.4 compatibility report.

The analyzer is intentionally observational. It classifies the method-boundary path that
was recorded; it does not claim a root cause or recommend a mutation without device evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_HOOK_MASK = (1 << 7) - 1
TXN_RE = re.compile(r"^editorTraceR31Txn(?P<slot>\d+)=(?P<body>.*)$")

PAIR_FIELDS = ("start", "end", "release", "maxd", "obj", "smart", "gizmo")
TRIPLE_FIELDS = ("ray", "r1", "r2", "select", "drag")
INT_FIELDS = ("seq", "state", "touch", "hf", "held")


def _int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field}: expected integer, got {value!r}") from exc


def _tuple(value: str, size: int, field: str) -> list[int]:
    parts = value.split(":")
    if len(parts) != size:
        raise ValueError(f"{field}: expected {size} colon-separated integers, got {value!r}")
    return [_int(part, field) for part in parts]


def _screen(value: str) -> list[int]:
    parts = value.split("x")
    if len(parts) != 2:
        raise ValueError(f"screen: expected WIDTHxHEIGHT, got {value!r}")
    return [_int(parts[0], "screen"), _int(parts[1], "screen")]


def parse_transaction(slot: int, body: str) -> tuple[dict[str, Any] | None, str | None]:
    raw: dict[str, str] = {}
    try:
        for part in body.split(","):
            if ":" not in part:
                raise ValueError(f"transaction field has no colon: {part!r}")
            name, value = part.split(":", 1)
            if not name:
                raise ValueError("transaction field has empty name")
            raw[name] = value

        txn: dict[str, Any] = {"slot": slot}
        for field in INT_FIELDS:
            if field not in raw:
                raise ValueError(f"missing transaction field: {field}")
            txn[field] = _int(raw[field], field)
        if "screen" not in raw:
            raise ValueError("missing transaction field: screen")
        txn["screen"] = _screen(raw["screen"])

        for field in PAIR_FIELDS:
            if field not in raw:
                raise ValueError(f"missing transaction field: {field}")
            txn[field] = _tuple(raw[field], 2, field)
        for field in TRIPLE_FIELDS:
            if field not in raw:
                raise ValueError(f"missing transaction field: {field}")
            txn[field] = _tuple(raw[field], 3, field)

        known = set(INT_FIELDS) | set(PAIR_FIELDS) | set(TRIPLE_FIELDS) | {"screen"}
        txn["unknown_fields"] = {
            key: value for key, value in raw.items() if key not in known
        }
        return txn, None
    except ValueError as exc:
        return None, f"slot {slot}: {exc}"


def parse_report(text: str) -> dict[str, Any]:
    metadata: dict[str, str] = {}
    txns: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TXN_RE.match(line)
        if match:
            txn, error = parse_transaction(int(match.group("slot")), match.group("body"))
            if error:
                warnings.append(error)
            elif txn is not None and txn["seq"] > 0:
                txns.append(txn)
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    txns.sort(key=lambda txn: txn["seq"])
    return {"metadata": metadata, "transactions": txns, "warnings": warnings}


def _meta_int(metadata: dict[str, str], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def runtime_gate(parsed: dict[str, Any]) -> dict[str, Any]:
    metadata = parsed["metadata"]
    problems: list[str] = []

    revision = _meta_int(metadata, "editorTraceRevision")
    if revision != 31:
        problems.append(f"editorTraceRevision expected 31, got {metadata.get('editorTraceRevision', 'missing')}")

    if metadata.get("editorTraceR31Policy") != "bounded-read-only-pointer-transaction":
        problems.append("editorTraceR31Policy missing or unexpected")
    if _meta_int(metadata, "editorTraceR31Mutation") != 0:
        problems.append("editorTraceR31Mutation is not proven 0")
    if _meta_int(metadata, "editorTraceR31InputAbiGuard") != 1:
        problems.append("editorTraceR31InputAbiGuard is not 1")
    if _meta_int(metadata, "editorTraceR31AbiMask") != EXPECTED_HOOK_MASK:
        problems.append(
            f"editorTraceR31AbiMask is not full 0x{EXPECTED_HOOK_MASK:x}"
        )
    if _meta_int(metadata, "editorTraceR31InstalledMask") != EXPECTED_HOOK_MASK:
        problems.append(
            f"editorTraceR31InstalledMask is not full 0x{EXPECTED_HOOK_MASK:x}"
        )
    if _meta_int(metadata, "editorTraceR31InstallAttempted") != 1:
        problems.append("editorTraceR31InstallAttempted is not 1")
    if _meta_int(metadata, "editorTraceR31MarkerReady") != 1:
        problems.append("editorTraceR31MarkerReady is not 1")
    if _meta_int(metadata, "editorTraceR31RecoveryState") != 0:
        problems.append("editorTraceR31RecoveryState is not 0")

    native_probe = metadata.get("nativeProbe")
    if native_probe in {"not-loaded", "unavailable"}:
        problems.append(f"nativeProbe={native_probe}")

    return {
        "ready": not problems,
        "problems": problems,
        "abi_mask": _meta_int(metadata, "editorTraceR31AbiMask"),
        "installed_mask": _meta_int(metadata, "editorTraceR31InstalledMask"),
        "call_proven_mask": _meta_int(metadata, "editorTraceR31CallProvenMask"),
        "latest_seq": _meta_int(metadata, "editorTraceR31LatestSeq"),
        "active_version": metadata.get("activeVersion"),
        "runtime_readiness": metadata.get("runtimeReadiness"),
    }


def classify_transaction(txn: dict[str, Any]) -> dict[str, Any]:
    state = txn["state"]
    obj_calls, obj_last = txn["obj"]
    ray_calls, ray1_count, ray2_count = txn["ray"]
    smart_calls, smart_non_null = txn["smart"]
    gizmo_calls, gizmo_non_null = txn["gizmo"]
    select_calls, select_non_null, camera_jump = txn["select"]
    drag_camera, drag_tiles_start, drag_tiles = txn["drag"]

    world_query = obj_calls > 0 or ray_calls > 0
    positive_world_hit = obj_last > 0 or ray1_count > 0 or ray2_count > 0
    explicit_paths = sum(
        (
            select_calls > 0,
            drag_camera > 0,
            drag_tiles_start > 0 or drag_tiles > 0,
        )
    )

    if state == 1:
        route = "ACTIVE_TRANSACTION"
        boundary = "Transaction is still active; recopy the report after the tap has settled."
    elif state == 3:
        route = "SUPERSEDED_TRANSACTION"
        boundary = "A newer pointer-down replaced this transaction before normal completion; do not use it alone to localize the failure."
    elif explicit_paths > 1:
        route = "MIXED_EXPLICIT_PATH"
        boundary = "More than one explicit selection/drag path ran in the same transaction; preserve this transaction for exact branch reconstruction."
    elif select_calls > 0:
        route = "SELECT_FLOOR_REACHED"
        if select_non_null > 0:
            boundary = "SelectFloor received a non-null floor. If visible selection failed, inspect state after SelectFloor rather than the raycast boundary."
        else:
            boundary = "SelectFloor was called with no observed non-null floor argument; inspect the caller/arbitration boundary."
    elif drag_tiles_start > 0 or drag_tiles > 0:
        route = "TILE_DRAG_REACHED"
        boundary = "The pointer transaction entered tile-drag behavior; inspect tap-versus-drag arbitration."
    elif drag_camera > 0:
        route = "CAMERA_DRAG_REACHED"
        boundary = "The pointer transaction entered camera-drag behavior; inspect tap-versus-camera arbitration."
    elif positive_world_hit:
        route = "WORLD_HIT_WITHOUT_SELECT"
        if smart_non_null > 0 or gizmo_non_null > 0:
            boundary = "World picking returned a hit and an alternate object/gizmo path also returned non-null, but SelectFloor was not called; inspect post-hit object arbitration."
        else:
            boundary = "World picking returned a positive hit but SelectFloor was not called; inspect the post-hit selection/arbitration branch."
    elif world_query:
        ray_counts = [ray1_count, ray2_count]
        ray_sentinel_error = any(count in {-1, -2} for count in ray_counts)
        zero_hit_proven = (
            obj_calls > 0
            and obj_last == 0
            and not ray_sentinel_error
        ) or (
            obj_calls == 0
            and ray_calls > 0
            and not ray_sentinel_error
            and any(count == 0 for count in ray_counts)
            and all(count in {-3, 0} for count in ray_counts)
        )
        if zero_hit_proven:
            route = "WORLD_QUERY_NO_HIT"
            boundary = "World picking ran with no positive observed hit; inspect coordinate conversion, collider/layer eligibility, and query inputs from this same transaction."
        else:
            route = "WORLD_QUERY_AMBIGUOUS"
            boundary = "World picking ran, but result counters are not sufficient for a positive/zero-hit conclusion; inspect the raw transaction before changing behavior."
    elif smart_calls > 0 or gizmo_calls > 0:
        route = "NON_FLOOR_EDITOR_PATH"
        boundary = "SmartObjectSelect/GizmoAtMouse ran without an observed world-query/select/drag path; inspect editor object arbitration before floor selection."
    else:
        route = "PRE_WORLD_QUERY_PATH"
        boundary = "No world query, SelectFloor, or drag path was observed; inspect earlier HandleMouseActions/UI/mode gating."

    return {
        **txn,
        "state_name": {0: "empty", 1: "active", 2: "completed", 3: "superseded"}.get(
            state, f"unknown-{state}"
        ),
        "route": route,
        "boundary": boundary,
        "world_query": world_query,
        "positive_world_hit": positive_world_hit,
        "pointer_max_delta_px": [txn["maxd"][0] / 100.0, txn["maxd"][1] / 100.0],
        "screen_start_px": [txn["start"][0] / 100.0, txn["start"][1] / 100.0],
        "screen_end_px": [txn["end"][0] / 100.0, txn["end"][1] / 100.0],
        "world_ray1": [txn["r1"][0] / 1000.0, txn["r1"][1] / 1000.0],
        "world_ray2": [txn["r2"][0] / 1000.0, txn["r2"][1] / 1000.0],
        "release_seen": bool(txn["release"][0]),
        "post_release_frames": txn["release"][1],
        "smart_non_null": smart_non_null,
        "gizmo_non_null": gizmo_non_null,
        "select_non_null": select_non_null,
        "select_camera_jump": bool(camera_jump),
    }


def analyze_report(text: str) -> dict[str, Any]:
    parsed = parse_report(text)
    gate = runtime_gate(parsed)
    txns = [classify_transaction(txn) for txn in parsed["transactions"]]
    route_counts = Counter(txn["route"] for txn in txns)

    latest_seq = gate["latest_seq"]
    warnings = list(parsed["warnings"])
    if latest_seq is not None and txns and max(txn["seq"] for txn in txns) != latest_seq:
        warnings.append(
            "editorTraceR31LatestSeq does not match the newest parsed transaction; report may be partial"
        )

    if not gate["ready"]:
        status = "RUNTIME_NOT_READY"
    elif not txns:
        status = "NO_TRANSACTIONS"
    elif any(txn["state"] in {1, 3} for txn in txns):
        status = "PARTIAL_TRANSACTIONS"
    else:
        status = "ANALYZED"

    return {
        "schema": 1,
        "status": status,
        "runtime": gate,
        "transaction_count": len(txns),
        "route_counts": dict(sorted(route_counts.items())),
        "transactions": txns,
        "warnings": warnings,
    }


def format_text(result: dict[str, Any]) -> str:
    runtime = result["runtime"]
    lines = [
        f"status={result['status']}",
        "runtime="
        + ("ready" if runtime["ready"] else "not-ready")
        + f" abiMask={runtime['abi_mask']} installedMask={runtime['installed_mask']}"
        + f" latestSeq={runtime['latest_seq']}",
    ]
    if runtime["active_version"]:
        lines.append(f"activeVersion={runtime['active_version']}")
    if runtime["runtime_readiness"]:
        lines.append(f"runtimeReadiness={runtime['runtime_readiness']}")
    for problem in runtime["problems"]:
        lines.append(f"runtimeProblem={problem}")
    for warning in result["warnings"]:
        lines.append(f"warning={warning}")

    if not result["transactions"]:
        lines.append("transactions=none")
    else:
        lines.append("routeCounts=" + json.dumps(result["route_counts"], sort_keys=True))
        for txn in result["transactions"]:
            lines.append(
                "txn"
                f" seq={txn['seq']} slot={txn['slot']} state={txn['state_name']}"
                f" route={txn['route']}"
                f" maxDeltaPx={txn['pointer_max_delta_px'][0]:.2f}:{txn['pointer_max_delta_px'][1]:.2f}"
                f" obj={txn['obj'][0]}:{txn['obj'][1]}"
                f" ray={txn['ray'][0]}:{txn['ray'][1]}:{txn['ray'][2]}"
                f" smart={txn['smart'][0]}:{txn['smart'][1]}"
                f" gizmo={txn['gizmo'][0]}:{txn['gizmo'][1]}"
                f" select={txn['select'][0]}:{txn['select'][1]}:{txn['select'][2]}"
                f" drag={txn['drag'][0]}:{txn['drag'][1]}:{txn['drag'][2]}"
            )
            lines.append(f"  boundary={txn['boundary']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze r31 pointer transactions from an ADOFAI v2.4 compatibility report."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Report text file. Omit or use '-' to read from stdin.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.report in (None, "-"):
        text = sys.stdin.read()
    else:
        text = Path(args.report).read_text(encoding="utf-8")

    result = analyze_report(text)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(format_text(result))

    return 0 if result["status"] in {"ANALYZED", "PARTIAL_TRANSACTIONS", "NO_TRANSACTIONS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
