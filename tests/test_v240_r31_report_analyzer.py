import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "tools/analyze_v240_r31_report.py"

spec = importlib.util.spec_from_file_location("v240_r31_analyzer", ANALYZER)
analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analyzer)


def runtime_lines(latest_seq: int = 1, **overrides) -> list[str]:
    values = {
        "editorTraceRevision": "31",
        "editorTraceR31Policy": "bounded-read-only-pointer-transaction",
        "editorTraceR31Mutation": "0",
        "editorTraceR31Slots": "8",
        "editorTraceR31AbiMask": "127",
        "editorTraceR31InputAbiGuard": "1",
        "editorTraceR31InstalledMask": "127",
        "editorTraceR31InstallAttempted": "1",
        "editorTraceR31MarkerReady": "1",
        "editorTraceR31RecoveryState": "0",
        "editorTraceR31CallProvenMask": "1",
        "editorTraceR31LatestSeq": str(latest_seq),
        "activeVersion": "407b744a2a85889a2182c2e1385a4e582eabea69",
        "runtimeReadiness": "loaded-health-confirmed",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    return [f"{key}={value}" for key, value in values.items()]


def txn_line(
    slot: int,
    seq: int,
    *,
    state: int = 2,
    start=(10000, 20000),
    end=(10020, 20010),
    screen=(1080, 2340),
    touch=1,
    hf=2,
    held=1,
    release=(1, 1),
    maxd=(20, 10),
    obj=(1, 0),
    ray=(2, 0, 0),
    r1=(1250, -3400, 256),
    r2=(1250, -3400, 512),
    smart=(0, 0),
    gizmo=(0, 0),
    select=(0, 0, 0),
    drag=(0, 0, 0),
) -> str:
    return (
        f"editorTraceR31Txn{slot}="
        f"seq:{seq},state:{state},"
        f"start:{start[0]}:{start[1]},end:{end[0]}:{end[1]},"
        f"screen:{screen[0]}x{screen[1]},touch:{touch},"
        f"hf:{hf},held:{held},release:{release[0]}:{release[1]},"
        f"maxd:{maxd[0]}:{maxd[1]},"
        f"obj:{obj[0]}:{obj[1]},ray:{ray[0]}:{ray[1]}:{ray[2]},"
        f"r1:{r1[0]}:{r1[1]}:{r1[2]},r2:{r2[0]}:{r2[1]}:{r2[2]},"
        f"smart:{smart[0]}:{smart[1]},gizmo:{gizmo[0]}:{gizmo[1]},"
        f"select:{select[0]}:{select[1]}:{select[2]},"
        f"drag:{drag[0]}:{drag[1]}:{drag[2]}"
    )


def report(*txns: str, latest_seq: int | None = None, metadata=None) -> str:
    if latest_seq is None:
        latest_seq = max(
            (
                int(item.split("seq:", 1)[1].split(",", 1)[0])
                for item in txns
                if "seq:" in item
            ),
            default=0,
        )
    lines = runtime_lines(latest_seq)
    if metadata:
        replacements = {str(key): str(value) for key, value in metadata.items()}
        parsed = dict(line.split("=", 1) for line in lines)
        parsed.update(replacements)
        lines = [f"{key}={value}" for key, value in parsed.items()]
    return "\n".join([*lines, *txns, ""])


class V240R31ReportAnalyzerTest(unittest.TestCase):
    def test_select_floor_reached_is_reported_without_root_cause_claim(self):
        result = analyzer.analyze_report(
            report(
                txn_line(
                    0,
                    1,
                    obj=(1, 2),
                    ray=(2, 2, 1),
                    select=(1, 1, 0),
                )
            )
        )
        self.assertEqual(result["status"], "ANALYZED")
        self.assertTrue(result["runtime"]["ready"])
        self.assertEqual(result["transactions"][0]["route"], "SELECT_FLOOR_REACHED")
        self.assertIn("after SelectFloor", result["transactions"][0]["boundary"])
        self.assertNotIn("root cause", result["transactions"][0]["boundary"].lower())

    def test_positive_hit_without_select_narrows_to_post_hit_arbitration(self):
        result = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(1, 3), ray=(2, 3, 0)))
        )
        txn = result["transactions"][0]
        self.assertEqual(txn["route"], "WORLD_HIT_WITHOUT_SELECT")
        self.assertTrue(txn["positive_world_hit"])
        self.assertIn("post-hit", txn["boundary"])

    def test_zero_hit_query_is_distinct_from_pre_world_gate(self):
        zero_hit = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(1, 0), ray=(2, 0, 0)))
        )["transactions"][0]
        pre_world = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(0, -3), ray=(0, -3, -3)))
        )["transactions"][0]
        self.assertEqual(zero_hit["route"], "WORLD_QUERY_NO_HIT")
        self.assertEqual(pre_world["route"], "PRE_WORLD_QUERY_PATH")

    def test_explicit_camera_and_tile_drag_paths_are_not_conflated(self):
        camera = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(0, -3), ray=(0, -3, -3), drag=(1, 0, 0)))
        )["transactions"][0]
        tiles = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(0, -3), ray=(0, -3, -3), drag=(0, 1, 2)))
        )["transactions"][0]
        self.assertEqual(camera["route"], "CAMERA_DRAG_REACHED")
        self.assertEqual(tiles["route"], "TILE_DRAG_REACHED")

    def test_mixed_explicit_paths_fail_open_to_ambiguous_route(self):
        txn = analyzer.analyze_report(
            report(
                txn_line(
                    0,
                    1,
                    select=(1, 1, 0),
                    drag=(1, 0, 0),
                )
            )
        )["transactions"][0]
        self.assertEqual(txn["route"], "MIXED_EXPLICIT_PATH")
        self.assertIn("More than one explicit", txn["boundary"])

    def test_active_transaction_is_partial_not_complete_evidence(self):
        result = analyzer.analyze_report(report(txn_line(0, 1, state=1, release=(0, 0))))
        self.assertEqual(result["status"], "PARTIAL_TRANSACTIONS")
        self.assertEqual(result["transactions"][0]["route"], "ACTIVE_TRANSACTION")

    def test_superseded_transaction_is_partial_and_not_localized(self):
        result = analyzer.analyze_report(
            report(
                txn_line(
                    0,
                    1,
                    state=3,
                    obj=(1, 2),
                    ray=(2, 2, 1),
                    select=(1, 1, 0),
                )
            )
        )
        self.assertEqual(result["status"], "PARTIAL_TRANSACTIONS")
        self.assertEqual(result["transactions"][0]["route"], "SUPERSEDED_TRANSACTION")
        self.assertIn("do not use it alone", result["transactions"][0]["boundary"])

    def test_query_sentinel_error_stays_ambiguous(self):
        txn = analyzer.analyze_report(
            report(txn_line(0, 1, obj=(1, -1), ray=(2, 0, -3)))
        )["transactions"][0]
        self.assertEqual(txn["route"], "WORLD_QUERY_AMBIGUOUS")

    def test_ring_slots_are_sorted_by_sequence_not_slot(self):
        result = analyzer.analyze_report(
            report(
                txn_line(7, 8, obj=(1, 0), ray=(2, 0, 0)),
                txn_line(0, 9, obj=(1, 2), ray=(2, 2, 1)),
                txn_line(1, 10, obj=(1, 1), ray=(2, 1, 0)),
                latest_seq=10,
            )
        )
        self.assertEqual([txn["seq"] for txn in result["transactions"]], [8, 9, 10])

    def test_malformed_transaction_warns_instead_of_crashing(self):
        text = report(latest_seq=1) + "editorTraceR31Txn0=seq:1,state:oops\n"
        result = analyzer.analyze_report(text)
        self.assertEqual(result["status"], "NO_TRANSACTIONS")
        self.assertTrue(result["warnings"])
        self.assertIn("expected integer", result["warnings"][0])

    def test_runtime_gate_blocks_analysis_claim_when_hooks_not_fully_installed(self):
        result = analyzer.analyze_report(
            report(
                txn_line(0, 1),
                metadata={"editorTraceR31InstalledMask": 3},
            )
        )
        self.assertEqual(result["status"], "RUNTIME_NOT_READY")
        self.assertFalse(result["runtime"]["ready"])
        self.assertTrue(
            any("InstalledMask" in problem for problem in result["runtime"]["problems"])
        )

    def test_missing_r31_report_is_runtime_not_ready(self):
        result = analyzer.analyze_report("runtimeReadiness=loaded-health-confirmed\n")
        self.assertEqual(result["status"], "RUNTIME_NOT_READY")
        self.assertFalse(result["runtime"]["ready"])
        self.assertEqual(result["transaction_count"], 0)

    def test_cli_json_is_machine_readable(self):
        completed = subprocess.run(
            [sys.executable, str(ANALYZER), "--json"],
            input=report(txn_line(0, 1, select=(1, 1, 0))),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ANALYZED")
        self.assertEqual(payload["transactions"][0]["route"], "SELECT_FLOOR_REACHED")

    def test_cli_returns_nonzero_when_runtime_contract_is_missing(self):
        completed = subprocess.run(
            [sys.executable, str(ANALYZER)],
            input="runtimeReadiness=loaded-health-confirmed\n",
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("status=RUNTIME_NOT_READY", completed.stdout)


if __name__ == "__main__":
    unittest.main()
