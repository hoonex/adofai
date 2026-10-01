from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
R10 = ROOT / "scripts/apply-v240-r10-raycast-probe.py"
R11 = ROOT / "scripts/apply-v240-r11-editor-probe.py"
R12 = ROOT / "scripts/apply-v240-r12-input-calibration.py"
R13 = ROOT / "scripts/apply-v240-r13-hit-probe-calibration-fix.py"
R15 = ROOT / "scripts/apply-v240-r15-exact-calibration-and-wide-inventory.py"
R30 = ROOT / "scripts/apply-v240-r30-tile-result-observation.py"
R31 = ROOT / "scripts/apply-v240-r31-editor-transaction-trace.py"
BUILD = ROOT / "scripts/build-v240-cache-native.sh"
WORKFLOW = ROOT / ".github/workflows/v240-runtime-channel.yml"


def generate_final_source() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "V240CacheLoader.cpp"
        source = BASE.read_text(encoding="utf-8")
        replacements = (
            ("jmethodID g_dynamicDiagnostics = nullptr;",
             "jmethodID g_dynamicDiagnosticsMethod = nullptr;"),
            ("g_dynamicAwait != nullptr && g_dynamicDiagnostics != nullptr;",
             "g_dynamicAwait != nullptr && g_dynamicDiagnosticsMethod != nullptr;"),
            ("diagnostics = g_dynamicDiagnostics;",
             "diagnostics = g_dynamicDiagnosticsMethod;"),
            ("g_dynamicDiagnostics = diagnostics;",
             "g_dynamicDiagnosticsMethod = diagnostics;"),
        )
        for old, new in replacements:
            if source.count(old) != 1:
                raise AssertionError(f"build-copy rename anchor changed: {old!r}")
            source = source.replace(old, new, 1)
        target.write_text(source, encoding="utf-8")

        for overlay in (R10, R11, R12, R13, R15):
            subprocess.run(
                ["python3", str(overlay), str(target)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        return target.read_text(encoding="utf-8")


class V240RuntimeR31ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r30 = R30.read_text(encoding="utf-8")
        cls.r31 = R31.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.generated = generate_final_source()

    def test_r30_chains_r31(self):
        self.assertIn("apply-v240-r31-editor-transaction-trace.py", self.r30)

    def test_final_source_has_exactly_seven_new_trace_hook_sites(self):
        self.assertEqual(self.generated.count("BasicHook("), 25)
        for marker in (
            "BasicHook(handle, HookEditorTraceR31Handle, g_oldEditorTraceR31Handle)",
            "BasicHook(select, HookEditorTraceR31Select, g_oldEditorTraceR31Select)",
            "BasicHook(smart, HookEditorTraceR31Smart, g_oldEditorTraceR31Smart)",
            "BasicHook(gizmoAtMouse, HookEditorTraceR31Gizmo, g_oldEditorTraceR31Gizmo)",
            "BasicHook(dragCamera, HookEditorTraceR31DragCamera, g_oldEditorTraceR31DragCamera)",
            "BasicHook(dragTilesStart, HookEditorTraceR31DragTilesStart",
            "BasicHook(dragTiles, HookEditorTraceR31DragTiles, g_oldEditorTraceR31DragTiles)",
        ):
            self.assertIn(marker, self.generated)

    def test_trace_hooks_are_exact_abi_guarded(self):
        for marker in (
            'GetMethod("HandleMouseActions", 0)',
            'GetMethod("SelectFloor", 2)',
            'GetMethod("SmartObjectSelect", 1)',
            'GetMethod("GizmoAtMouse", 0)',
            'GetMethod("DragCamera", 1)',
            'GetMethod("DragTilesStart", 0)',
            'GetMethod("DragTiles", 1)',
            "TypeByRef(selectInfo->parameters[0]) == 0",
            "SameClass(Class(selectInfo->parameters[0]), floor)",
            "SameClass(Class(selectInfo->parameters[1]), boolClass)",
            "SameClass(Class(smartInfo->return_type), gameObject)",
            "SameClass(Class(gizmoInfo->return_type), gizmo)",
            "EditorTraceR31Vector3VoidAbi",
            "EditorTraceR31StaticMouseButtonAbi",
            "editorTraceR31InputAbiGuard=",
        ):
            self.assertIn(marker, self.generated)

    def test_trace_is_bounded_and_transaction_correlated(self):
        for marker in (
            "kEditorTraceR31Slots = 8",
            "EditorTraceR31BeginTxn",
            "EditorTraceR31UpdatePointer",
            "EditorTraceR31FinishTxn",
            "EditorTraceR31RecordObjects(count);",
            "EditorTraceR31RecordRaycast(ordinal, count, origin, layerMask);",
            "editorTraceR31LatestSeq=",
            "editorTraceR31Txn",
            ",release:",
            ",maxd:",
            ",obj:",
            ",ray:",
            ",smart:",
            ",gizmo:",
            ",select:",
            ",drag:",
        ):
            self.assertIn(marker, self.generated)
        self.assertNotIn("LOGI(", self.r31)
        self.assertNotIn("LOGD(", self.r31)
        self.assertIn("g_editorTraceR31FinishOnNextHandle", self.generated)
        self.assertIn("finishReleasedAfterThisHandle", self.generated)
        self.assertIn("txn->releaseSeen.store(1", self.generated)
        self.assertIn("txn->postReleaseFrames.fetch_add(1", self.generated)
        self.assertNotIn("if (up) EditorTraceR31FinishTxn();", self.generated)

    def test_all_r31_behavior_hooks_are_pass_through(self):
        for original_call in (
            "g_oldEditorTraceR31Handle(self, methodInfo);",
            "g_oldEditorTraceR31Select(self, floor, cameraJump, methodInfo);",
            "g_oldEditorTraceR31Smart(self, allowCycling, methodInfo)",
            "g_oldEditorTraceR31Gizmo(self, methodInfo)",
            "g_oldEditorTraceR31DragCamera(self, delta, methodInfo);",
            "g_oldEditorTraceR31DragTilesStart(self, methodInfo);",
            "g_oldEditorTraceR31DragTiles(self, delta, methodInfo);",
        ):
            self.assertIn(original_call, self.generated)
        for forbidden in (
            "g_editorTraceR31Select.Call",
            'GetField("pointerDownObjectType")',
            'GetField("mousePosition0")',
        ):
            self.assertNotIn(forbidden, self.r31)
        self.assertIn("editorTraceR31Mutation=0", self.generated)
        self.assertNotIn("Vector3::zero", self.generated)
        self.assertNotIn("Method<Vector3>(mousePosition)", self.generated)

    def test_r31_has_install_and_per_hook_first_call_fuses(self):
        for marker in (
            "editor-r31-trace-install.pending",
            "editor-r31-trace-call-",
            "PrepareEditorTraceR31Fuse",
            "EditorTraceR31BeginCanary",
            "EditorTraceR31EndCanary",
            "editorTraceR31CallProvenMask=",
            "editorTraceR31RecoveryState=",
        ):
            self.assertIn(marker, self.generated)

    def test_r31_does_not_reenable_retired_r11_probe_installers(self):
        reconcile = self.generated[
            self.generated.index("void ReconcileInstallState()"):
            self.generated.index("std::string CurrentReport")
        ]
        self.assertIn("MaybeInstallEditorTraceR31();", reconcile)
        self.assertNotIn("MaybeInstallEditorProbe();", reconcile)
        self.assertNotIn("MaybeInstallEditorObjectsProbe();", reconcile)
        self.assertNotIn("MaybeInstallEditorPhysicsSync();", reconcile)

    def test_production_hook_counter_semantics_stay_separate(self):
        start = self.generated.index('<< "gameHooksInstalled="')
        end = self.generated.index('<< "gameHooksInstalledSemantics=', start)
        counter = self.generated[start:end]
        self.assertNotIn("editorTraceR31", counter)
        self.assertIn(
            "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2",
            self.generated,
        )
        self.assertIn("editorTraceR31InstalledMask=", self.generated)

    def test_build_and_channel_contract_require_r31(self):
        for marker in (
            'R31_OVERLAY="${ROOT}/scripts/apply-v240-r31-editor-transaction-trace.py"',
            'test -f "${R31_OVERLAY}"',
            "editorTraceRevision=31",
            "editorTraceR31Policy=bounded-read-only-pointer-transaction",
            "editorTraceR31Mutation=0",
            "editor-r31-trace-install.pending",
        ):
            self.assertIn(marker, self.build)
        self.assertIn(
            "'scripts/apply-v240-r31-editor-transaction-trace.py'",
            self.workflow,
        )
        self.assertIn("'tests/test_v240_runtime_r31.py'", self.workflow)


if __name__ == "__main__":
    unittest.main()
