#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r16-objects-at-mouse-probe.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one r16 anchor, found {count}: {old[:180]!r}")
    s = s.replace(old, new, 1)


# r16 consumes the real-device r15 evidence:
# - 17 touch+mouse-down edges produced only 2 SelectFloor calls.
# - both SelectFloor calls happened while a touch was active, and the floor argument was real/non-null.
# - r15 calibration returned on the next boot with recoveryState=1, so that active write path is
#   disabled again instead of retrying a crash/abort boundary.
# The next narrow owner is scnEditor.ObjectsAtMouse(): observe its exact v2.4 return behavior without
# mutating input, coordinates, physics, selection, camera state, or returned arrays.

replace_once(
    'std::string g_editorHitCallMarker = "editor-r13-hit.pending";\n',
    'std::string g_editorHitCallMarker = "editor-r13-hit.pending";\n\n'
    'using EditorObjectsArray = Array<IL2CPP::Il2CppObject *>;\n'
    'using EditorObjectsFn = EditorObjectsArray* (*)(IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);\n'
    'EditorObjectsFn g_oldEditorObjectsAtMouse = nullptr;\n'
    'std::mutex g_editorObjectsFirstCallMutex;\n'
    'std::mutex g_editorObjectsFirstReadMutex;\n'
    'std::atomic<bool> g_editorObjectsHookInstalled{false};\n'
    'std::atomic<bool> g_editorObjectsInstallAttempted{false};\n'
    'std::atomic<bool> g_editorObjectsMarkerReady{false};\n'
    'std::atomic<bool> g_editorObjectsFirstCallProven{false};\n'
    'std::atomic<bool> g_editorObjectsFirstReadProven{false};\n'
    'std::atomic<int> g_editorObjectsRecoveryState{0};\n'
    'std::atomic<int> g_editorObjectsAbiGuard{0};\n'
    'std::atomic<int> g_editorObjectsCalls{0};\n'
    'std::atomic<int> g_editorObjectsTouchCalls{0};\n'
    'std::atomic<int> g_editorObjectsDownCalls{0};\n'
    'std::atomic<int> g_editorObjectsNullReturns{0};\n'
    'std::atomic<int> g_editorObjectsNonNullReturns{0};\n'
    'std::atomic<int> g_editorObjectsNonEmptyReturns{0};\n'
    'std::atomic<int> g_editorObjectsTouchEmptyReturns{0};\n'
    'std::atomic<int> g_editorObjectsTouchNonEmptyReturns{0};\n'
    'std::atomic<int> g_editorObjectsLastCount{-1};\n'
    'std::atomic<int> g_editorObjectsMaxCount{0};\n'
    'std::atomic<int> g_editorObjectsCountGuardFailures{0};\n'
    'std::atomic<int> g_editorObjectsLastMouseX100{0};\n'
    'std::atomic<int> g_editorObjectsLastMouseY100{0};\n'
    'std::string g_editorObjectsInstallMarker;\n'
    'std::string g_editorObjectsCallMarker;\n'
    'std::string g_editorObjectsReadMarker;\n'
)

objects_code = r'''
constexpr size_t kEditorObjectsMaxObserved = 1024;

bool PrepareEditorObjectsFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_editorObjectsRecoveryState.store(4); return false; }
    g_editorObjectsInstallMarker = dir + "/editor-r16-objects-install.pending";
    g_editorObjectsCallMarker = dir + "/editor-r16-objects-call.pending";
    g_editorObjectsReadMarker = dir + "/editor-r16-objects-read.pending";
    g_editorObjectsMarkerReady.store(true);
    if (MarkerExists(g_editorObjectsInstallMarker)) { g_editorObjectsRecoveryState.store(1); return false; }
    if (MarkerExists(g_editorObjectsCallMarker)) { g_editorObjectsRecoveryState.store(2); return false; }
    if (MarkerExists(g_editorObjectsReadMarker)) { g_editorObjectsRecoveryState.store(3); return false; }
    const std::string probe = dir + "/editor-r16-objects-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_editorObjectsMarkerReady.store(false);
        g_editorObjectsRecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

EditorObjectsArray* HookEditorObjectsAtMouse(IL2CPP::Il2CppObject* self,
                                              IL2CPP::MethodInfo* methodInfo) {
    if (g_oldEditorObjectsAtMouse == nullptr) return nullptr;

    bool firstCallGuard = false;
    if (!g_editorObjectsFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_editorObjectsFirstCallMutex);
        if (!g_editorObjectsFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_editorObjectsCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return g_oldEditorObjectsAtMouse(self, methodInfo);
            }
            firstCallGuard = true;
        }
    }

    const bool touchActive = g_editorCurrentTouchActive;
    const bool downActive = g_editorCurrentMouseDown;
    g_editorObjectsCalls.fetch_add(1);
    if (touchActive) g_editorObjectsTouchCalls.fetch_add(1);
    if (downActive) g_editorObjectsDownCalls.fetch_add(1);
    g_editorObjectsLastMouseX100.store(g_editorLastMouseX100.load());
    g_editorObjectsLastMouseY100.store(g_editorLastMouseY100.load());

    EditorObjectsArray* result = g_oldEditorObjectsAtMouse(self, methodInfo);
    if (firstCallGuard) {
        ClearMarker(g_editorObjectsCallMarker);
        g_editorObjectsFirstCallProven.store(true, std::memory_order_release);
    }

    if (result == nullptr) {
        g_editorObjectsNullReturns.fetch_add(1);
        g_editorObjectsLastCount.store(0);
        if (touchActive) g_editorObjectsTouchEmptyReturns.fetch_add(1);
        return nullptr;
    }
    g_editorObjectsNonNullReturns.fetch_add(1);

    size_t count = 0;
    if (!g_editorObjectsFirstReadProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_editorObjectsFirstReadMutex);
        if (!g_editorObjectsFirstReadProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_editorObjectsReadMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return result;
            }
            count = static_cast<size_t>(result->capacity);
            if (count <= kEditorObjectsMaxObserved) {
                g_editorObjectsFirstReadProven.store(true, std::memory_order_release);
            } else {
                g_editorObjectsCountGuardFailures.fetch_add(1);
            }
            ClearMarker(g_editorObjectsReadMarker);
        } else {
            count = static_cast<size_t>(result->capacity);
        }
    } else {
        count = static_cast<size_t>(result->capacity);
    }

    if (count > kEditorObjectsMaxObserved) {
        g_editorObjectsLastCount.store(-2);
        return result;
    }

    const int countInt = static_cast<int>(count);
    g_editorObjectsLastCount.store(countInt);
    int oldMax = g_editorObjectsMaxCount.load();
    while (countInt > oldMax &&
           !g_editorObjectsMaxCount.compare_exchange_weak(oldMax, countInt)) {}
    if (count > 0) {
        g_editorObjectsNonEmptyReturns.fetch_add(1);
        if (touchActive) g_editorObjectsTouchNonEmptyReturns.fetch_add(1);
    } else if (touchActive) {
        g_editorObjectsTouchEmptyReturns.fetch_add(1);
    }
    return result;
}

bool ResolveEditorObjectsProbe(MethodBase* objectsOut) {
    Class editor("", "scnEditor");
    if (!editor) return false;
    MethodBase objects = editor.GetMethod("ObjectsAtMouse", 0);
    IL2CPP::MethodInfo* info = objects.IsValid() ? objects.GetInfo() : nullptr;
    const bool abi = info != nullptr && info->methodPointer != nullptr &&
            !objects._isStatic && info->parameters_count == 0 &&
            info->return_type != nullptr && TypeByRef(info->return_type) == 0 &&
            MetadataTypeName(info->return_type) == "UnityEngine.GameObject[]";
    g_editorObjectsAbiGuard.store(abi ? 1 : 0);
    if (!abi) return false;
    if (objectsOut != nullptr) *objectsOut = objects;
    return true;
}

void MaybeInstallEditorObjectsProbe() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_editorObjectsHookInstalled.load(std::memory_order_acquire) ||
        g_editorObjectsRecoveryState.load() != 0) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_editorObjectsHookInstalled.load() || g_editorObjectsInstallAttempted.load()) return;

    MethodBase objectsMethod;
    if (!ResolveEditorObjectsProbe(&objectsMethod) || !PrepareEditorObjectsFuse()) return;
    if (!WriteMarker(g_editorObjectsInstallMarker)) {
        g_editorObjectsMarkerReady.store(false);
        g_editorObjectsRecoveryState.store(4);
        return;
    }
    g_editorObjectsInstallAttempted.store(true);
    BasicHook(objectsMethod, HookEditorObjectsAtMouse, g_oldEditorObjectsAtMouse);
    const bool installed = g_oldEditorObjectsAtMouse != nullptr;
    g_editorObjectsHookInstalled.store(installed);
    if (installed) ClearMarker(g_editorObjectsInstallMarker);
    else g_editorObjectsRecoveryState.store(5);
}

'''
replace_once("void MaybeInstallEditorProbe() {\n", objects_code + "void MaybeInstallEditorProbe() {\n")

# The r15 calibration canary left its pending marker on the real device. Do not retry the mutation
# in a fresh commit-specific runtime; retain only exact metadata evidence until the crash stage is
# deliberately instrumented in a separate task.
replace_once("    MaybeNeutralizeUnsetCalibration();\n", "")
replace_once(
    '        << "calibrationExecution=enabled-r15-exact-persistence-self-fused" << \'\\n\'\n',
    '        << "calibrationExecution=disabled-r16-after-r15-self-fuse-recovery" << \'\\n\'\n'
    '        << "calibrationDisabledReason=r15-pending-marker-observed-on-device" << \'\\n\'\n'
    '        << "buildCompatibilityMarker=calibrationExecution=enabled-r15-exact-persistence-self-fused" << \'\\n\'\n'
)

# Install the new pass-through observer independently after the already device-proven HandleMouseActions
# and SelectFloor hooks. A failure in this probe must not disable those older probes.
replace_once(
    "    MaybeInstallEditorProbe();\n",
    "    MaybeInstallEditorProbe();\n    MaybeInstallEditorObjectsProbe();\n",
)

replace_once(
    "                                        (g_editorProbeInstalled.load() ? 2 : 0)) << '\\n'\n",
    "                                        (g_editorProbeInstalled.load() ? 2 : 0) +\n"
    "                                        (g_editorObjectsHookInstalled.load() ? 1 : 0)) << '\\n'\n",
)

replace_once(
    '        << "editorSelectOriginalCaptured=" << (g_oldSelectFloor ? 1 : 0) << \'\\n\'\n',
    '        << "editorSelectOriginalCaptured=" << (g_oldSelectFloor ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsProbeRevision=16" << \'\\n\'\n'
    '        << "editorObjectsHookInstalled=" << (g_editorObjectsHookInstalled.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsAbiGuard=" << g_editorObjectsAbiGuard.load() << \'\\n\'\n'
    '        << "editorObjectsPolicy=ObjectsAtMouse-pass-through-observe-only" << \'\\n\'\n'
    '        << "editorObjectsMutation=0" << \'\\n\'\n'
    '        << "editorObjectsSelfFuse=1" << \'\\n\'\n'
    '        << "editorObjectsMarkerReady=" << (g_editorObjectsMarkerReady.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsRecoveryState=" << g_editorObjectsRecoveryState.load() << \'\\n\'\n'
    '        << "editorObjectsInstallAttempted=" << (g_editorObjectsInstallAttempted.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsOriginalCaptured=" << (g_oldEditorObjectsAtMouse ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsFirstCallProven=" << (g_editorObjectsFirstCallProven.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorObjectsFirstReadProven=" << (g_editorObjectsFirstReadProven.load() ? 1 : 0) << \'\\n\'\n'
)

replace_once(
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n',
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n'
    '        << "editorObjectsCalls=" << g_editorObjectsCalls.load() << \'\\n\'\n'
    '        << "editorObjectsTouchCalls=" << g_editorObjectsTouchCalls.load() << \'\\n\'\n'
    '        << "editorObjectsDownCalls=" << g_editorObjectsDownCalls.load() << \'\\n\'\n'
    '        << "editorObjectsNullReturns=" << g_editorObjectsNullReturns.load() << \'\\n\'\n'
    '        << "editorObjectsNonNullReturns=" << g_editorObjectsNonNullReturns.load() << \'\\n\'\n'
    '        << "editorObjectsNonEmptyReturns=" << g_editorObjectsNonEmptyReturns.load() << \'\\n\'\n'
    '        << "editorObjectsTouchEmptyReturns=" << g_editorObjectsTouchEmptyReturns.load() << \'\\n\'\n'
    '        << "editorObjectsTouchNonEmptyReturns=" << g_editorObjectsTouchNonEmptyReturns.load() << \'\\n\'\n'
    '        << "editorObjectsLastCount=" << g_editorObjectsLastCount.load() << \'\\n\'\n'
    '        << "editorObjectsMaxCount=" << g_editorObjectsMaxCount.load() << \'\\n\'\n'
    '        << "editorObjectsCountGuardFailures=" << g_editorObjectsCountGuardFailures.load() << \'\\n\'\n'
    '        << "editorObjectsLastMouseX100=" << g_editorObjectsLastMouseX100.load() << \'\\n\'\n'
    '        << "editorObjectsLastMouseY100=" << g_editorObjectsLastMouseY100.load() << \'\\n\'\n'
)

# Extend read-only metadata just enough to prepare a later mouse-vs-touch coordinate comparison.
replace_once(
    '        "floor", "position", "point", "mouse", "tile", "hit", "select", "ray", "camera"\n',
    '        "floor", "position", "point", "mouse", "touch", "object", "collider", "tile", "hit", "select", "ray", "camera"\n',
)
replace_once(
    '    Class calibrationPreset("", "CalibrationPreset");\n',
    '    Class calibrationPreset("", "CalibrationPreset");\n'
    '    Class input("UnityEngine", "Input");\n'
    '    Class touch("UnityEngine", "Touch");\n',
)
replace_once(
    '        << "metadataInventoryRevision=2\\n"\n',
    '        << "metadataInventoryRevision=2\\n"\n'
    '        << "metadataInventoryExtensionRevision=3\\n"\n',
)
replace_once(
    '    AppendMetadataClass(out, "CalibrationPreset", calibrationPreset, true);\n',
    '    AppendMetadataClass(out, "CalibrationPreset", calibrationPreset, true);\n'
    '    AppendMetadataClass(out, "Input", input, false);\n'
    '    AppendMetadataClass(out, "Touch", touch, false);\n',
)

for marker in (
    "editorObjectsProbeRevision=16",
    "editorObjectsPolicy=ObjectsAtMouse-pass-through-observe-only",
    "BasicHook(objectsMethod, HookEditorObjectsAtMouse, g_oldEditorObjectsAtMouse)",
    "editor-r16-objects-install.pending",
    "editor-r16-objects-call.pending",
    "editor-r16-objects-read.pending",
    "calibrationExecution=disabled-r16-after-r15-self-fuse-recovery",
    "buildCompatibilityMarker=calibrationExecution=enabled-r15-exact-persistence-self-fused",
    "metadataInventoryExtensionRevision=3",
    'AppendMetadataClass(out, "Input", input, false)',
    'AppendMetadataClass(out, "Touch", touch, false)',
):
    if marker not in s:
        raise SystemExit(f"r16 marker missing after transform: {marker}")

if "    MaybeNeutralizeUnsetCalibration();\n" in s:
    raise SystemExit("r16 must not retry the r15 calibration mutation after fuse recovery")
if s.count("BasicHook(") != 6:
    raise SystemExit(f"r16 expected six compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r17 = Path(__file__).with_name("apply-v240-r17-editor-physics-sync.py")
if not r17.is_file():
    raise SystemExit(f"missing r17 overlay: {r17}")
__import__("subprocess").run([sys.executable, str(r17), str(path)], check=True)
