#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r12-input-calibration.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor, found {count}: {old[:160]!r}")
    s = s.replace(old, new, 1)


# r12 keeps r11's editor probes pass-through only. It observes the exact legacy
# mouse edge state that HandleMouseActions sees on Android, and separately fixes
# the repeated startup calibration only when Persistence reports its own exact
# inputOffsetNotSet sentinel. Existing calibrated values are never overwritten.

replace_once(
    "Method<int> g_editorGetScreenHeight;\n",
    "Method<int> g_editorGetScreenHeight;\n"
    "Method<bool> g_editorGetMouseButton;\n"
    "Method<bool> g_editorGetMouseButtonDown;\n"
    "Method<bool> g_editorGetMouseButtonUp;\n"
    "std::atomic<int> g_editorMouseHeldFrames{0};\n"
    "std::atomic<int> g_editorMouseDownEdges{0};\n"
    "std::atomic<int> g_editorMouseUpEdges{0};\n"
    "std::atomic<int> g_editorTouchAndDownFrames{0};\n"
    "std::atomic<int> g_editorSelectWhileMouseDown{0};\n"
    "std::atomic<int> g_editorSelectWhileTouch{0};\n"
    "std::atomic<int> g_editorLastMouseHeld{0};\n"
    "std::atomic<int> g_editorLastMouseDown{0};\n"
    "std::atomic<int> g_editorLastMouseUp{0};\n"
    "thread_local bool g_editorCurrentMouseDown = false;\n"
    "thread_local bool g_editorCurrentTouchActive = false;\n\n"
    "std::atomic<bool> g_calibrationAttempted{false};\n"
    "std::atomic<int> g_calibrationAbiGuard{0};\n"
    "std::atomic<int> g_calibrationMarkerReady{0};\n"
    "std::atomic<int> g_calibrationRecoveryState{0};\n"
    "std::atomic<int> g_calibrationUnsetDetected{0};\n"
    "std::atomic<int> g_calibrationNeutralized{0};\n"
    "std::atomic<int> g_calibrationSaveCalled{0};\n"
    "std::atomic<int> g_calibrationSentinelX100{0};\n"
    "std::atomic<int> g_calibrationBeforeX100{0};\n"
    "std::atomic<int> g_calibrationAfterX100{0};\n"
    "std::string g_calibrationMarker;\n"
)

old_snapshot = '''void SnapshotEditorInput() {
    if (!g_editorGetTouchCount.IsValid()) return;
    const int touchCount = g_editorGetTouchCount.Call();
    if (touchCount <= 0) return;
    g_editorTouchFrames.fetch_add(1);
    g_editorLastTouchCount.store(touchCount);
    if (g_editorGetMousePosition.IsValid()) {
        const Vector3 pos = g_editorGetMousePosition.Call();
        g_editorLastMouseX100.store(static_cast<int>(pos.x * 100.0f));
        g_editorLastMouseY100.store(static_cast<int>(pos.y * 100.0f));
    }
    if (g_editorGetScreenWidth.IsValid()) g_editorScreenWidth.store(g_editorGetScreenWidth.Call());
    if (g_editorGetScreenHeight.IsValid()) g_editorScreenHeight.store(g_editorGetScreenHeight.Call());
}
'''
new_snapshot = '''void SnapshotEditorInput() {
    const bool held = g_editorGetMouseButton.IsValid() && g_editorGetMouseButton.Call(0);
    const bool down = g_editorGetMouseButtonDown.IsValid() && g_editorGetMouseButtonDown.Call(0);
    const bool up = g_editorGetMouseButtonUp.IsValid() && g_editorGetMouseButtonUp.Call(0);
    const int touchCount = g_editorGetTouchCount.IsValid() ? g_editorGetTouchCount.Call() : 0;

    g_editorLastMouseHeld.store(held ? 1 : 0);
    g_editorLastMouseDown.store(down ? 1 : 0);
    g_editorLastMouseUp.store(up ? 1 : 0);
    g_editorLastTouchCount.store(touchCount);
    g_editorCurrentMouseDown = down;
    g_editorCurrentTouchActive = touchCount > 0;

    if (held) g_editorMouseHeldFrames.fetch_add(1);
    if (down) g_editorMouseDownEdges.fetch_add(1);
    if (up) g_editorMouseUpEdges.fetch_add(1);
    if (touchCount > 0) {
        g_editorTouchFrames.fetch_add(1);
        if (down) g_editorTouchAndDownFrames.fetch_add(1);
        if (g_editorGetMousePosition.IsValid()) {
            const Vector3 pos = g_editorGetMousePosition.Call();
            g_editorLastMouseX100.store(static_cast<int>(pos.x * 100.0f));
            g_editorLastMouseY100.store(static_cast<int>(pos.y * 100.0f));
        }
        if (g_editorGetScreenWidth.IsValid()) g_editorScreenWidth.store(g_editorGetScreenWidth.Call());
        if (g_editorGetScreenHeight.IsValid()) g_editorScreenHeight.store(g_editorGetScreenHeight.Call());
    }
}
'''
replace_once(old_snapshot, new_snapshot)

replace_once(
    "    SnapshotEditorInput();\n    g_oldEditorMouse(self, methodInfo);\n",
    "    SnapshotEditorInput();\n"
    "    g_oldEditorMouse(self, methodInfo);\n"
    "    g_editorCurrentMouseDown = false;\n"
    "    g_editorCurrentTouchActive = false;\n"
)

replace_once(
    "    g_editorSelectFloorCalls.fetch_add(1);\n"
    "    g_editorLastFloorNonNull.store(floor != nullptr ? 1 : 0);\n",
    "    g_editorSelectFloorCalls.fetch_add(1);\n"
    "    if (g_editorCurrentMouseDown) g_editorSelectWhileMouseDown.fetch_add(1);\n"
    "    if (g_editorCurrentTouchActive) g_editorSelectWhileTouch.fetch_add(1);\n"
    "    g_editorLastFloorNonNull.store(floor != nullptr ? 1 : 0);\n"
)

replace_once(
    '    g_editorGetScreenHeight = screen.GetMethod("get_height", 0);\n'
    '    if (!g_editorGetMousePosition.IsValid() || !g_editorGetTouchCount.IsValid() ||\n'
    '        !g_editorGetScreenWidth.IsValid() || !g_editorGetScreenHeight.IsValid()) return false;\n',
    '    g_editorGetScreenHeight = screen.GetMethod("get_height", 0);\n'
    '    g_editorGetMouseButton = input.GetMethod("GetMouseButton", 1);\n'
    '    g_editorGetMouseButtonDown = input.GetMethod("GetMouseButtonDown", 1);\n'
    '    g_editorGetMouseButtonUp = input.GetMethod("GetMouseButtonUp", 1);\n'
    '    if (!g_editorGetMousePosition.IsValid() || !g_editorGetTouchCount.IsValid() ||\n'
    '        !g_editorGetScreenWidth.IsValid() || !g_editorGetScreenHeight.IsValid() ||\n'
    '        !g_editorGetMouseButton.IsValid() || !g_editorGetMouseButtonDown.IsValid() ||\n'
    '        !g_editorGetMouseButtonUp.IsValid()) return false;\n'
)

calibration_code = r'''
bool PrepareCalibrationFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_calibrationRecoveryState.store(3); return false; }
    g_calibrationMarker = dir + "/calibration-r12-write.pending";
    g_calibrationMarkerReady.store(1);
    if (MarkerExists(g_calibrationMarker)) { g_calibrationRecoveryState.store(1); return false; }
    const std::string probe = dir + "/calibration-r12-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_calibrationMarkerReady.store(0);
        g_calibrationRecoveryState.store(3);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void MaybeNeutralizeUnsetCalibration() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_calibrationAttempted.load(std::memory_order_acquire) ||
        g_calibrationRecoveryState.load() != 0) return;

    Class persistence("", "Persistence");
    Class floatClass = Defaults::Get<float>();
    Class playerPrefs("UnityEngine", "PlayerPrefs");
    if (!persistence || !floatClass || !playerPrefs) return;

    FieldBase sentinelBase = persistence.GetField("inputOffsetNotSet");
    MethodBase getterBase = persistence.GetMethod("get_inputOffset", 0);
    MethodBase setterBase = persistence.GetMethod("set_inputOffset", 1);
    MethodBase saveBase = playerPrefs.GetMethod("Save", 0);
    IL2CPP::MethodInfo* getterInfo = getterBase.IsValid() ? getterBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* setterInfo = setterBase.IsValid() ? setterBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* saveInfo = saveBase.IsValid() ? saveBase.GetInfo() : nullptr;
    const bool setterParam = setterInfo != nullptr && setterInfo->parameters_count == 1 &&
            setterInfo->parameters != nullptr && setterInfo->parameters[0] != nullptr;
    const bool abi = sentinelBase.IsValid() && sentinelBase._isStatic && sentinelBase._isConst &&
            SameClass(sentinelBase.GetType(), floatClass) &&
            getterInfo != nullptr && getterInfo->methodPointer != nullptr && getterBase._isStatic &&
            getterInfo->parameters_count == 0 && SameClass(Class(getterInfo->return_type), floatClass) &&
            setterInfo != nullptr && setterInfo->methodPointer != nullptr && setterBase._isStatic &&
            setterParam && SameClass(Class(setterInfo->parameters[0]), floatClass) &&
            saveInfo != nullptr && saveInfo->methodPointer != nullptr && saveBase._isStatic &&
            saveInfo->parameters_count == 0;
    g_calibrationAbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareCalibrationFuse()) return;

    g_calibrationAttempted.store(true, std::memory_order_release);
    if (!WriteMarker(g_calibrationMarker)) {
        g_calibrationMarkerReady.store(0);
        g_calibrationRecoveryState.store(3);
        return;
    }

    Field<float> sentinelField(sentinelBase);
    Method<float> getter(getterBase);
    Method<void> setter(setterBase);
    Method<void> save(saveBase);
    const float sentinel = sentinelField.Get();
    const float before = getter.Call();
    g_calibrationSentinelX100.store(static_cast<int>(sentinel * 100.0f));
    g_calibrationBeforeX100.store(static_cast<int>(before * 100.0f));
    if (before == sentinel) {
        g_calibrationUnsetDetected.store(1);
        setter.Call(0.0f);
        save.Call();
        g_calibrationSaveCalled.store(1);
        const float after = getter.Call();
        g_calibrationAfterX100.store(static_cast<int>(after * 100.0f));
        if (after != sentinel) g_calibrationNeutralized.store(1);
    } else {
        g_calibrationAfterX100.store(static_cast<int>(before * 100.0f));
    }
    ClearMarker(g_calibrationMarker);
}

'''
replace_once("void MaybeInstallEditorProbe() {\n", calibration_code + "void MaybeInstallEditorProbe() {\n")

replace_once(
    "void ReconcileInstallState() {\n    MaybeInstallEditorProbe();\n",
    "void ReconcileInstallState() {\n"
    "    MaybeNeutralizeUnsetCalibration();\n"
    "    MaybeInstallEditorProbe();\n"
)

replace_once(
    "nativeProbe=cache-post-bnm-scneditor-input-observe-v1\\n",
    "nativeProbe=cache-post-bnm-scneditor-input-edge-calibration-v1\\n"
)
replace_once(
    "nativeStage=post-bnm-scneditor-handlemouse-selectfloor\\n",
    "nativeStage=post-bnm-scneditor-input-edge-and-calibration\\n"
)
replace_once("abiProbeRevision=11\\n", "abiProbeRevision=12\\n")

replace_once(
    '        << "editorLastCameraJump=" << g_editorLastCameraJump.load() << \'\\n\'\n',
    '        << "editorLastCameraJump=" << g_editorLastCameraJump.load() << \'\\n\'\n'
    '        << "editorMouseHeldFrames=" << g_editorMouseHeldFrames.load() << \'\\n\'\n'
    '        << "editorMouseDownEdges=" << g_editorMouseDownEdges.load() << \'\\n\'\n'
    '        << "editorMouseUpEdges=" << g_editorMouseUpEdges.load() << \'\\n\'\n'
    '        << "editorTouchAndDownFrames=" << g_editorTouchAndDownFrames.load() << \'\\n\'\n'
    '        << "editorSelectWhileMouseDown=" << g_editorSelectWhileMouseDown.load() << \'\\n\'\n'
    '        << "editorSelectWhileTouch=" << g_editorSelectWhileTouch.load() << \'\\n\'\n'
    '        << "editorLastMouseHeld=" << g_editorLastMouseHeld.load() << \'\\n\'\n'
    '        << "editorLastMouseDown=" << g_editorLastMouseDown.load() << \'\\n\'\n'
    '        << "editorLastMouseUp=" << g_editorLastMouseUp.load() << \'\\n\'\n'
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n'
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel" << \'\\n\'\n'
    '        << "calibrationAbiGuard=" << g_calibrationAbiGuard.load() << \'\\n\'\n'
    '        << "calibrationMarkerReady=" << g_calibrationMarkerReady.load() << \'\\n\'\n'
    '        << "calibrationRecoveryState=" << g_calibrationRecoveryState.load() << \'\\n\'\n'
    '        << "calibrationAttempted=" << (g_calibrationAttempted.load() ? 1 : 0) << \'\\n\'\n'
    '        << "calibrationUnsetDetected=" << g_calibrationUnsetDetected.load() << \'\\n\'\n'
    '        << "calibrationNeutralized=" << g_calibrationNeutralized.load() << \'\\n\'\n'
    '        << "calibrationSaveCalled=" << g_calibrationSaveCalled.load() << \'\\n\'\n'
    '        << "calibrationSentinelX100=" << g_calibrationSentinelX100.load() << \'\\n\'\n'
    '        << "calibrationBeforeX100=" << g_calibrationBeforeX100.load() << \'\\n\'\n'
    '        << "calibrationAfterX100=" << g_calibrationAfterX100.load() << \'\\n\'\n'
)

for marker in (
    "abiProbeRevision=12",
    "nativeProbe=cache-post-bnm-scneditor-input-edge-calibration-v1",
    "editorInputEdgePolicy=observe-only",
    "GetMouseButtonDown",
    "editorMouseDownEdges=",
    "editorTouchAndDownFrames=",
    "editorSelectWhileMouseDown=",
    "calibrationPolicy=neutralize-only-exact-game-sentinel",
    'GetField("inputOffsetNotSet")',
    'GetMethod("get_inputOffset", 0)',
    'GetMethod("set_inputOffset", 1)',
    'Class playerPrefs("UnityEngine", "PlayerPrefs")',
    'playerPrefs.GetMethod("Save", 0)',
    "calibration-r12-write.pending",
    "MaybeNeutralizeUnsetCalibration();",
):
    if marker not in s:
        raise SystemExit(f"r12 marker missing after overlay: {marker}")

path.write_text(s, encoding="utf-8")
