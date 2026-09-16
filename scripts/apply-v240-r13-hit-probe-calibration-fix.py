#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r13-hit-probe-calibration-fix.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor, found {count}: {old[:180]!r}")
    s = s.replace(old, new, 1)


# r13 follows real-device r12 evidence:
# - Android touch -> legacy mouse down/up edges are present, so do not synthesize clicks.
# - selection is rare after the edge, so observe the screen->world->floor hit path read-only.
# - r12 calibration guard was impossible because BNM marks const fields _isConst=1 and
#   deliberately _isStatic=0. Accept the literal-const representation but still write only
#   when Persistence.get_inputOffset() exactly equals the game's own sentinel value.

replace_once(
    "thread_local bool g_editorCurrentTouchActive = false;\n\n",
    "thread_local bool g_editorCurrentTouchActive = false;\n"
    "thread_local IL2CPP::Il2CppObject* g_editorCurrentHitFloor = nullptr;\n\n"
    "Field<IL2CPP::Il2CppObject*> g_editorHitCameraField;\n"
    "Method<IL2CPP::Il2CppObject*> g_editorHitGetMainCamera;\n"
    "Method<Vector3> g_editorHitScreenToWorld;\n"
    "Method<IL2CPP::Il2CppObject*> g_editorHitGetFloorAtPosition;\n"
    "std::mutex g_editorHitFirstCallMutex;\n"
    "std::atomic<bool> g_editorHitFirstCallProven{false};\n"
    "std::atomic<int> g_editorHitAbiGuard{0};\n"
    "std::atomic<int> g_editorHitCameraFieldGuard{0};\n"
    "std::atomic<int> g_editorHitMainCameraGuard{0};\n"
    "std::atomic<int> g_editorHitScreenToWorldGuard{0};\n"
    "std::atomic<int> g_editorHitFloorMethodGuard{0};\n"
    "std::atomic<int> g_editorHitMarkerReady{0};\n"
    "std::atomic<int> g_editorHitRecoveryState{0};\n"
    "std::atomic<int> g_editorHitProbeCalls{0};\n"
    "std::atomic<int> g_editorHitProbeDownCalls{0};\n"
    "std::atomic<int> g_editorHitFloorFound{0};\n"
    "std::atomic<int> g_editorHitDownFloorFound{0};\n"
    "std::atomic<int> g_editorHitHeldFloorFound{0};\n"
    "std::atomic<int> g_editorHitCameraSource{0};\n"
    "std::atomic<int> g_editorHitLastWorldX100{0};\n"
    "std::atomic<int> g_editorHitLastWorldY100{0};\n"
    "std::atomic<int> g_editorHitLastFloorNonNull{0};\n"
    "std::atomic<int> g_editorSelectWhileHitFound{0};\n"
    "std::atomic<int> g_editorSelectMatchesHit{0};\n"
    "std::string g_editorHitCallMarker;\n\n"
)

replace_once(
    "std::atomic<int> g_calibrationAfterX100{0};\n",
    "std::atomic<int> g_calibrationAfterX100{0};\n"
    "std::atomic<int> g_calibrationSentinelValid{0};\n"
    "std::atomic<int> g_calibrationSentinelConst{0};\n"
    "std::atomic<int> g_calibrationSentinelStatic{0};\n"
    "std::atomic<int> g_calibrationSentinelTypeFloat{0};\n"
    "std::atomic<int> g_calibrationGetterResolved{0};\n"
    "std::atomic<int> g_calibrationSetterResolved{0};\n"
    "std::atomic<int> g_calibrationSaveResolved{0};\n"
)

hit_probe_code = r'''
bool PrepareEditorHitFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_editorHitRecoveryState.store(3); return false; }
    g_editorHitCallMarker = dir + "/editor-r13-hit.pending";
    g_editorHitMarkerReady.store(1);
    if (MarkerExists(g_editorHitCallMarker)) {
        g_editorHitRecoveryState.store(1);
        return false;
    }
    const std::string probe = dir + "/editor-r13-hit-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_editorHitMarkerReady.store(0);
        g_editorHitRecoveryState.store(3);
        return false;
    }
    ClearMarker(probe);
    return true;
}

bool ResolveEditorHitProbe() {
    Class editor("", "scnEditor");
    Class camera("UnityEngine", "Camera");
    Class rdUtils("", "RDUtils");
    Class floor("", "scrFloor");
    Class vector2 = Defaults::Get<Vector2>();
    Class vector3 = Defaults::Get<Vector3>();
    if (!editor || !camera || !rdUtils || !floor || !vector2 || !vector3) return false;

    FieldBase cameraField = editor.GetField("camera");
    const bool cameraFieldAbi = cameraField.IsValid() && !cameraField._isStatic &&
            !cameraField._isThreadStatic && !cameraField._isConst &&
            SameClass(cameraField.GetType(), camera);
    g_editorHitCameraFieldGuard.store(cameraFieldAbi ? 1 : 0);

    MethodBase mainBase = camera.GetMethod("get_main", 0);
    IL2CPP::MethodInfo* mainInfo = mainBase.IsValid() ? mainBase.GetInfo() : nullptr;
    const bool mainAbi = mainInfo != nullptr && mainInfo->methodPointer != nullptr &&
            mainBase._isStatic && mainInfo->parameters_count == 0 &&
            SameClass(Class(mainInfo->return_type), camera);
    g_editorHitMainCameraGuard.store(mainAbi ? 1 : 0);

    MethodBase screenBase = camera.GetMethod("ScreenToWorldPoint", 1);
    IL2CPP::MethodInfo* screenInfo = screenBase.IsValid() ? screenBase.GetInfo() : nullptr;
    const bool screenParam = screenInfo != nullptr && screenInfo->parameters_count == 1 &&
            screenInfo->parameters != nullptr && screenInfo->parameters[0] != nullptr;
    const bool screenAbi = screenInfo != nullptr && screenInfo->methodPointer != nullptr &&
            !screenBase._isStatic && screenParam &&
            SameClass(Class(screenInfo->parameters[0]), vector3) &&
            TypeByRef(screenInfo->parameters[0]) == 0 &&
            SameClass(Class(screenInfo->return_type), vector3);
    g_editorHitScreenToWorldGuard.store(screenAbi ? 1 : 0);

    MethodBase floorBase = rdUtils.GetMethod("GetFloorAtPosition", 1);
    IL2CPP::MethodInfo* floorInfo = floorBase.IsValid() ? floorBase.GetInfo() : nullptr;
    const bool floorParam = floorInfo != nullptr && floorInfo->parameters_count == 1 &&
            floorInfo->parameters != nullptr && floorInfo->parameters[0] != nullptr;
    const bool floorAbi = floorInfo != nullptr && floorInfo->methodPointer != nullptr &&
            floorBase._isStatic && floorParam &&
            SameClass(Class(floorInfo->parameters[0]), vector2) &&
            TypeByRef(floorInfo->parameters[0]) == 0 &&
            SameClass(Class(floorInfo->return_type), floor);
    g_editorHitFloorMethodGuard.store(floorAbi ? 1 : 0);

    const bool abi = (cameraFieldAbi || mainAbi) && screenAbi && floorAbi;
    g_editorHitAbiGuard.store(abi ? 1 : 0);
    if (!abi) return false;

    if (cameraFieldAbi) g_editorHitCameraField = Field<IL2CPP::Il2CppObject*>(cameraField);
    if (mainAbi) g_editorHitGetMainCamera = Method<IL2CPP::Il2CppObject*>(mainBase);
    g_editorHitScreenToWorld = Method<Vector3>(screenBase);
    g_editorHitGetFloorAtPosition = Method<IL2CPP::Il2CppObject*>(floorBase);
    return PrepareEditorHitFuse();
}

void ProbeEditorFloorHit(IL2CPP::Il2CppObject* editorSelf, bool down, int touchCount) {
    g_editorCurrentHitFloor = nullptr;
    if (editorSelf == nullptr || touchCount <= 0 ||
        g_editorHitAbiGuard.load(std::memory_order_acquire) == 0 ||
        g_editorHitRecoveryState.load(std::memory_order_acquire) != 0) return;

    bool firstCanary = false;
    if (!g_editorHitFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_editorHitFirstCallMutex);
        if (!g_editorHitFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_editorHitCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_editorHitRecoveryState.store(3);
                return;
            }
            firstCanary = true;
        }
    }

    g_editorHitProbeCalls.fetch_add(1);
    if (down) g_editorHitProbeDownCalls.fetch_add(1);

    IL2CPP::Il2CppObject* cameraObject = nullptr;
    int cameraSource = 0;
    if (g_editorHitCameraField.IsValid()) {
        Field<IL2CPP::Il2CppObject*> cameraField(g_editorHitCameraField);
        cameraObject = cameraField[editorSelf].Get();
        if (cameraObject != nullptr) cameraSource = 1;
    }
    if (cameraObject == nullptr && g_editorHitGetMainCamera.IsValid()) {
        cameraObject = g_editorHitGetMainCamera.Call();
        if (cameraObject != nullptr) cameraSource = 2;
    }
    g_editorHitCameraSource.store(cameraSource);

    if (cameraObject != nullptr && g_editorGetMousePosition.IsValid()) {
        const Vector3 mouse = g_editorGetMousePosition.Call();
        Method<Vector3> screenToWorld(g_editorHitScreenToWorld);
        const Vector3 world = screenToWorld[cameraObject].Call(mouse);
        const Vector2 point{world.x, world.y};
        IL2CPP::Il2CppObject* floor = g_editorHitGetFloorAtPosition.Call(point);
        g_editorCurrentHitFloor = floor;
        g_editorHitLastWorldX100.store(static_cast<int>(world.x * 100.0f));
        g_editorHitLastWorldY100.store(static_cast<int>(world.y * 100.0f));
        g_editorHitLastFloorNonNull.store(floor != nullptr ? 1 : 0);
        if (floor != nullptr) {
            g_editorHitFloorFound.fetch_add(1);
            if (down) g_editorHitDownFloorFound.fetch_add(1);
            else g_editorHitHeldFloorFound.fetch_add(1);
        }
    } else {
        g_editorHitLastFloorNonNull.store(0);
    }

    if (firstCanary) {
        ClearMarker(g_editorHitCallMarker);
        g_editorHitFirstCallProven.store(true, std::memory_order_release);
    }
}

'''
replace_once("void SnapshotEditorInput() {\n", hit_probe_code + "void SnapshotEditorInput(IL2CPP::Il2CppObject* editorSelf) {\n")

replace_once(
    "        if (g_editorGetScreenHeight.IsValid()) g_editorScreenHeight.store(g_editorGetScreenHeight.Call());\n"
    "    }\n"
    "}\n\n"
    "void HookEditorMouse",
    "        if (g_editorGetScreenHeight.IsValid()) g_editorScreenHeight.store(g_editorGetScreenHeight.Call());\n"
    "    }\n"
    "    ProbeEditorFloorHit(editorSelf, down, touchCount);\n"
    "}\n\n"
    "void HookEditorMouse"
)

replace_once("    SnapshotEditorInput();\n", "    SnapshotEditorInput(self);\n")
replace_once(
    "    g_editorCurrentMouseDown = false;\n"
    "    g_editorCurrentTouchActive = false;\n",
    "    g_editorCurrentMouseDown = false;\n"
    "    g_editorCurrentTouchActive = false;\n"
    "    g_editorCurrentHitFloor = nullptr;\n"
)

replace_once(
    "    if (g_editorCurrentMouseDown) g_editorSelectWhileMouseDown.fetch_add(1);\n"
    "    if (g_editorCurrentTouchActive) g_editorSelectWhileTouch.fetch_add(1);\n"
    "    g_editorLastFloorNonNull.store(floor != nullptr ? 1 : 0);\n",
    "    if (g_editorCurrentMouseDown) g_editorSelectWhileMouseDown.fetch_add(1);\n"
    "    if (g_editorCurrentTouchActive) g_editorSelectWhileTouch.fetch_add(1);\n"
    "    if (g_editorCurrentHitFloor != nullptr) {\n"
    "        g_editorSelectWhileHitFound.fetch_add(1);\n"
    "        if (floor == g_editorCurrentHitFloor) g_editorSelectMatchesHit.fetch_add(1);\n"
    "    }\n"
    "    g_editorLastFloorNonNull.store(floor != nullptr ? 1 : 0);\n"
)

replace_once(
    "    if (!ResolveEditorProbe(&handleMethod, &selectMethod) || !PrepareEditorFuse()) return;\n",
    "    if (!ResolveEditorProbe(&handleMethod, &selectMethod) || !PrepareEditorFuse()) return;\n"
    "    ResolveEditorHitProbe();\n"
)

# Fix the exact r12 calibration guard bug exposed on-device. BNM literal constants are
# intentionally marked const/non-static; Field<T>::Get handles _isConst through the static-value API.
replace_once(
    "    const bool abi = sentinelBase.IsValid() && sentinelBase._isStatic && sentinelBase._isConst &&\n",
    "    g_calibrationSentinelValid.store(sentinelBase.IsValid() ? 1 : 0);\n"
    "    g_calibrationSentinelConst.store(sentinelBase._isConst ? 1 : 0);\n"
    "    g_calibrationSentinelStatic.store(sentinelBase._isStatic ? 1 : 0);\n"
    "    g_calibrationSentinelTypeFloat.store(\n"
    "            sentinelBase.IsValid() && SameClass(sentinelBase.GetType(), floatClass) ? 1 : 0);\n"
    "    g_calibrationGetterResolved.store(\n"
    "            getterInfo != nullptr && getterInfo->methodPointer != nullptr ? 1 : 0);\n"
    "    g_calibrationSetterResolved.store(\n"
    "            setterInfo != nullptr && setterInfo->methodPointer != nullptr ? 1 : 0);\n"
    "    g_calibrationSaveResolved.store(\n"
    "            saveInfo != nullptr && saveInfo->methodPointer != nullptr ? 1 : 0);\n"
    "    const bool abi = sentinelBase.IsValid() && sentinelBase._isConst &&\n"
    "            !sentinelBase._isThreadStatic &&\n"
)

replace_once(
    "nativeProbe=cache-post-bnm-scneditor-input-edge-calibration-v1\\n",
    "nativeProbe=cache-post-bnm-scneditor-hitprobe-calibration-v2\\n"
)
replace_once(
    "nativeStage=post-bnm-scneditor-input-edge-and-calibration\\n",
    "nativeStage=post-bnm-scneditor-world-hit-and-calibration\\n"
)
replace_once("abiProbeRevision=12\\n", "abiProbeRevision=13\\n")
replace_once(
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n',
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n'
    '        << "editorHitPolicy=screen-to-world-rdutils-observe-only" << \'\\n\'\n'
    '        << "editorHitMutation=0" << \'\\n\'\n'
    '        << "editorHitAbiGuard=" << g_editorHitAbiGuard.load() << \'\\n\'\n'
    '        << "editorHitCameraFieldGuard=" << g_editorHitCameraFieldGuard.load() << \'\\n\'\n'
    '        << "editorHitMainCameraGuard=" << g_editorHitMainCameraGuard.load() << \'\\n\'\n'
    '        << "editorHitScreenToWorldGuard=" << g_editorHitScreenToWorldGuard.load() << \'\\n\'\n'
    '        << "editorHitFloorMethodGuard=" << g_editorHitFloorMethodGuard.load() << \'\\n\'\n'
    '        << "editorHitMarkerReady=" << g_editorHitMarkerReady.load() << \'\\n\'\n'
    '        << "editorHitRecoveryState=" << g_editorHitRecoveryState.load() << \'\\n\'\n'
    '        << "editorHitFirstCallProven=" << (g_editorHitFirstCallProven.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorHitProbeCalls=" << g_editorHitProbeCalls.load() << \'\\n\'\n'
    '        << "editorHitProbeDownCalls=" << g_editorHitProbeDownCalls.load() << \'\\n\'\n'
    '        << "editorHitFloorFound=" << g_editorHitFloorFound.load() << \'\\n\'\n'
    '        << "editorHitDownFloorFound=" << g_editorHitDownFloorFound.load() << \'\\n\'\n'
    '        << "editorHitHeldFloorFound=" << g_editorHitHeldFloorFound.load() << \'\\n\'\n'
    '        << "editorHitCameraSource=" << g_editorHitCameraSource.load() << \'\\n\'\n'
    '        << "editorHitLastWorldX100=" << g_editorHitLastWorldX100.load() << \'\\n\'\n'
    '        << "editorHitLastWorldY100=" << g_editorHitLastWorldY100.load() << \'\\n\'\n'
    '        << "editorHitLastFloorNonNull=" << g_editorHitLastFloorNonNull.load() << \'\\n\'\n'
    '        << "editorSelectWhileHitFound=" << g_editorSelectWhileHitFound.load() << \'\\n\'\n'
    '        << "editorSelectMatchesHit=" << g_editorSelectMatchesHit.load() << \'\\n\'\n'
)

replace_once(
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel" << \'\\n\'\n',
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel-bnm-const-v2" << \'\\n\'\n'
)
replace_once(
    '        << "calibrationAfterX100=" << g_calibrationAfterX100.load() << \'\\n\'\n',
    '        << "calibrationAfterX100=" << g_calibrationAfterX100.load() << \'\\n\'\n'
    '        << "calibrationSentinelValid=" << g_calibrationSentinelValid.load() << \'\\n\'\n'
    '        << "calibrationSentinelConst=" << g_calibrationSentinelConst.load() << \'\\n\'\n'
    '        << "calibrationSentinelStatic=" << g_calibrationSentinelStatic.load() << \'\\n\'\n'
    '        << "calibrationSentinelTypeFloat=" << g_calibrationSentinelTypeFloat.load() << \'\\n\'\n'
    '        << "calibrationGetterResolved=" << g_calibrationGetterResolved.load() << \'\\n\'\n'
    '        << "calibrationSetterResolved=" << g_calibrationSetterResolved.load() << \'\\n\'\n'
    '        << "calibrationSaveResolved=" << g_calibrationSaveResolved.load() << \'\\n\'\n'
)

for marker in (
    "abiProbeRevision=13",
    "nativeProbe=cache-post-bnm-scneditor-hitprobe-calibration-v2",
    "editorInputEdgePolicy=observe-only",
    "editorHitPolicy=screen-to-world-rdutils-observe-only",
    "editorHitMutation=0",
    'Class rdUtils("", "RDUtils")',
    'rdUtils.GetMethod("GetFloorAtPosition", 1)',
    'camera.GetMethod("ScreenToWorldPoint", 1)',
    'editor.GetField("camera")',
    "editor-r13-hit.pending",
    "editorHitFloorFound=",
    "editorSelectMatchesHit=",
    "calibrationPolicy=neutralize-only-exact-game-sentinel-bnm-const-v2",
    "sentinelBase.IsValid() && sentinelBase._isConst",
    "!sentinelBase._isThreadStatic",
    "calibrationSentinelConst=",
):
    if marker not in s:
        raise SystemExit(f"r13 marker missing after overlay: {marker}")

if "sentinelBase.IsValid() && sentinelBase._isStatic && sentinelBase._isConst" in s:
    raise SystemExit("r12 impossible const/static calibration guard survived")
if "g_oldEditorMouse(self, methodInfo)" not in s or "g_oldSelectFloor(self, floor, cameraJump, methodInfo)" not in s:
    raise SystemExit("r13 must preserve editor pass-through originals")

path.write_text(s, encoding="utf-8")
