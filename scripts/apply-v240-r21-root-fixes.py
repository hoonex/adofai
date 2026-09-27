#!/usr/bin/env python3
"""R21: exact persisted-calibration confidence restore + narrow editor collider synchronization.

Both fixes come from the authoritative v2.4 APK, not inferred device coordinates.

Calibration root cause
----------------------
CalibrationPreset is a value type. Exact target disassembly and metadata establish:
  CalibrationPreset.ToDict        RVA 0x1F2BF78
  CalibrationPreset.FromDict      RVA 0x1F2C0E0
  scnSplash.GoToMenu              RVA 0x0C4F5A0
  CalibrationPreset.confident     unboxed offset +24

ToDict serializes outputType/outputName/inputOffset but omits confident. FromDict restores the
serialized members but never restores confident. scnSplash.GoToMenu gates startup calibration on
scrConductor.currentPreset.confident. Therefore a user preset can be saved correctly yet reload as
unconfident on every launch.

R21 hooks only CalibrationPreset.FromDict. After the original successfully returns, the exact
metadata-validated confident byte is restored to true. Fresh/default/fallback presets do not pass
through FromDict and retain the game's original one-time calibration behavior. R19/R20 offset writes
are disabled.

Editor tile root cause
----------------------
Exact scnEditor.ObjectsAtMouse disassembly (RVA 0x22E8DF4) creates/enables transient floor
colliders and immediately calls Physics2D.RaycastAll (RVA 0x1B9FF44). On devices where physics
transforms are not auto-synchronized before that query, clicks become frame/timing dependent.
R21 preserves the original mouse position, camera conversion, layer mask, raycast arguments/results
and SelectFloor logic. It only calls the engine's own Physics2D::SyncTransforms once, immediately
before the first original RaycastAll inside a touch-driven ObjectsAtMouse invocation.

The existing SHORT_EDGES/full-width Android compatibility policy remains in place.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r21-root-fixes.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r21 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)

once(
    'std::string g_calibrationR20Marker;\n',
    'std::string g_calibrationR20Marker;\n'
    'using CalibrationFromDictFn = void (*)(void*, IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);\n'
    'CalibrationFromDictFn g_oldCalibrationR21FromDict = nullptr;\n'
    'std::atomic<int> g_calibrationR21AbiGuard{0};\n'
    'std::atomic<int> g_calibrationR21HookInstalled{0};\n'
    'std::atomic<int> g_calibrationR21InstallAttempted{0};\n'
    'std::atomic<int> g_calibrationR21MarkerReady{0};\n'
    'std::atomic<int> g_calibrationR21RecoveryState{0};\n'
    'std::atomic<int> g_calibrationR21FirstMutationProven{0};\n'
    'std::atomic<int> g_calibrationR21Calls{0};\n'
    'std::atomic<int> g_calibrationR21ConfidenceRestored{0};\n'
    'std::atomic<int> g_calibrationR21AlreadyConfident{0};\n'
    'std::atomic<int> g_calibrationR21ConfidentOffset{-1};\n'
    'std::mutex g_calibrationR21FirstMutationMutex;\n'
    'std::string g_calibrationR21InstallMarker;\n'
    'std::string g_calibrationR21MutationMarker;\n'
    'EditorObjectsFn g_oldTileR21ObjectsAtMouse = nullptr;\n'
    'EditorRaycastFn g_oldTileR21Raycast = nullptr;\n'
    'PhysicsSyncFn g_tileR21SyncTransforms = nullptr;\n'
    'Method<int> g_tileR21GetTouchCount;\n'
    'thread_local int g_tileR21ObjectsDepth = 0;\n'
    'thread_local bool g_tileR21TouchActive = false;\n'
    'thread_local bool g_tileR21SyncedThisObjectsCall = false;\n'
    'std::atomic<int> g_tileR21AbiGuard{0};\n'
    'std::atomic<int> g_tileR21IcallReady{0};\n'
    'std::atomic<int> g_tileR21HookInstalled{0};\n'
    'std::atomic<int> g_tileR21InstallAttempted{0};\n'
    'std::atomic<int> g_tileR21MarkerReady{0};\n'
    'std::atomic<int> g_tileR21RecoveryState{0};\n'
    'std::atomic<int> g_tileR21ObjectsFirstCallProven{0};\n'
    'std::atomic<int> g_tileR21RaycastFirstCallProven{0};\n'
    'std::atomic<int> g_tileR21ObjectsCalls{0};\n'
    'std::atomic<int> g_tileR21TouchObjectsCalls{0};\n'
    'std::atomic<int> g_tileR21RaycastCalls{0};\n'
    'std::atomic<int> g_tileR21SyncCalls{0};\n'
    'std::mutex g_tileR21ObjectsFirstCallMutex;\n'
    'std::mutex g_tileR21RaycastFirstCallMutex;\n'
    'std::string g_tileR21InstallMarker;\n'
    'std::string g_tileR21ObjectsCallMarker;\n'
    'std::string g_tileR21RaycastCallMarker;\n'
)

r21_code = r'''
constexpr BNM_PTR kCalibrationR21ConfidentOffset = 24;

bool PrepareCalibrationR21Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_calibrationR21RecoveryState.store(4);
        return false;
    }
    g_calibrationR21InstallMarker = dir + "/calibration-r21-install.pending";
    g_calibrationR21MutationMarker = dir + "/calibration-r21-confidence.pending";
    g_calibrationR21MarkerReady.store(1);
    if (MarkerExists(g_calibrationR21InstallMarker)) {
        g_calibrationR21RecoveryState.store(1);
        return false;
    }
    if (MarkerExists(g_calibrationR21MutationMarker)) {
        g_calibrationR21RecoveryState.store(2);
        return false;
    }
    const std::string probe = dir + "/calibration-r21-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_calibrationR21MarkerReady.store(0);
        g_calibrationR21RecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void HookCalibrationR21FromDict(void* self, IL2CPP::Il2CppObject* dict,
                                IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldCalibrationR21FromDict) return;
    // First let the exact original deserialize outputType/outputName/inputOffset.
    g_oldCalibrationR21FromDict(self, dict, methodInfo);
    g_calibrationR21Calls.fetch_add(1);
    if (!self || g_calibrationR21ConfidentOffset.load() !=
                 static_cast<int>(kCalibrationR21ConfidentOffset)) return;

    auto* confident = reinterpret_cast<uint8_t*>(
            reinterpret_cast<uintptr_t>(self) + kCalibrationR21ConfidentOffset);
    if (*confident != 0) {
        g_calibrationR21AlreadyConfident.fetch_add(1);
        return;
    }

    bool firstMutation = false;
    if (!g_calibrationR21FirstMutationProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_calibrationR21FirstMutationMutex);
        if (!g_calibrationR21FirstMutationProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_calibrationR21MutationMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return;
            }
            firstMutation = true;
        }
    }

    *confident = 1;
    g_calibrationR21ConfidenceRestored.fetch_add(1);
    if (firstMutation) {
        ClearMarker(g_calibrationR21MutationMarker);
        g_calibrationR21FirstMutationProven.store(1, std::memory_order_release);
    }
}

void MaybeInstallCalibrationR21() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_calibrationR21HookInstalled.load() ||
        g_calibrationR21InstallAttempted.load() ||
        g_calibrationR21RecoveryState.load() != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_calibrationR21HookInstalled.load() ||
        g_calibrationR21InstallAttempted.load()) return;

    Class preset("", "CalibrationPreset");
    Class boolClass = Defaults::Get<bool>();
    if (!preset || !boolClass || !preset.GetIl2CppType() ||
        preset.GetIl2CppType()->type != IL2CPP::IL2CPP_TYPE_VALUETYPE) return;

    MethodBase fromDict = preset.GetMethod("FromDict", 1);
    FieldBase confident = preset.GetField("confident");
    IL2CPP::MethodInfo* info = fromDict.IsValid() ? fromDict.GetInfo() : nullptr;

    const bool methodAbi = info && info->methodPointer && !fromDict._isStatic &&
            info->parameters_count == 1 && info->parameters &&
            info->parameters[0] && TypeByRef(info->parameters[0]) == 0 &&
            info->return_type && TypeCode(info->return_type) == 1;
    const bool fieldAbi = confident.IsValid() && !confident._isStatic &&
            !confident._isConst && SameClass(confident.GetType(), boolClass) &&
            confident.GetOffset() == kCalibrationR21ConfidentOffset;
    const bool abi = methodAbi && fieldAbi;
    g_calibrationR21AbiGuard.store(abi ? 1 : 0);
    g_calibrationR21ConfidentOffset.store(
            fieldAbi ? static_cast<int>(confident.GetOffset()) : -1);
    if (!abi || !PrepareCalibrationR21Fuse()) return;

    if (!WriteMarker(g_calibrationR21InstallMarker)) {
        g_calibrationR21MarkerReady.store(0);
        g_calibrationR21RecoveryState.store(4);
        return;
    }
    g_calibrationR21InstallAttempted.store(1);
    BasicHook(fromDict, HookCalibrationR21FromDict, g_oldCalibrationR21FromDict);
    const bool installed = g_oldCalibrationR21FromDict != nullptr;
    g_calibrationR21HookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_calibrationR21InstallMarker);
    else g_calibrationR21RecoveryState.store(5);
}

bool PrepareTileR21Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_tileR21RecoveryState.store(4);
        return false;
    }
    g_tileR21InstallMarker = dir + "/editor-r21-tile-install.pending";
    g_tileR21ObjectsCallMarker = dir + "/editor-r21-objects-call.pending";
    g_tileR21RaycastCallMarker = dir + "/editor-r21-raycast-call.pending";
    g_tileR21MarkerReady.store(1);
    if (MarkerExists(g_tileR21InstallMarker)) {
        g_tileR21RecoveryState.store(1);
        return false;
    }
    if (MarkerExists(g_tileR21ObjectsCallMarker)) {
        g_tileR21RecoveryState.store(2);
        return false;
    }
    if (MarkerExists(g_tileR21RaycastCallMarker)) {
        g_tileR21RecoveryState.store(3);
        return false;
    }
    const std::string probe = dir + "/editor-r21-tile-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_tileR21MarkerReady.store(0);
        g_tileR21RecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

EditorObjectsArray* HookTileR21ObjectsAtMouse(IL2CPP::Il2CppObject* self,
                                               IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldTileR21ObjectsAtMouse) return nullptr;
    bool first = false;
    if (!g_tileR21ObjectsFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_tileR21ObjectsFirstCallMutex);
        if (!g_tileR21ObjectsFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_tileR21ObjectsCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return g_oldTileR21ObjectsAtMouse(self, methodInfo);
            }
            first = true;
        }
    }

    const bool outer = g_tileR21ObjectsDepth == 0;
    const bool previousTouch = g_tileR21TouchActive;
    const bool previousSynced = g_tileR21SyncedThisObjectsCall;
    ++g_tileR21ObjectsDepth;
    if (outer) {
        g_tileR21TouchActive =
                g_tileR21GetTouchCount.IsValid() && g_tileR21GetTouchCount.Call() > 0;
        g_tileR21SyncedThisObjectsCall = false;
        g_tileR21ObjectsCalls.fetch_add(1);
        if (g_tileR21TouchActive) g_tileR21TouchObjectsCalls.fetch_add(1);
    }

    EditorObjectsArray* result = g_oldTileR21ObjectsAtMouse(self, methodInfo);

    --g_tileR21ObjectsDepth;
    if (outer) {
        g_tileR21TouchActive = previousTouch;
        g_tileR21SyncedThisObjectsCall = previousSynced;
    }
    if (first) {
        ClearMarker(g_tileR21ObjectsCallMarker);
        g_tileR21ObjectsFirstCallProven.store(1, std::memory_order_release);
    }
    return result;
}

Array<IL2CPP::Il2CppObject*>* HookTileR21Raycast(
        Vector2 origin, Vector2 direction, float distance, int layerMask,
        IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldTileR21Raycast) return nullptr;
    if (g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive) {
        return g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);
    }

    bool first = false;
    if (!g_tileR21RaycastFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_tileR21RaycastFirstCallMutex);
        if (!g_tileR21RaycastFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_tileR21RaycastCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);
            }
            first = true;
        }
    }

    g_tileR21RaycastCalls.fetch_add(1);
    if (!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms) {
        // Synchronize exactly after ObjectsAtMouse created/enabled its transient floor
        // colliders and immediately before its first original raycast.
        g_tileR21SyncTransforms();
        g_tileR21SyncedThisObjectsCall = true;
        g_tileR21SyncCalls.fetch_add(1);
    }

    auto* result = g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);
    if (first) {
        ClearMarker(g_tileR21RaycastCallMarker);
        g_tileR21RaycastFirstCallProven.store(1, std::memory_order_release);
    }
    return result;
}

void MaybeInstallTileR21() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_tileR21HookInstalled.load() ||
        g_tileR21InstallAttempted.load() ||
        g_tileR21RecoveryState.load() != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_tileR21HookInstalled.load() || g_tileR21InstallAttempted.load()) return;

    Class editor("", "scnEditor");
    Class physics("UnityEngine", "Physics2D");
    Class input("UnityEngine", "Input");
    Class vector2 = Defaults::Get<Vector2>();
    Class floatClass = Defaults::Get<float>();
    Class intClass = Defaults::Get<int>();
    if (!editor || !physics || !input || !vector2 || !floatClass || !intClass) return;

    MethodBase objects = editor.GetMethod("ObjectsAtMouse", 0);
    MethodBase raycast = physics.GetMethod("RaycastAll", 4);
    IL2CPP::MethodInfo* objectsInfo = objects.IsValid() ? objects.GetInfo() : nullptr;
    IL2CPP::MethodInfo* raycastInfo = raycast.IsValid() ? raycast.GetInfo() : nullptr;

    const bool objectsAbi = objectsInfo && objectsInfo->methodPointer &&
            !objects._isStatic && objectsInfo->parameters_count == 0 &&
            objectsInfo->return_type && TypeByRef(objectsInfo->return_type) == 0 &&
            MetadataTypeName(objectsInfo->return_type) == "UnityEngine.GameObject[]";

    const bool raycastParams = raycastInfo && raycastInfo->parameters_count == 4 &&
            raycastInfo->parameters && raycastInfo->parameters[0] &&
            raycastInfo->parameters[1] && raycastInfo->parameters[2] &&
            raycastInfo->parameters[3];
    const bool raycastAbi = raycastInfo && raycastInfo->methodPointer &&
            raycast._isStatic && raycastParams && raycastInfo->return_type &&
            TypeCode(raycastInfo->return_type) == 29 &&
            SameClass(Class(raycastInfo->parameters[0]), vector2) &&
            SameClass(Class(raycastInfo->parameters[1]), vector2) &&
            SameClass(Class(raycastInfo->parameters[2]), floatClass) &&
            SameClass(Class(raycastInfo->parameters[3]), intClass) &&
            TypeByRef(raycastInfo->parameters[0]) == 0 &&
            TypeByRef(raycastInfo->parameters[1]) == 0 &&
            TypeByRef(raycastInfo->parameters[2]) == 0 &&
            TypeByRef(raycastInfo->parameters[3]) == 0;

    g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);
    const bool touchAbi = g_tileR21GetTouchCount.IsValid();

    auto resolve = reinterpret_cast<ResolveIcallFn>(
            dlsym(RTLD_DEFAULT, "il2cpp_resolve_icall"));
    PhysicsSyncFn sync = nullptr;
    if (resolve) {
        sync = reinterpret_cast<PhysicsSyncFn>(
                resolve("UnityEngine.Physics2D::SyncTransforms()"));
        if (!sync) {
            sync = reinterpret_cast<PhysicsSyncFn>(
                    resolve("UnityEngine.Physics2D::SyncTransforms"));
        }
    }
    g_tileR21IcallReady.store(sync ? 1 : 0);

    const bool abi = objectsAbi && raycastAbi && touchAbi && sync;
    g_tileR21AbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareTileR21Fuse()) return;

    if (!WriteMarker(g_tileR21InstallMarker)) {
        g_tileR21MarkerReady.store(0);
        g_tileR21RecoveryState.store(4);
        return;
    }
    g_tileR21InstallAttempted.store(1);
    g_tileR21SyncTransforms = sync;

    // Install both before clearing the fuse. The ObjectsAtMouse hook is only context
    // tracking; without the raycast hook it does not change game behavior.
    BasicHook(objects, HookTileR21ObjectsAtMouse, g_oldTileR21ObjectsAtMouse);
    BasicHook(raycast, HookTileR21Raycast, g_oldTileR21Raycast);
    const bool installed = g_oldTileR21ObjectsAtMouse != nullptr &&
            g_oldTileR21Raycast != nullptr && g_tileR21SyncTransforms != nullptr;
    g_tileR21HookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_tileR21InstallMarker);
    else g_tileR21RecoveryState.store(5);
}

'''
once('constexpr float kCalibrationR20Unset = 999.0f;\n',
     r21_code + 'constexpr float kCalibrationR20Unset = 999.0f;\n')

# R20 repaired a symptom (offset sentinel) but not the persistence defect. Disable all
# calibration-value writes; install only the exact FromDict confidence repair.
# R21 also enables the narrowly-scoped collider synchronization independently of old probes.
once(
    '    MaybeRepairCalibrationR20();\n'
    '    MaybeInstallSfbHook();\n',
    '    MaybeInstallCalibrationR21();\n'
    '    MaybeInstallTileR21();\n'
    '    MaybeInstallSfbHook();\n'
)

once('        << "stabilityRevision=20" << \'\\n\'\n',
     '        << "stabilityRevision=21" << \'\\n\'\n')
once(
    'activeTilePolicy=original-v240-editor-path-plus-full-width-window',
    'activeTilePolicy=r21-transient-collider-sync-plus-full-width-window'
)
once(
    'editorPhysicsPolicy=disabled-r18-window-viewport-fix',
    'editorPhysicsPolicy=forensic-r17-disabled-r21-exact-production-sync-active'
)
once(
    'calibrationR20Policy=exact-Persistence-PlayerPrefsJson-sentinel-999-to-zero',
    'calibrationR20Policy=disabled-r21-symptom-write-forensic-only'
)
once(
    '        << "calibrationR20AfterX100=" << g_calibrationR20AfterX100.load() << \'\\n\'\n',
    '        << "calibrationR20AfterX100=" << g_calibrationR20AfterX100.load() << \'\\n\'\n'
    '        << "calibrationR21Policy=persisted-CalibrationPreset-FromDict-restore-confident" << \'\\n\'\n'
    '        << "calibrationR21Source=authoritative-v240-binary" << \'\\n\'\n'
    '        << "calibrationR21Mutation=memory-only-persisted-preset-confidence-no-offset-write" << \'\\n\'\n'
    '        << "calibrationR21FromDictRva=0x1F2C0E0" << \'\\n\'\n'
    '        << "calibrationR21ToDictRva=0x1F2BF78" << \'\\n\'\n'
    '        << "calibrationR21SplashGoToMenuRva=0x0C4F5A0" << \'\\n\'\n'
    '        << "calibrationR21ConfidentOffsetExpected=24" << \'\\n\'\n'
    '        << "calibrationR21ConfidentOffset=" << g_calibrationR21ConfidentOffset.load() << \'\\n\'\n'
    '        << "calibrationR21AbiGuard=" << g_calibrationR21AbiGuard.load() << \'\\n\'\n'
    '        << "calibrationR21HookInstalled=" << g_calibrationR21HookInstalled.load() << \'\\n\'\n'
    '        << "calibrationR21InstallAttempted=" << g_calibrationR21InstallAttempted.load() << \'\\n\'\n'
    '        << "calibrationR21MarkerReady=" << g_calibrationR21MarkerReady.load() << \'\\n\'\n'
    '        << "calibrationR21RecoveryState=" << g_calibrationR21RecoveryState.load() << \'\\n\'\n'
    '        << "calibrationR21FirstMutationProven=" << g_calibrationR21FirstMutationProven.load() << \'\\n\'\n'
    '        << "calibrationR21Calls=" << g_calibrationR21Calls.load() << \'\\n\'\n'
    '        << "calibrationR21ConfidenceRestored=" << g_calibrationR21ConfidenceRestored.load() << \'\\n\'\n'
    '        << "calibrationR21AlreadyConfident=" << g_calibrationR21AlreadyConfident.load() << \'\\n\'\n'
    '        << "tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll" << \'\\n\'\n'
    '        << "tileR21Source=authoritative-v240-binary" << \'\\n\'\n'
    '        << "tileR21ObjectsAtMouseRva=0x22E8DF4" << \'\\n\'\n'
    '        << "tileR21RaycastAllRva=0x1B9FF44" << \'\\n\'\n'
    '        << "tileR21CoordinatesModified=0" << \'\\n\'\n'
    '        << "tileR21RaycastArgumentsModified=0" << \'\\n\'\n'
    '        << "tileR21AbiGuard=" << g_tileR21AbiGuard.load() << \'\\n\'\n'
    '        << "tileR21IcallReady=" << g_tileR21IcallReady.load() << \'\\n\'\n'
    '        << "tileR21HookInstalled=" << g_tileR21HookInstalled.load() << \'\\n\'\n'
    '        << "tileR21InstallAttempted=" << g_tileR21InstallAttempted.load() << \'\\n\'\n'
    '        << "tileR21MarkerReady=" << g_tileR21MarkerReady.load() << \'\\n\'\n'
    '        << "tileR21RecoveryState=" << g_tileR21RecoveryState.load() << \'\\n\'\n'
    '        << "tileR21ObjectsFirstCallProven=" << g_tileR21ObjectsFirstCallProven.load() << \'\\n\'\n'
    '        << "tileR21RaycastFirstCallProven=" << g_tileR21RaycastFirstCallProven.load() << \'\\n\'\n'
    '        << "tileR21ObjectsCalls=" << g_tileR21ObjectsCalls.load() << \'\\n\'\n'
    '        << "tileR21TouchObjectsCalls=" << g_tileR21TouchObjectsCalls.load() << \'\\n\'\n'
    '        << "tileR21RaycastCalls=" << g_tileR21RaycastCalls.load() << \'\\n\'\n'
    '        << "tileR21SyncCalls=" << g_tileR21SyncCalls.load() << \'\\n\'\n'
)

for marker in (
    'stabilityRevision=21',
    'calibrationR20Policy=disabled-r21-symptom-write-forensic-only',
    'calibrationR21Policy=persisted-CalibrationPreset-FromDict-restore-confident',
    'calibrationR21FromDictRva=0x1F2C0E0',
    'calibrationR21ConfidentOffsetExpected=24',
    'preset.GetField("confident")',
    'confident.GetOffset() == kCalibrationR21ConfidentOffset',
    'BasicHook(fromDict, HookCalibrationR21FromDict, g_oldCalibrationR21FromDict)',
    'calibration-r21-confidence.pending',
    'tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll',
    'tileR21ObjectsAtMouseRva=0x22E8DF4',
    'tileR21RaycastAllRva=0x1B9FF44',
    'tileR21CoordinatesModified=0',
    'resolve("UnityEngine.Physics2D::SyncTransforms()")',
    'BasicHook(objects, HookTileR21ObjectsAtMouse, g_oldTileR21ObjectsAtMouse)',
    'BasicHook(raycast, HookTileR21Raycast, g_oldTileR21Raycast)',
    'MaybeInstallCalibrationR21();',
    'MaybeInstallTileR21();',
):
    if marker not in s:
        raise SystemExit(f"r21 marker missing after transform: {marker}")

for forbidden_call in (
    '    MaybeRepairCalibrationR20();\n',
    '    MaybeRepairCalibrationR19();\n',
    '    MaybeNeutralizeUnsetCalibration();\n',
    '    MaybeInstallEditorProbe();\n',
    '    MaybeInstallEditorObjectsProbe();\n',
    '    MaybeInstallEditorPhysicsSync();\n',
):
    if forbidden_call in s:
        raise SystemExit(f"r21 forbidden active call survived: {forbidden_call.strip()}")

if s.count('BasicHook(') != 12:
    raise SystemExit(f"r21 expected twelve compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r22 = Path(__file__).with_name("apply-v240-r22-calibration-persist.py")
if not r22.is_file():
    raise SystemExit(f"missing r22 overlay: {r22}")
__import__("subprocess").run([sys.executable, str(r22), str(path)], check=True)
