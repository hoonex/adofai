#!/usr/bin/env python3
"""R17: synchronize dynamically created editor floor colliders before their original raycast.

Source evidence: exact v2.4 metadata for Assembly-CSharp.dll:
 scnEditor.ObjectsAtMouse token 0x060006f2, RVA 0x22e8df4;
 Physics2D.RaycastAll token 0x06000023, RVA 0x1b9ff44.
 At 0x22e91bc/0x22e9268 the method creates/enables floor colliders;
 at 0x22e9400/0x22e9454 it performs zero-distance 2D raycasts on those colliders.
 The original libunity.so contains UnityEngine.Physics2D::SyncTransforms.
This avoids changing screen coordinates, input edges, layer masks, raycast arguments or results.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r17-editor-physics-sync.py <V240CacheLoader.cpp>")
path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")
def once(old, new):
    global s
    n=s.count(old)
    if n != 1:
        raise SystemExit(f"r17 anchor must occur once, got {n}: {old[:110]!r}")
    s=s.replace(old,new,1)

once(
    'std::string g_editorObjectsReadMarker;\n',
    'std::string g_editorObjectsReadMarker;\n'
    'using EditorRaycastFn = Array<IL2CPP::Il2CppObject*>* (*)(Vector2, Vector2, float, int, IL2CPP::MethodInfo*);\n'
    'using ResolveIcallFn = void* (*)(const char*);\n'
    'using PhysicsSyncFn = void (*)();\n'
    'EditorRaycastFn g_oldEditorRaycast = nullptr;\n'
    'PhysicsSyncFn g_editorPhysicsSync = nullptr;\n'
    'Method<int> g_editorPhysicsFrameCount;\n'
    'thread_local int g_editorObjectsDepth = 0;\n'
    'std::atomic<int> g_editorPhysicsLastFrame{-1};\n'
    'std::atomic<int> g_editorPhysicsSyncCalls{0};\n'
    'std::atomic<int> g_editorPhysicsRaycastCalls{0};\n'
    'std::atomic<int> g_editorPhysicsTouchRaycasts{0};\n'
    'std::atomic<int> g_editorPhysicsSyncUnavailable{0};\n'
    'std::atomic<int> g_editorPhysicsAbiGuard{0};\n'
    'std::atomic<int> g_editorPhysicsIcallReady{0};\n'
    'std::atomic<int> g_editorPhysicsHookInstalled{0};\n'
    'std::atomic<int> g_editorPhysicsRecoveryState{0};\n'
    'std::atomic<int> g_editorPhysicsMarkerReady{0};\n'
    'std::atomic<int> g_editorPhysicsFirstCallProven{0};\n'
    'std::atomic<int> g_editorPhysicsInstallAttempted{0};\n'
    'std::mutex g_editorPhysicsFirstCallMutex;\n'
    'std::string g_editorPhysicsInstallMarker;\n'
    'std::string g_editorPhysicsCallMarker;\n'
)
once(
    '    EditorObjectsArray* result = g_oldEditorObjectsAtMouse(self, methodInfo);\n',
    '    ++g_editorObjectsDepth;\n'
    '    EditorObjectsArray* result = g_oldEditorObjectsAtMouse(self, methodInfo);\n'
    '    --g_editorObjectsDepth;\n'
)
physics = r"""
// The historical game constructs/enables floor colliders INSIDE ObjectsAtMouse and
// immediately performs two Physics2D.RaycastAll queries. SyncTransforms must run after
// collider creation and before the first original raycast, not before ObjectsAtMouse.
bool PrepareEditorPhysicsFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_editorPhysicsRecoveryState.store(4); return false; }
    g_editorPhysicsInstallMarker = dir + "/editor-r17-physics-install.pending";
    g_editorPhysicsCallMarker = dir + "/editor-r17-physics-call.pending";
    g_editorPhysicsMarkerReady.store(1);
    if (MarkerExists(g_editorPhysicsInstallMarker)) { g_editorPhysicsRecoveryState.store(1); return false; }
    if (MarkerExists(g_editorPhysicsCallMarker)) { g_editorPhysicsRecoveryState.store(2); return false; }
    const std::string probe = dir + "/editor-r17-physics-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) { g_editorPhysicsMarkerReady.store(0); g_editorPhysicsRecoveryState.store(4); return false; }
    ClearMarker(probe);
    return true;
}

Array<IL2CPP::Il2CppObject*>* HookEditorPhysicsRaycast(
        Vector2 origin, Vector2 direction, float distance, int layerMask,
        IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorRaycast) return nullptr;
    // No physics changes during gameplay, menus, UI input or non-touch editor input.
    if (g_editorObjectsDepth <= 0 || !g_editorCurrentTouchActive) {
        return g_oldEditorRaycast(origin, direction, distance, layerMask, methodInfo);
    }
    g_editorPhysicsTouchRaycasts.fetch_add(1);
    bool first = false;
    if (!g_editorPhysicsFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_editorPhysicsFirstCallMutex);
        if (!g_editorPhysicsFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_editorPhysicsCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return g_oldEditorRaycast(origin, direction, distance, layerMask, methodInfo);
            }
            first = true;
        }
    }
    g_editorPhysicsRaycastCalls.fetch_add(1);
    // The original ObjectsAtMouse caches once per frame. Avoid two expensive syncs
    // for its two floor-layer raycasts, and never touch its origin/layer/distance.
    const int frame = g_editorPhysicsFrameCount.IsValid()
            ? g_editorPhysicsFrameCount.Call() : -1;
    if (frame >= 0 && g_editorPhysicsLastFrame.exchange(frame) != frame) {
        if (g_editorPhysicsSync) {
            g_editorPhysicsSync();
            g_editorPhysicsSyncCalls.fetch_add(1);
        } else {
            g_editorPhysicsSyncUnavailable.fetch_add(1);
        }
    }
    auto* result = g_oldEditorRaycast(origin, direction, distance, layerMask, methodInfo);
    if (first) {
        ClearMarker(g_editorPhysicsCallMarker);
        g_editorPhysicsFirstCallProven.store(1, std::memory_order_release);
    }
    return result;
}

void MaybeInstallEditorPhysicsSync() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire)
        || !g_editorObjectsHookInstalled.load()
        || g_editorPhysicsHookInstalled.load()
        || g_editorPhysicsInstallAttempted.load()
        || g_editorPhysicsRecoveryState.load()) return;
    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_editorPhysicsHookInstalled.load() || g_editorPhysicsInstallAttempted.load()) return;

    Class physics("UnityEngine", "Physics2D");
    Class time("UnityEngine", "Time");
    if (!physics || !time) return;
    MethodBase raycast = physics.GetMethod("RaycastAll", 4);
    IL2CPP::MethodInfo* info = raycast.IsValid() ? raycast.GetInfo() : nullptr;
    const bool abi = info && info->methodPointer && raycast._isStatic
            && info->parameters_count == 4 && info->return_type
            && TypeByRef(info->return_type) == 0
            && TypeCode(info->return_type) == 29;
    g_editorPhysicsAbiGuard.store(abi ? 1 : 0);
    if (!abi) return;
    g_editorPhysicsFrameCount = time.GetMethod("get_frameCount", 0);
    if (!g_editorPhysicsFrameCount.IsValid()) return;

    // Resolve against original Unity's registered internal call, not an unproven
    // replacement binary or a stripped managed SyncTransforms method.
    auto resolve = reinterpret_cast<ResolveIcallFn>(
            dlsym(RTLD_DEFAULT, "il2cpp_resolve_icall"));
    if (!resolve) return;
    auto sync = reinterpret_cast<PhysicsSyncFn>(
            resolve("UnityEngine.Physics2D::SyncTransforms()"));
    if (!sync) sync = reinterpret_cast<PhysicsSyncFn>(
            resolve("UnityEngine.Physics2D::SyncTransforms"));
    if (!sync) return; // fail open: keep the original editor path.
    g_editorPhysicsSync = sync;
    g_editorPhysicsIcallReady.store(1);
    if (!PrepareEditorPhysicsFuse()) return;
    if (!WriteMarker(g_editorPhysicsInstallMarker)) {
        g_editorPhysicsRecoveryState.store(4); return;
    }
    g_editorPhysicsInstallAttempted.store(1);
    BasicHook(raycast, HookEditorPhysicsRaycast, g_oldEditorRaycast);
    const bool installed = g_oldEditorRaycast != nullptr;
    g_editorPhysicsHookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_editorPhysicsInstallMarker);
    else g_editorPhysicsRecoveryState.store(5);
}

"""
once('void MaybeInstallEditorObjectsProbe() {\n',physics+'void MaybeInstallEditorObjectsProbe() {\n')
once(
    '    MaybeInstallEditorObjectsProbe();\n',
    '    MaybeInstallEditorObjectsProbe();\n'
    '    MaybeInstallEditorPhysicsSync();\n'
)
once(
    '                                        (g_editorObjectsHookInstalled.load() ? 1 : 0)) << \'\\n\'\n',
    '                                        (g_editorObjectsHookInstalled.load() ? 1 : 0) +\n'
    '                                        (g_editorPhysicsHookInstalled.load() ? 1 : 0)) << \'\\n\'\n'
)
once(
    '        << "editorObjectsProbeRevision=16" << \'\\n\'\n',
    '        << "editorObjectsProbeRevision=16" << \'\\n\'\n'
    '        << "editorPhysicsRevision=17" << \'\\n\'\n'
    '        << "editorPhysicsPolicy=touch-editor-objects-raycast-sync-once-per-frame" << \'\\n\'\n'
    '        << "editorPhysicsOriginalRaycastUnmodified=1" << \'\\n\'\n'
    '        << "editorPhysicsAbiGuard=" << g_editorPhysicsAbiGuard.load() << \'\\n\'\n'
    '        << "editorPhysicsIcallReady=" << g_editorPhysicsIcallReady.load() << \'\\n\'\n'
    '        << "editorPhysicsHookInstalled=" << g_editorPhysicsHookInstalled.load() << \'\\n\'\n'
    '        << "editorPhysicsRecoveryState=" << g_editorPhysicsRecoveryState.load() << \'\\n\'\n'
    '        << "editorPhysicsMarkerReady=" << g_editorPhysicsMarkerReady.load() << \'\\n\'\n'
    '        << "editorPhysicsInstallAttempted=" << g_editorPhysicsInstallAttempted.load() << \'\\n\'\n'
    '        << "editorPhysicsFirstCallProven=" << g_editorPhysicsFirstCallProven.load() << \'\\n\'\n'
    '        << "editorPhysicsOriginalCaptured=" << (g_oldEditorRaycast ? 1 : 0) << \'\\n\'\n'
)
once(
    '        << "editorObjectsCalls=" << g_editorObjectsCalls.load() << \'\\n\'\n',
    '        << "editorObjectsCalls=" << g_editorObjectsCalls.load() << \'\\n\'\n'
    '        << "editorPhysicsRaycastCalls=" << g_editorPhysicsRaycastCalls.load() << \'\\n\'\n'
    '        << "editorPhysicsTouchRaycasts=" << g_editorPhysicsTouchRaycasts.load() << \'\\n\'\n'
    '        << "editorPhysicsSyncCalls=" << g_editorPhysicsSyncCalls.load() << \'\\n\'\n'
    '        << "editorPhysicsSyncUnavailable=" << g_editorPhysicsSyncUnavailable.load() << \'\\n\'\n'
)
assert 'calibrationExecution=disabled-r16-after-r15-self-fuse-recovery' in s
assert 'editorPhysicsPolicy=touch-editor-objects-raycast-sync-once-per-frame' in s
assert s.count('BasicHook(') == 7, s.count('BasicHook(')
path.write_text(s, encoding="utf-8")
