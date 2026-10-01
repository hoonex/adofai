#!/usr/bin/env python3
"""R30: retire the ineffective SyncTransforms mutation and observe exact tile-hit results.

Real-device evidence from runtime 3701b119 proved:
- r29 loaded healthy;
- the exact ObjectsAtMouse and RaycastAll hooks installed and executed;
- Physics2D.SyncTransforms resolved and was called;
- tile selection still failed.

Therefore the SyncTransforms hypothesis is falsified on the affected device.

Exact original v2.4 disassembly of scnEditor.ObjectsAtMouse (RVA 0x22E8DF4) shows:
- RaycastAll callsites 0x22E9400 and 0x22E9454;
- those two RaycastHit2D[] results are combined;
- the combined hits are then filtered into the final GameObject[] returned by ObjectsAtMouse.

R30 keeps the same two already-proven hook sites but restores original raycast behavior:
no coordinate changes, no layer changes, no physics sync, no selection mutation.
It records bounded array counts for each original RaycastAll and for the final
ObjectsAtMouse result, plus the exact raycast arguments, so one device report can
distinguish "physics query is empty" from "post-raycast filtering drops the hit".
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r30-tile-result-observation.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r30 anchor must occur once, got {n}: {old[:220]!r}")
    s = s.replace(old, new, 1)


once(
    'std::atomic<int> g_tileR29ResolveIcallResult{0};\n',
    'std::atomic<int> g_tileR29ResolveIcallResult{0};\n'
    'thread_local int g_tileR30RaycastOrdinal = 0;\n'
    'std::atomic<int> g_tileR30ObjectsNullReturns{0};\n'
    'std::atomic<int> g_tileR30ObjectsNonNullReturns{0};\n'
    'std::atomic<int> g_tileR30ObjectsEmptyReturns{0};\n'
    'std::atomic<int> g_tileR30ObjectsNonEmptyReturns{0};\n'
    'std::atomic<int> g_tileR30ObjectsLastCount{-1};\n'
    'std::atomic<int> g_tileR30ObjectsMaxCount{0};\n'
    'std::atomic<int> g_tileR30RaycastNullReturns{0};\n'
    'std::atomic<int> g_tileR30RaycastEmptyReturns{0};\n'
    'std::atomic<int> g_tileR30RaycastNonEmptyReturns{0};\n'
    'std::atomic<int> g_tileR30FirstRaycastLastCount{-1};\n'
    'std::atomic<int> g_tileR30SecondRaycastLastCount{-1};\n'
    'std::atomic<int> g_tileR30FirstRaycastMaxCount{0};\n'
    'std::atomic<int> g_tileR30SecondRaycastMaxCount{0};\n'
    'std::atomic<int> g_tileR30CountGuardFailures{0};\n'
    'std::atomic<int> g_tileR30FirstOriginX1000{0};\n'
    'std::atomic<int> g_tileR30FirstOriginY1000{0};\n'
    'std::atomic<int> g_tileR30SecondOriginX1000{0};\n'
    'std::atomic<int> g_tileR30SecondOriginY1000{0};\n'
    'std::atomic<int> g_tileR30FirstDirectionX1000{0};\n'
    'std::atomic<int> g_tileR30FirstDirectionY1000{0};\n'
    'std::atomic<int> g_tileR30SecondDirectionX1000{0};\n'
    'std::atomic<int> g_tileR30SecondDirectionY1000{0};\n'
    'std::atomic<int> g_tileR30FirstDistanceX1000{0};\n'
    'std::atomic<int> g_tileR30SecondDistanceX1000{0};\n'
    'std::atomic<int> g_tileR30FirstLayerMask{0};\n'
    'std::atomic<int> g_tileR30SecondLayerMask{0};\n'
)

helper = r'''
constexpr size_t kTileR30MaxObservedArray = 4096;

void UpdateMaxAtomic(std::atomic<int>& target, int value) {
    int old = target.load(std::memory_order_relaxed);
    while (value > old &&
           !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {}
}

int ObserveTileR30ArrayCount(Array<IL2CPP::Il2CppObject*>* result) {
    if (!result) return -1;
    const size_t count = static_cast<size_t>(result->capacity);
    if (count > kTileR30MaxObservedArray) {
        g_tileR30CountGuardFailures.fetch_add(1, std::memory_order_relaxed);
        return -2;
    }
    return static_cast<int>(count);
}

'''
once(
    'EditorObjectsArray* HookTileR21ObjectsAtMouse(IL2CPP::Il2CppObject* self,\n',
    helper + 'EditorObjectsArray* HookTileR21ObjectsAtMouse(IL2CPP::Il2CppObject* self,\n'
)

once(
    '        g_tileR21SyncedThisObjectsCall = false;\n'
    '        g_tileR21ObjectsCalls.fetch_add(1);\n',
    '        g_tileR21SyncedThisObjectsCall = false;\n'
    '        g_tileR30RaycastOrdinal = 0;\n'
    '        g_tileR21ObjectsCalls.fetch_add(1);\n'
)

once(
    '    EditorObjectsArray* result = g_oldTileR21ObjectsAtMouse(self, methodInfo);\n\n'
    '    --g_tileR21ObjectsDepth;\n',
    '    EditorObjectsArray* result = g_oldTileR21ObjectsAtMouse(self, methodInfo);\n\n'
    '    if (outer) {\n'
    '        const int count = ObserveTileR30ArrayCount(result);\n'
    '        g_tileR30ObjectsLastCount.store(count, std::memory_order_relaxed);\n'
    '        if (count == -1) {\n'
    '            g_tileR30ObjectsNullReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '        } else {\n'
    '            g_tileR30ObjectsNonNullReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '            if (count == 0) {\n'
    '                g_tileR30ObjectsEmptyReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '            } else if (count > 0) {\n'
    '                g_tileR30ObjectsNonEmptyReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '                UpdateMaxAtomic(g_tileR30ObjectsMaxCount, count);\n'
    '            }\n'
    '        }\n'
    '    }\n\n'
    '    --g_tileR21ObjectsDepth;\n'
)

once(
    '    g_tileR21RaycastCalls.fetch_add(1);\n'
    '    if (!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms) {\n'
    '        // ObjectsAtMouse itself is the exact ownership boundary. Do not depend on\n'
    '        // Android touchCount: legacy mouse emulation can run this path one frame later.\n'
    '        if (!g_tileR21TouchActive) g_tileR28SyncWithoutTouchCalls.fetch_add(1);\n'
    '        g_tileR21SyncTransforms();\n'
    '        g_tileR21SyncedThisObjectsCall = true;\n'
    '        g_tileR21SyncCalls.fetch_add(1);\n'
    '    }\n\n'
    '    auto* result = g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);\n',
    '    g_tileR21RaycastCalls.fetch_add(1);\n'
    '    const int ordinal = ++g_tileR30RaycastOrdinal;\n'
    '    if (ordinal == 1) {\n'
    '        g_tileR30FirstOriginX1000.store(static_cast<int>(origin.x * 1000.0f));\n'
    '        g_tileR30FirstOriginY1000.store(static_cast<int>(origin.y * 1000.0f));\n'
    '        g_tileR30FirstDirectionX1000.store(static_cast<int>(direction.x * 1000.0f));\n'
    '        g_tileR30FirstDirectionY1000.store(static_cast<int>(direction.y * 1000.0f));\n'
    '        g_tileR30FirstDistanceX1000.store(static_cast<int>(distance * 1000.0f));\n'
    '        g_tileR30FirstLayerMask.store(layerMask);\n'
    '    } else if (ordinal == 2) {\n'
    '        g_tileR30SecondOriginX1000.store(static_cast<int>(origin.x * 1000.0f));\n'
    '        g_tileR30SecondOriginY1000.store(static_cast<int>(origin.y * 1000.0f));\n'
    '        g_tileR30SecondDirectionX1000.store(static_cast<int>(direction.x * 1000.0f));\n'
    '        g_tileR30SecondDirectionY1000.store(static_cast<int>(direction.y * 1000.0f));\n'
    '        g_tileR30SecondDistanceX1000.store(static_cast<int>(distance * 1000.0f));\n'
    '        g_tileR30SecondLayerMask.store(layerMask);\n'
    '    }\n\n'
    '    // R29 proved SyncTransforms resolves and executes, but the affected phone still\n'
    '    // cannot select tiles. Restore the original query path and observe only.\n'
    '    auto* result = g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);\n'
    '    const int count = ObserveTileR30ArrayCount(result);\n'
    '    if (count == -1) {\n'
    '        g_tileR30RaycastNullReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '    } else if (count == 0) {\n'
    '        g_tileR30RaycastEmptyReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '    } else if (count > 0) {\n'
    '        g_tileR30RaycastNonEmptyReturns.fetch_add(1, std::memory_order_relaxed);\n'
    '    }\n'
    '    if (ordinal == 1) {\n'
    '        g_tileR30FirstRaycastLastCount.store(count, std::memory_order_relaxed);\n'
    '        if (count > 0) UpdateMaxAtomic(g_tileR30FirstRaycastMaxCount, count);\n'
    '    } else if (ordinal == 2) {\n'
    '        g_tileR30SecondRaycastLastCount.store(count, std::memory_order_relaxed);\n'
    '        if (count > 0) UpdateMaxAtomic(g_tileR30SecondRaycastMaxCount, count);\n'
    '    }\n'
)

once(
    '    const bool abi = objectsAbi && raycastAbi && sync;\n',
    '    const bool abi = objectsAbi && raycastAbi;\n'
)

once(
    '    const bool installed = g_oldTileR21ObjectsAtMouse != nullptr &&\n'
    '            g_oldTileR21Raycast != nullptr && g_tileR21SyncTransforms != nullptr;\n',
    '    const bool installed = g_oldTileR21ObjectsAtMouse != nullptr &&\n'
    '            g_oldTileR21Raycast != nullptr;\n'
)

once(
    'activeTilePolicy=r28-objects-scope-exact-raycast-sync-plus-full-width-window',
    'activeTilePolicy=r30-read-only-original-raycast-and-objects-results'
)
once(
    'tileR21Policy=ObjectsAtMouse-all-calls-SyncTransforms-before-original-RayCastAll',
    'tileR21Policy=ObjectsAtMouse-original-results-observe-only-r30'
)

once(
    '        << "tileR29ResolveIcallResult=" << g_tileR29ResolveIcallResult.load() << \'\\n\'\n',
    '        << "tileR29ResolveIcallResult=" << g_tileR29ResolveIcallResult.load() << \'\\n\'\n'
    '        << "tileResultRevision=30" << \'\\n\'\n'
    '        << "tileR30Policy=read-only-two-raycasts-plus-final-objects-result" << \'\\n\'\n'
    '        << "tileR30Mutation=0" << \'\\n\'\n'
    '        << "tileR30SyncRetired=1" << \'\\n\'\n'
    '        << "tileR30ObjectsNullReturns=" << g_tileR30ObjectsNullReturns.load() << \'\\n\'\n'
    '        << "tileR30ObjectsNonNullReturns=" << g_tileR30ObjectsNonNullReturns.load() << \'\\n\'\n'
    '        << "tileR30ObjectsEmptyReturns=" << g_tileR30ObjectsEmptyReturns.load() << \'\\n\'\n'
    '        << "tileR30ObjectsNonEmptyReturns=" << g_tileR30ObjectsNonEmptyReturns.load() << \'\\n\'\n'
    '        << "tileR30ObjectsLastCount=" << g_tileR30ObjectsLastCount.load() << \'\\n\'\n'
    '        << "tileR30ObjectsMaxCount=" << g_tileR30ObjectsMaxCount.load() << \'\\n\'\n'
    '        << "tileR30RaycastNullReturns=" << g_tileR30RaycastNullReturns.load() << \'\\n\'\n'
    '        << "tileR30RaycastEmptyReturns=" << g_tileR30RaycastEmptyReturns.load() << \'\\n\'\n'
    '        << "tileR30RaycastNonEmptyReturns=" << g_tileR30RaycastNonEmptyReturns.load() << \'\\n\'\n'
    '        << "tileR30FirstRaycastLastCount=" << g_tileR30FirstRaycastLastCount.load() << \'\\n\'\n'
    '        << "tileR30SecondRaycastLastCount=" << g_tileR30SecondRaycastLastCount.load() << \'\\n\'\n'
    '        << "tileR30FirstRaycastMaxCount=" << g_tileR30FirstRaycastMaxCount.load() << \'\\n\'\n'
    '        << "tileR30SecondRaycastMaxCount=" << g_tileR30SecondRaycastMaxCount.load() << \'\\n\'\n'
    '        << "tileR30CountGuardFailures=" << g_tileR30CountGuardFailures.load() << \'\\n\'\n'
    '        << "tileR30FirstOriginX1000=" << g_tileR30FirstOriginX1000.load() << \'\\n\'\n'
    '        << "tileR30FirstOriginY1000=" << g_tileR30FirstOriginY1000.load() << \'\\n\'\n'
    '        << "tileR30SecondOriginX1000=" << g_tileR30SecondOriginX1000.load() << \'\\n\'\n'
    '        << "tileR30SecondOriginY1000=" << g_tileR30SecondOriginY1000.load() << \'\\n\'\n'
    '        << "tileR30FirstDirectionX1000=" << g_tileR30FirstDirectionX1000.load() << \'\\n\'\n'
    '        << "tileR30FirstDirectionY1000=" << g_tileR30FirstDirectionY1000.load() << \'\\n\'\n'
    '        << "tileR30SecondDirectionX1000=" << g_tileR30SecondDirectionX1000.load() << \'\\n\'\n'
    '        << "tileR30SecondDirectionY1000=" << g_tileR30SecondDirectionY1000.load() << \'\\n\'\n'
    '        << "tileR30FirstDistanceX1000=" << g_tileR30FirstDistanceX1000.load() << \'\\n\'\n'
    '        << "tileR30SecondDistanceX1000=" << g_tileR30SecondDistanceX1000.load() << \'\\n\'\n'
    '        << "tileR30FirstLayerMask=" << g_tileR30FirstLayerMask.load() << \'\\n\'\n'
    '        << "tileR30SecondLayerMask=" << g_tileR30SecondLayerMask.load() << \'\\n\'\n'
)

for marker in (
    "tileResultRevision=30",
    "tileR30Policy=read-only-two-raycasts-plus-final-objects-result",
    "tileR30Mutation=0",
    "tileR30SyncRetired=1",
    "tileR30ObjectsLastCount=",
    "tileR30FirstRaycastLastCount=",
    "tileR30SecondRaycastLastCount=",
    "tileR30FirstOriginX1000=",
    "tileR30SecondOriginX1000=",
    "tileR30FirstLayerMask=",
    "tileR30SecondLayerMask=",
    "activeTilePolicy=r30-read-only-original-raycast-and-objects-results",
    "tileR21Policy=ObjectsAtMouse-original-results-observe-only-r30",
):
    if marker not in s:
        raise SystemExit(f"r30 marker missing: {marker}")

if "g_tileR21SyncTransforms();" in s:
    raise SystemExit("r30 must retire the active SyncTransforms mutation")

if s.count("BasicHook(") != 18:
    raise SystemExit(f"r30 adds no hooks; expected 18 compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r31 = Path(__file__).with_name("apply-v240-r31-editor-transaction-trace.py")
if not r31.is_file():
    raise SystemExit(f"missing r31 overlay: {r31}")
__import__("subprocess").run([sys.executable, str(r31), str(path)], check=True)
