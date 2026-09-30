#!/usr/bin/env python3
"""R28: make the exact v2.4 transient-collider synchronization independent of Android touch timing.

Authoritative v2.4 binary audit:
- scnEditor.ObjectsAtMouse is RVA 0x22E8DF4.
- It calls FloorMesh.GenerateCollider at 0x22E91BC, enables the generated behaviour at
  0x22E91D0, and calls scrFloor.GenerateCollider at 0x22E9268.
- The same method then immediately calls the exact
  Physics2D.RaycastAll(Vector2, Vector2, float, int) (RVA 0x1B9FF44) at 0x22E9400
  and again at 0x22E9454. There is no intervening collider mutation between those two queries.
- R21/R23 correctly synchronized immediately before the first original raycast, but
  unnecessarily gated that repair on Input.touchCount > 0. Legacy mouse emulation can reach
  ObjectsAtMouse on a frame where touchCount is already zero, reintroducing device/timing
  dependence even though the ownership boundary itself is exact.

R28 keeps the same two exact hooks and every original raycast argument/result. It removes only
the touchCount correctness gate: while execution is inside ObjectsAtMouse, the first exact
RaycastAll gets one Physics2D.SyncTransforms call. That same synchronized physics state covers
the second original query because no collider state changes between them. Calls outside
ObjectsAtMouse remain original.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r28-objects-scope-tile-sync.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r28 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)


once(
    'std::atomic<int> g_tileR21SyncCalls{0};\n',
    'std::atomic<int> g_tileR21SyncCalls{0};\n'
    'std::atomic<int> g_tileR28SyncWithoutTouchCalls{0};\n'
)

once(
    '    if (g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive) {\n'
    '        return g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);\n'
    '    }\n',
    '    if (g_tileR21ObjectsDepth <= 0) {\n'
    '        return g_oldTileR21Raycast(origin, direction, distance, layerMask, methodInfo);\n'
    '    }\n'
)

once(
    '    if (!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms) {\n'
    '        // Synchronize exactly after ObjectsAtMouse created/enabled its transient floor\n'
    '        // colliders and immediately before its first original raycast.\n'
    '        g_tileR21SyncTransforms();\n'
    '        g_tileR21SyncedThisObjectsCall = true;\n'
    '        g_tileR21SyncCalls.fetch_add(1);\n'
    '    }\n',
    '    if (!g_tileR21SyncedThisObjectsCall && g_tileR21SyncTransforms) {\n'
    '        // ObjectsAtMouse itself is the exact ownership boundary. Do not depend on\n'
    '        // Android touchCount: legacy mouse emulation can run this path one frame later.\n'
    '        if (!g_tileR21TouchActive) g_tileR28SyncWithoutTouchCalls.fetch_add(1);\n'
    '        g_tileR21SyncTransforms();\n'
    '        g_tileR21SyncedThisObjectsCall = true;\n'
    '        g_tileR21SyncCalls.fetch_add(1);\n'
    '    }\n'
)

once(
    '    if (!editor || !physics || !input || !vector2 || !floatClass || !intClass) return;\n',
    '    if (!editor || !physics || !vector2 || !floatClass || !intClass) return;\n'
)

once(
    '    g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);\n'
    '    const bool touchAbi = g_tileR21GetTouchCount.IsValid();\n',
    '    if (input) g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);\n'
    '    const bool touchTelemetryAvailable = g_tileR21GetTouchCount.IsValid();\n'
)

once(
    '    const bool abi = objectsAbi && raycastAbi && touchAbi && sync;\n',
    '    const bool abi = objectsAbi && raycastAbi && sync;\n'
)

once(
    'activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window',
    'activeTilePolicy=r28-objects-scope-exact-raycast-sync-plus-full-width-window'
)

once(
    'tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll',
    'tileR21Policy=ObjectsAtMouse-all-calls-SyncTransforms-before-original-RayCastAll'
)

once(
    '        << "tileR21SyncCalls=" << g_tileR21SyncCalls.load() << \'\\n\'\n',
    '        << "tileR21SyncCalls=" << g_tileR21SyncCalls.load() << \'\\n\'\n'
    '        << "tileRepairRevision=28" << \'\\n\'\n'
    '        << "tileR28OwnershipBoundary=ObjectsAtMouse" << \'\\n\'\n'
    '        << "tileR28TouchGateRemoved=1" << \'\\n\'\n'
    '        << "tileR28TouchTelemetryAvailable=" << (g_tileR21GetTouchCount.IsValid() ? 1 : 0) << \'\\n\'\n'
    '        << "tileR28SyncWithoutTouchCalls=" << g_tileR28SyncWithoutTouchCalls.load() << \'\\n\'\n'
)

for marker in (
    "tileRepairRevision=28",
    "activeTilePolicy=r28-objects-scope-exact-raycast-sync-plus-full-width-window",
    "tileR21Policy=ObjectsAtMouse-all-calls-SyncTransforms-before-original-RayCastAll",
    "tileR28OwnershipBoundary=ObjectsAtMouse",
    "tileR28TouchGateRemoved=1",
    "tileR28SyncWithoutTouchCalls=",
    "const bool abi = objectsAbi && raycastAbi && sync;",
    "if (g_tileR21ObjectsDepth <= 0) {",
):
    if marker not in s:
        raise SystemExit(f"r28 marker missing: {marker}")

for forbidden in (
    "g_tileR21ObjectsDepth <= 0 || !g_tileR21TouchActive",
    "const bool abi = objectsAbi && raycastAbi && touchAbi && sync;",
):
    if forbidden in s:
        raise SystemExit(f"r28 obsolete touch gate survived: {forbidden}")

if s.count("BasicHook(") != 18:
    raise SystemExit(f"r28 adds no hooks; expected 18 compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r29 = Path(__file__).with_name("apply-v240-r29-il2cpp-resolver-fallback.py")
if not r29.is_file():
    raise SystemExit(f"missing r29 overlay: {r29}")
__import__("subprocess").run([sys.executable, str(r29), str(path)], check=True)
