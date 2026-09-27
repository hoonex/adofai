#!/usr/bin/env python3
"""R23: resolve the exact Physics2D.RaycastAll overload used by v2.4 editor tile picking.

R21/R22 already narrowed the tile repair to touch-driven scnEditor.ObjectsAtMouse and preserved
all original raycast arguments/results. The remaining ambiguity was method lookup by name plus
parameter count. R23 removes that ambiguity: BNM resolves the exact
RaycastAll(Vector2, Vector2, float, int) overload and additionally requires RaycastHit2D[] return.

No new hooks or behavior changes are introduced; this only makes the existing collider-sync repair
fail closed on the exact managed ABI proven by the authoritative v2.4 binary.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r23-exact-raycast-resolution.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r23 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)


once(
    '    MethodBase objects = editor.GetMethod("ObjectsAtMouse", 0);\n'
    '    MethodBase raycast = physics.GetMethod("RaycastAll", 4);\n',
    '    MethodBase objects = editor.GetMethod("ObjectsAtMouse", 0);\n'
    '    MethodBase raycast = physics.GetMethod("RaycastAll", {\n'
    '            vector2.GetCompileTimeClass(), vector2.GetCompileTimeClass(),\n'
    '            floatClass.GetCompileTimeClass(), intClass.GetCompileTimeClass()});\n'
)

once(
    '            raycast._isStatic && raycastParams && raycastInfo->return_type &&\n'
    '            TypeCode(raycastInfo->return_type) == 29 &&\n',
    '            raycast._isStatic && raycastParams && raycastInfo->return_type &&\n'
    '            TypeCode(raycastInfo->return_type) == 29 &&\n'
    '            MetadataTypeName(raycastInfo->return_type) == "UnityEngine.RaycastHit2D[]" &&\n'
)

once('        << "stabilityRevision=22" << \'\\n\'\n',
     '        << "stabilityRevision=23" << \'\\n\'\n')

once(
    'activeTilePolicy=r21-transient-collider-sync-plus-full-width-window',
    'activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window'
)

once(
    '        << "tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll" << \'\\n\'\n',
    '        << "tileR21Policy=ObjectsAtMouse-touch-SyncTransforms-before-original-RayCastAll" << \'\\n\'\n'
    '        << "tileR23Resolution=RaycastAll-Vector2-Vector2-float-int-exact" << \'\\n\'\n'
    '        << "tileR23ReturnType=UnityEngine.RaycastHit2D[]" << \'\\n\'\n'
    '        << "tileR23MutationDelta=0" << \'\\n\'\n'
)

for marker in (
    'stabilityRevision=23',
    'activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window',
    'vector2.GetCompileTimeClass(), vector2.GetCompileTimeClass()',
    'floatClass.GetCompileTimeClass(), intClass.GetCompileTimeClass()',
    'MetadataTypeName(raycastInfo->return_type) == "UnityEngine.RaycastHit2D[]"',
    'tileR23Resolution=RaycastAll-Vector2-Vector2-float-int-exact',
    'tileR23ReturnType=UnityEngine.RaycastHit2D[]',
    'tileR23MutationDelta=0',
    'MaybeInstallCalibrationR22();',
    'MaybeInstallTileR21();',
):
    if marker not in s:
        raise SystemExit(f"r23 marker missing after transform: {marker}")

if s.count('BasicHook(') != 13:
    raise SystemExit(f"r23 adds no hooks; expected thirteen compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
