#!/usr/bin/env python3
# R24: final diagnostic integrity only.
# Current active hook sites: SFB(1), calibration r22(1), tile r21/r23(2).
# No game behavior changes are introduced here.
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r24-diagnostic-integrity.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r24 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)

old_count = """        << "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +
                                        (g_editorProbeInstalled.load() ? 2 : 0) +
                                        (g_editorObjectsHookInstalled.load() ? 1 : 0) +
                                        (g_editorPhysicsHookInstalled.load() ? 1 : 0)) << '\\n'
"""
new_count = """        << "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +
                                        (g_calibrationR22HookInstalled.load() ? 1 : 0) +
                                        (g_tileR21HookInstalled.load() ? 2 : 0)) << '\\n'
        << "diagnosticRevision=24\\n"
        << "activeHookPolicy=sfb-1-calibrationR22-1-tileR21-2\\n"
        << "gameHooksInstalledSemantics=active-installed-hook-sites\\n"
        << "historicalInactiveHookSitesExcluded=1\\n"
"""
once(old_count, new_count)

active = """void ReconcileInstallState() {
    CollectV240MetadataInventory();
    MaybeInstallCalibrationR22();
    MaybeInstallTileR21();
    MaybeInstallSfbHook();
    BuildStaticReport();
}
"""
if active not in s:
    raise SystemExit("r24 final active reconciliation contract changed unexpectedly")

for marker in (
    "stabilityRevision=23",
    "diagnosticRevision=24",
    "activeHookPolicy=sfb-1-calibrationR22-1-tileR21-2",
    "gameHooksInstalledSemantics=active-installed-hook-sites",
    "historicalInactiveHookSitesExcluded=1",
    "(g_calibrationR22HookInstalled.load() ? 1 : 0)",
    "(g_tileR21HookInstalled.load() ? 2 : 0)",
    "activeTilePolicy=r23-exact-raycast-transient-collider-sync-plus-full-width-window",
    "calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced",
):
    if marker not in s:
        raise SystemExit(f"r24 marker missing after transform: {marker}")

report = s[s.index("void BuildStaticReport()"):s.index("std::string CurrentReport")]
for stale in (
    "(g_editorProbeInstalled.load() ? 2 : 0) +",
    "(g_editorObjectsHookInstalled.load() ? 1 : 0) +",
    "(g_editorPhysicsHookInstalled.load() ? 1 : 0)) <<",
):
    if stale in report:
        raise SystemExit(f"r24 stale top-level hook counter survived: {stale}")

if s.count("BasicHook(") != 13:
    raise SystemExit(f"r24 must add no hooks; expected thirteen compiled sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r26 = Path(__file__).with_name("apply-v240-r26-sfb-save-folder.py")
if not r26.is_file():
    raise SystemExit(f"missing r26 overlay: {r26}")
__import__("subprocess").run([sys.executable, str(r26), str(path)], check=True)
