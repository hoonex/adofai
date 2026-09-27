#!/usr/bin/env python3
"""R22: persist calibration at the actual mutation boundary; keep exact tile physics repair.

Authoritative v2.4 binary evidence:
- scrCalibrationPlanet.PostSong writes currentPreset.inputOffset then calls
  scrConductor.SaveCurrentPreset at RVA 0x218576C.
- SaveCurrentPreset only updates scrConductor.userPresets in memory. It never calls
  Persistence.Save / WriteSaveToDisk.
- Persistence.WriteSaveToDisk (RVA 0x1169E64) is reached through Persistence.Save
  (RVA 0x115F748) -> SaveCo(0.5s), and serializes userPresets using
  CalibrationPreset.ToDict before PlayerPrefsJson.SetList/SaveAllFiles.
- Persistence.Load explicitly writes confident=true into the 32-byte preset at +24
  before calling CalibrationPreset.FromDict. Therefore r21's FromDict confidence
  mutation is redundant and is not the startup-calibration root cause.

R22 hooks the exact zero-argument SaveCurrentPreset. It calls the original first,
then invokes the game's own debounced Persistence.Save. No offset, output identity,
preset contents, or calibration confidence bytes are changed.

The r21 tile fix remains active: it only synchronizes Physics2D transforms immediately
before the first original RaycastAll inside touch-driven scnEditor.ObjectsAtMouse.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r22-calibration-persist.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r22 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)


once(
    'std::string g_tileR21RaycastCallMarker;\n',
    'std::string g_tileR21RaycastCallMarker;\n'
    'using CalibrationSaveCurrentPresetFn = void (*)(IL2CPP::MethodInfo*);\n'
    'CalibrationSaveCurrentPresetFn g_oldCalibrationR22SaveCurrentPreset = nullptr;\n'
    'Method<void> g_calibrationR22PersistenceSave;\n'
    'std::atomic<int> g_calibrationR22AbiGuard{0};\n'
    'std::atomic<int> g_calibrationR22HookInstalled{0};\n'
    'std::atomic<int> g_calibrationR22InstallAttempted{0};\n'
    'std::atomic<int> g_calibrationR22MarkerReady{0};\n'
    'std::atomic<int> g_calibrationR22RecoveryState{0};\n'
    'std::atomic<int> g_calibrationR22FirstCallProven{0};\n'
    'std::atomic<int> g_calibrationR22SaveCurrentPresetCalls{0};\n'
    'std::atomic<int> g_calibrationR22PersistenceSaveRequests{0};\n'
    'std::mutex g_calibrationR22FirstCallMutex;\n'
    'std::string g_calibrationR22InstallMarker;\n'
    'std::string g_calibrationR22CallMarker;\n'
)

r22_code = r'''
bool PrepareCalibrationR22Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_calibrationR22RecoveryState.store(4);
        return false;
    }
    g_calibrationR22InstallMarker = dir + "/calibration-r22-persist-install.pending";
    g_calibrationR22CallMarker = dir + "/calibration-r22-persist-call.pending";
    g_calibrationR22MarkerReady.store(1);
    if (MarkerExists(g_calibrationR22InstallMarker)) {
        g_calibrationR22RecoveryState.store(1);
        return false;
    }
    if (MarkerExists(g_calibrationR22CallMarker)) {
        g_calibrationR22RecoveryState.store(2);
        return false;
    }
    const std::string probe = dir + "/calibration-r22-persist-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_calibrationR22MarkerReady.store(0);
        g_calibrationR22RecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void HookCalibrationR22SaveCurrentPreset(IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldCalibrationR22SaveCurrentPreset) return;

    bool first = false;
    if (!g_calibrationR22FirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_calibrationR22FirstCallMutex);
        if (!g_calibrationR22FirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_calibrationR22CallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldCalibrationR22SaveCurrentPreset(methodInfo);
                return;
            }
            first = true;
        }
    }

    // Preserve the exact preset update first. Persistence.Save is the game's existing
    // debounce path (SaveCo 0.5s), so repeated manual offset changes collapse into one write.
    g_oldCalibrationR22SaveCurrentPreset(methodInfo);
    g_calibrationR22SaveCurrentPresetCalls.fetch_add(1);

    if (g_calibrationR22PersistenceSave.IsValid()) {
        g_calibrationR22PersistenceSave.Call();
        g_calibrationR22PersistenceSaveRequests.fetch_add(1);
    }

    if (first) {
        ClearMarker(g_calibrationR22CallMarker);
        g_calibrationR22FirstCallProven.store(1, std::memory_order_release);
    }
}

void MaybeInstallCalibrationR22() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_calibrationR22HookInstalled.load() ||
        g_calibrationR22InstallAttempted.load() ||
        g_calibrationR22RecoveryState.load() != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_calibrationR22HookInstalled.load() ||
        g_calibrationR22InstallAttempted.load()) return;

    Class conductor("", "scrConductor");
    Class persistence("", "Persistence");
    if (!conductor || !persistence) return;

    MethodBase saveCurrentPreset = conductor.GetMethod("SaveCurrentPreset", 0);
    MethodBase persistenceSave = persistence.GetMethod("Save", 0);
    IL2CPP::MethodInfo* currentInfo =
            saveCurrentPreset.IsValid() ? saveCurrentPreset.GetInfo() : nullptr;
    IL2CPP::MethodInfo* persistenceInfo =
            persistenceSave.IsValid() ? persistenceSave.GetInfo() : nullptr;

    const bool currentAbi = currentInfo && currentInfo->methodPointer &&
            saveCurrentPreset._isStatic && currentInfo->parameters_count == 0 &&
            currentInfo->return_type && TypeCode(currentInfo->return_type) == 1;
    const bool persistenceAbi = persistenceInfo && persistenceInfo->methodPointer &&
            persistenceSave._isStatic && persistenceInfo->parameters_count == 0 &&
            persistenceInfo->return_type && TypeCode(persistenceInfo->return_type) == 1;
    const bool abi = currentAbi && persistenceAbi;
    g_calibrationR22AbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareCalibrationR22Fuse()) return;

    if (!WriteMarker(g_calibrationR22InstallMarker)) {
        g_calibrationR22MarkerReady.store(0);
        g_calibrationR22RecoveryState.store(4);
        return;
    }

    g_calibrationR22InstallAttempted.store(1);
    g_calibrationR22PersistenceSave = Method<void>(persistenceSave);
    BasicHook(saveCurrentPreset, HookCalibrationR22SaveCurrentPreset,
              g_oldCalibrationR22SaveCurrentPreset);

    const bool installed = g_oldCalibrationR22SaveCurrentPreset != nullptr &&
            g_calibrationR22PersistenceSave.IsValid();
    g_calibrationR22HookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_calibrationR22InstallMarker);
    else g_calibrationR22RecoveryState.store(5);
}

'''

once('constexpr BNM_PTR kCalibrationR21ConfidentOffset = 24;\n',
     r22_code + 'constexpr BNM_PTR kCalibrationR21ConfidentOffset = 24;\n')

# r21's confidence mutation is disproven by exact Persistence.Load disassembly:
# it writes true to +24 before FromDict. Activate only persistence scheduling plus tile sync.
once(
    '    MaybeInstallCalibrationR21();\n'
    '    MaybeInstallTileR21();\n'
    '    MaybeInstallSfbHook();\n',
    '    MaybeInstallCalibrationR22();\n'
    '    MaybeInstallTileR21();\n'
    '    MaybeInstallSfbHook();\n'
)

once('        << "stabilityRevision=21" << \'\\n\'\n',
     '        << "stabilityRevision=22" << \'\\n\'\n')

once(
    'calibrationR21Policy=persisted-CalibrationPreset-FromDict-restore-confident',
    'calibrationR21Policy=disabled-r22-PersistenceLoad-already-restores-confident'
)
once(
    'calibrationR21Mutation=memory-only-persisted-preset-confidence-no-offset-write',
    'calibrationR21Mutation=disabled-r22-forensic-only'
)

once(
    '        << "calibrationR21AlreadyConfident=" << g_calibrationR21AlreadyConfident.load() << \'\\n\'\n',
    '        << "calibrationR21AlreadyConfident=" << g_calibrationR21AlreadyConfident.load() << \'\\n\'\n'
    '        << "calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced" << \'\\n\'\n'
    '        << "calibrationR22Source=authoritative-v240-binary-callgraph" << \'\\n\'\n'
    '        << "calibrationR22Mutation=persistence-schedule-only-no-preset-value-change" << \'\\n\'\n'
    '        << "calibrationR22SaveCurrentPresetRva=0x218576C" << \'\\n\'\n'
    '        << "calibrationR22PersistenceSaveRva=0x115F748" << \'\\n\'\n'
    '        << "calibrationR22WriteSaveToDiskRva=0x1169E64" << \'\\n\'\n'
    '        << "calibrationR22PersistenceLoadConfidentStoreRva=0x1169674" << \'\\n\'\n'
    '        << "calibrationR22PersistenceLoadFromDictRva=0x11696B8" << \'\\n\'\n'
    '        << "calibrationR22AbiGuard=" << g_calibrationR22AbiGuard.load() << \'\\n\'\n'
    '        << "calibrationR22HookInstalled=" << g_calibrationR22HookInstalled.load() << \'\\n\'\n'
    '        << "calibrationR22InstallAttempted=" << g_calibrationR22InstallAttempted.load() << \'\\n\'\n'
    '        << "calibrationR22MarkerReady=" << g_calibrationR22MarkerReady.load() << \'\\n\'\n'
    '        << "calibrationR22RecoveryState=" << g_calibrationR22RecoveryState.load() << \'\\n\'\n'
    '        << "calibrationR22FirstCallProven=" << g_calibrationR22FirstCallProven.load() << \'\\n\'\n'
    '        << "calibrationR22SaveCurrentPresetCalls=" << g_calibrationR22SaveCurrentPresetCalls.load() << \'\\n\'\n'
    '        << "calibrationR22PersistenceSaveRequests=" << g_calibrationR22PersistenceSaveRequests.load() << \'\\n\'\n'
)

for marker in (
    'stabilityRevision=22',
    'calibrationR21Policy=disabled-r22-PersistenceLoad-already-restores-confident',
    'calibrationR22Policy=SaveCurrentPreset-then-Persistence.Save-debounced',
    'calibrationR22SaveCurrentPresetRva=0x218576C',
    'calibrationR22PersistenceSaveRva=0x115F748',
    'calibrationR22WriteSaveToDiskRva=0x1169E64',
    'calibrationR22PersistenceLoadConfidentStoreRva=0x1169674',
    'calibrationR22PersistenceLoadFromDictRva=0x11696B8',
    'conductor.GetMethod("SaveCurrentPreset", 0)',
    'persistence.GetMethod("Save", 0)',
    'g_calibrationR22PersistenceSave.Call();',
    'BasicHook(saveCurrentPreset, HookCalibrationR22SaveCurrentPreset',
    'calibration-r22-persist-install.pending',
    'calibration-r22-persist-call.pending',
    'MaybeInstallCalibrationR22();',
    'MaybeInstallTileR21();',
):
    if marker not in s:
        raise SystemExit(f"r22 marker missing after transform: {marker}")

if '    MaybeInstallCalibrationR21();\n' in s:
    raise SystemExit("r21 redundant confidence hook must not remain active in r22")
if '    MaybeRepairCalibrationR20();\n' in s:
    raise SystemExit("r20 symptom write must remain inactive in r22")
if s.count('BasicHook(') != 13:
    raise SystemExit(f"r22 expected thirteen compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")


r23 = Path(__file__).with_name("apply-v240-r23-exact-raycast-resolution.py")
if not r23.is_file():
    raise SystemExit(f"missing r23 overlay: {r23}")
__import__("subprocess").run([sys.executable, str(r23), str(path)], check=True)
