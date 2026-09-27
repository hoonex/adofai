#!/usr/bin/env python3
"""R20: persist only the exact unconfigured v2.4 calibration sentinel through the game's real backend.

Authoritative target analysis:
- Persistence.GetInputOffset() returns the game's input offset and uses the general PlayerPrefsJson.
- Persistence.SetInputOffset(float) writes the same logical setting into that PlayerPrefsJson.
- Persistence.get_generalPrefs() returns the exact PlayerPrefsJson instance that owns the setting.
- PlayerPrefsJson.Save() is the durable save operation for this backend.
- 999.0f is the exact unconfigured sentinel in the target APK.

R19 incorrectly targeted UnityEngine.PlayerPrefs. R20 leaves that code compiled only as historical
forensic evidence, removes its active call, and uses the exact Persistence -> PlayerPrefsJson path.
No existing calibrated value is overwritten.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r20-playerprefsjson-calibration.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"r20 anchor must occur once, got {count}: {old[:180]!r}")
    s = s.replace(old, new, 1)

once(
    'std::string g_calibrationR19Marker;\n',
    'std::string g_calibrationR19Marker;\n'
    'std::atomic<int> g_calibrationR20AbiGuard{0};\n'
    'std::atomic<int> g_calibrationR20Attempted{0};\n'
    'std::atomic<int> g_calibrationR20PrefsReady{0};\n'
    'std::atomic<int> g_calibrationR20UnsetDetected{0};\n'
    'std::atomic<int> g_calibrationR20Repaired{0};\n'
    'std::atomic<int> g_calibrationR20SaveCalled{0};\n'
    'std::atomic<int> g_calibrationR20MarkerReady{0};\n'
    'std::atomic<int> g_calibrationR20RecoveryState{0};\n'
    'std::atomic<int> g_calibrationR20BeforeX100{0};\n'
    'std::atomic<int> g_calibrationR20AfterX100{0};\n'
    'std::string g_calibrationR20Marker;\n'
)

code = r'''
constexpr float kCalibrationR20Unset = 999.0f;
constexpr float kCalibrationR20Neutral = 0.0f;

bool PrepareCalibrationR20Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_calibrationR20RecoveryState.store(3);
        return false;
    }
    g_calibrationR20Marker = dir + "/calibration-r20-playerprefsjson-write.pending";
    g_calibrationR20MarkerReady.store(1);
    if (MarkerExists(g_calibrationR20Marker)) {
        g_calibrationR20RecoveryState.store(1);
        return false;
    }
    const std::string probe = dir + "/calibration-r20-playerprefsjson-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_calibrationR20MarkerReady.store(0);
        g_calibrationR20RecoveryState.store(3);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void MaybeRepairCalibrationR20() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_calibrationR20Attempted.load(std::memory_order_acquire) ||
        g_calibrationR20RecoveryState.load() != 0) return;

    Class persistence("", "Persistence");
    Class prefsClass("", "PlayerPrefsJson");
    Class floatClass = Defaults::Get<float>();
    if (!persistence || !prefsClass || !floatClass) return;

    MethodBase getBase = persistence.GetMethod("GetInputOffset", 0);
    MethodBase setBase = persistence.GetMethod("SetInputOffset", 1);
    MethodBase generalBase = persistence.GetMethod("get_generalPrefs", 0);
    MethodBase saveBase = prefsClass.GetMethod("Save", 0);

    IL2CPP::MethodInfo* getInfo = getBase.IsValid() ? getBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* setInfo = setBase.IsValid() ? setBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* generalInfo = generalBase.IsValid() ? generalBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* saveInfo = saveBase.IsValid() ? saveBase.GetInfo() : nullptr;

    const bool setParam = setInfo && setInfo->parameters_count == 1 &&
            setInfo->parameters && setInfo->parameters[0];
    const bool abi =
            getInfo && getInfo->methodPointer && getBase._isStatic &&
            getInfo->parameters_count == 0 && getInfo->return_type &&
            SameClass(Class(getInfo->return_type), floatClass) &&
            setInfo && setInfo->methodPointer && setBase._isStatic && setParam &&
            setInfo->return_type && TypeCode(setInfo->return_type) == 1 &&
            SameClass(Class(setInfo->parameters[0]), floatClass) &&
            TypeByRef(setInfo->parameters[0]) == 0 &&
            generalInfo && generalInfo->methodPointer && generalBase._isStatic &&
            generalInfo->parameters_count == 0 && generalInfo->return_type &&
            SameClass(Class(generalInfo->return_type), prefsClass) &&
            saveInfo && saveInfo->methodPointer && !saveBase._isStatic &&
            saveInfo->parameters_count == 0 && saveInfo->return_type &&
            TypeCode(saveInfo->return_type) == 1;
    g_calibrationR20AbiGuard.store(abi ? 1 : 0);
    if (!abi) return;

    // Startup can reach the hot runtime before Persistence has selected its general save file.
    // Do not burn the one-shot attempt in that state; later automatic reconciles may retry.
    Method<IL2CPP::Il2CppObject*> getGeneral(generalBase);
    IL2CPP::Il2CppObject* prefs = getGeneral.Call();
    if (!prefs) return;
    g_calibrationR20PrefsReady.store(1);

    Method<float> getOffset(getBase);
    Method<void> setOffset(setBase);
    Method<void> savePrefs(saveBase);

    const float before = getOffset.Call();
    g_calibrationR20BeforeX100.store(static_cast<int>(before * 100.0f));
    g_calibrationR20AfterX100.store(static_cast<int>(before * 100.0f));
    g_calibrationR20Attempted.store(1, std::memory_order_release);

    // Every actual calibration value is preserved. Only the game's exact "not configured"
    // value is converted to a neutral zero offset.
    if (before != kCalibrationR20Unset) return;
    g_calibrationR20UnsetDetected.store(1);

    if (!PrepareCalibrationR20Fuse()) return;
    if (!WriteMarker(g_calibrationR20Marker)) {
        g_calibrationR20MarkerReady.store(0);
        g_calibrationR20RecoveryState.store(3);
        return;
    }

    setOffset.Call(kCalibrationR20Neutral);
    savePrefs[prefs].Call();
    g_calibrationR20SaveCalled.store(1);

    const float after = getOffset.Call();
    g_calibrationR20AfterX100.store(static_cast<int>(after * 100.0f));
    if (after == kCalibrationR20Neutral) {
        g_calibrationR20Repaired.store(1);
        ClearMarker(g_calibrationR20Marker);
    } else {
        // Fail closed. Never repeat a write whose postcondition was not observed.
        g_calibrationR20RecoveryState.store(5);
    }
}

'''
once('constexpr float kCalibrationR19Unset = 999.0f;\n',
     code + 'constexpr float kCalibrationR19Unset = 999.0f;\n')

# R19 used UnityEngine.PlayerPrefs, which is not the target game's calibration backend.
# Keep the old body only for source history; never call it.
once(
    '    MaybeRepairCalibrationR19();\n'
    '    MaybeInstallSfbHook();\n',
    '    MaybeRepairCalibrationR20();\n'
    '    MaybeInstallSfbHook();\n'
)

once('        << "stabilityRevision=19" << \'\\n\'\n',
     '        << "stabilityRevision=20" << \'\\n\'\n')
once(
    '        << "calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero" << \'\\n\'\n',
    '        << "calibrationR19Policy=disabled-r20-wrong-backend-forensic-only" << \'\\n\'\n'
    '        << "calibrationR20Policy=exact-Persistence-PlayerPrefsJson-sentinel-999-to-zero" << \'\\n\'\n'
    '        << "calibrationR20Backend=PlayerPrefsJson" << \'\\n\'\n'
    '        << "calibrationR20Mutation=only-if-GetInputOffset-exact-999-preserve-existing" << \'\\n\'\n'
    '        << "calibrationR20Source=authoritative-v240-native-analysis" << \'\\n\'\n'
    '        << "calibrationR20GetInputOffsetRva=0x1163F54" << \'\\n\'\n'
    '        << "calibrationR20SetInputOffsetRva=0x11666E8" << \'\\n\'\n'
    '        << "calibrationR20GeneralPrefsRva=0x1162CF4" << \'\\n\'\n'
    '        << "calibrationR20PlayerPrefsJsonSaveRva=0x0FD7D74" << \'\\n\'\n'
    '        << "calibrationR20AbiGuard=" << g_calibrationR20AbiGuard.load() << \'\\n\'\n'
    '        << "calibrationR20Attempted=" << g_calibrationR20Attempted.load() << \'\\n\'\n'
    '        << "calibrationR20PrefsReady=" << g_calibrationR20PrefsReady.load() << \'\\n\'\n'
    '        << "calibrationR20UnsetDetected=" << g_calibrationR20UnsetDetected.load() << \'\\n\'\n'
    '        << "calibrationR20Repaired=" << g_calibrationR20Repaired.load() << \'\\n\'\n'
    '        << "calibrationR20SaveCalled=" << g_calibrationR20SaveCalled.load() << \'\\n\'\n'
    '        << "calibrationR20MarkerReady=" << g_calibrationR20MarkerReady.load() << \'\\n\'\n'
    '        << "calibrationR20RecoveryState=" << g_calibrationR20RecoveryState.load() << \'\\n\'\n'
    '        << "calibrationR20BeforeX100=" << g_calibrationR20BeforeX100.load() << \'\\n\'\n'
    '        << "calibrationR20AfterX100=" << g_calibrationR20AfterX100.load() << \'\\n\'\n'
)

for marker in (
    'stabilityRevision=20',
    'calibrationR19Policy=disabled-r20-wrong-backend-forensic-only',
    'calibrationR20Policy=exact-Persistence-PlayerPrefsJson-sentinel-999-to-zero',
    'calibrationR20Backend=PlayerPrefsJson',
    'Class prefsClass("", "PlayerPrefsJson")',
    'persistence.GetMethod("GetInputOffset", 0)',
    'persistence.GetMethod("SetInputOffset", 1)',
    'persistence.GetMethod("get_generalPrefs", 0)',
    'prefsClass.GetMethod("Save", 0)',
    'savePrefs[prefs].Call();',
    'calibration-r20-playerprefsjson-write.pending',
    'MaybeRepairCalibrationR20();',
):
    if marker not in s:
        raise SystemExit(f"r20 marker missing after transform: {marker}")

if '    MaybeRepairCalibrationR19();\n' in s:
    raise SystemExit("r19 wrong-backend repair must not remain active")
if s.count('BasicHook(') != 9:
    raise SystemExit(f"r20 adds no hooks; expected nine compiled sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
