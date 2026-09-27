#!/usr/bin/env python3
"""R19: repair only the authoritative v2.4 unconfigured calibration preference.

Exact source evidence from the authoritative APK:
- Persistence.GetInputOffset RVA 0x1163F54 reads PlayerPrefs.GetFloat("offset", 999.0f).
- Persistence.SetInputOffset RVA 0x11666E8 persists the same logical value.
- 999.0f is the exact unconfigured sentinel.
- The reported startup screen is the full device-calibration scene, not an audio-output-change
  notification.

R19 therefore keeps the r18 full-width SHORT_EDGES tile fix, disables the unrelated r18 startup
audio notification hook, and repairs only PlayerPrefs "offset" when it is exactly 999.0f.
Existing calibration values are never overwritten.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r19-exact-calibration.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"r19 anchor must occur once, got {count}: {old[:160]!r}")
    s = s.replace(old, new, 1)

once(
    'std::string g_startupAudioCallMarker;\n',
    'std::string g_startupAudioCallMarker;\n'
    'std::atomic<int> g_calibrationR19AbiGuard{0};\n'
    'std::atomic<int> g_calibrationR19Attempted{0};\n'
    'std::atomic<int> g_calibrationR19UnsetDetected{0};\n'
    'std::atomic<int> g_calibrationR19Repaired{0};\n'
    'std::atomic<int> g_calibrationR19SaveCalled{0};\n'
    'std::atomic<int> g_calibrationR19MarkerReady{0};\n'
    'std::atomic<int> g_calibrationR19RecoveryState{0};\n'
    'std::atomic<int> g_calibrationR19BeforeX100{0};\n'
    'std::atomic<int> g_calibrationR19AfterX100{0};\n'
    'std::string g_calibrationR19Marker;\n'
)

calibration_code = r"""
constexpr float kCalibrationR19Unset = 999.0f;
constexpr float kCalibrationR19Neutral = 0.0f;

bool PrepareCalibrationR19Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_calibrationR19RecoveryState.store(3);
        return false;
    }
    g_calibrationR19Marker = dir + "/calibration-r19-playerprefs-write.pending";
    g_calibrationR19MarkerReady.store(1);
    if (MarkerExists(g_calibrationR19Marker)) {
        g_calibrationR19RecoveryState.store(1);
        return false;
    }
    const std::string probe = dir + "/calibration-r19-playerprefs-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_calibrationR19MarkerReady.store(0);
        g_calibrationR19RecoveryState.store(3);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void MaybeRepairCalibrationR19() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_calibrationR19Attempted.load(std::memory_order_acquire) ||
        g_calibrationR19RecoveryState.load() != 0) return;

    Class playerPrefs("UnityEngine", "PlayerPrefs");
    Class stringClass = Defaults::Get<String*>();
    Class floatClass = Defaults::Get<float>();
    Class voidClass = Defaults::Get<void>();
    if (!playerPrefs || !stringClass || !floatClass || !voidClass) return;

    MethodBase getBase = playerPrefs.GetMethod("GetFloat", 2);
    MethodBase setBase = playerPrefs.GetMethod("SetFloat", 2);
    MethodBase saveBase = playerPrefs.GetMethod("Save", 0);
    IL2CPP::MethodInfo* getInfo = getBase.IsValid() ? getBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* setInfo = setBase.IsValid() ? setBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* saveInfo = saveBase.IsValid() ? saveBase.GetInfo() : nullptr;

    const bool getParams = getInfo && getInfo->parameters_count == 2 && getInfo->parameters;
    const bool setParams = setInfo && setInfo->parameters_count == 2 && setInfo->parameters;
    const bool abi = getInfo && getInfo->methodPointer && getBase._isStatic && getParams &&
            SameClass(Class(getInfo->return_type), floatClass) &&
            SameClass(Class(getInfo->parameters[0]), stringClass) &&
            SameClass(Class(getInfo->parameters[1]), floatClass) &&
            TypeByRef(getInfo->parameters[0]) == 0 &&
            TypeByRef(getInfo->parameters[1]) == 0 &&
            setInfo && setInfo->methodPointer && setBase._isStatic && setParams &&
            SameClass(Class(setInfo->return_type), voidClass) &&
            SameClass(Class(setInfo->parameters[0]), stringClass) &&
            SameClass(Class(setInfo->parameters[1]), floatClass) &&
            TypeByRef(setInfo->parameters[0]) == 0 &&
            TypeByRef(setInfo->parameters[1]) == 0 &&
            saveInfo && saveInfo->methodPointer && saveBase._isStatic &&
            saveInfo->parameters_count == 0 &&
            SameClass(Class(saveInfo->return_type), voidClass);
    g_calibrationR19AbiGuard.store(abi ? 1 : 0);
    if (!abi) return;

    g_calibrationR19Attempted.store(1, std::memory_order_release);
    Method<float> getFloat(getBase);
    Method<void> setFloat(setBase);
    Method<void> save(saveBase);
    String* key = CreateMonoString("offset");
    if (!key) {
        g_calibrationR19RecoveryState.store(4);
        return;
    }

    const float before = getFloat.Call(key, kCalibrationR19Unset);
    g_calibrationR19BeforeX100.store(static_cast<int>(before * 100.0f));
    g_calibrationR19AfterX100.store(static_cast<int>(before * 100.0f));

    // Preserve every real calibrated value. Only the exact game's own sentinel is repaired.
    if (before != kCalibrationR19Unset) return;
    g_calibrationR19UnsetDetected.store(1);

    if (!PrepareCalibrationR19Fuse()) return;
    if (!WriteMarker(g_calibrationR19Marker)) {
        g_calibrationR19MarkerReady.store(0);
        g_calibrationR19RecoveryState.store(3);
        return;
    }

    setFloat.Call(key, kCalibrationR19Neutral);
    save.Call();
    g_calibrationR19SaveCalled.store(1);
    const float after = getFloat.Call(key, kCalibrationR19Unset);
    g_calibrationR19AfterX100.store(static_cast<int>(after * 100.0f));
    if (after == kCalibrationR19Neutral) {
        g_calibrationR19Repaired.store(1);
        ClearMarker(g_calibrationR19Marker);
    } else {
        // Leave the marker so a later process fails closed instead of repeating the write.
        g_calibrationR19RecoveryState.store(5);
    }
}

"""
once('bool PrepareStartupAudioFuse() {\n',
     calibration_code + 'bool PrepareStartupAudioFuse() {\n')

# The r18 hook suppresses an audio-output-change notification, not the reported calibration scene.
# Keep its code as forensic evidence but never install it in the final policy.
once(
    '    MaybeInstallStartupAudioBaselineHook();\n'
    '    MaybeInstallSfbHook();\n',
    '    MaybeRepairCalibrationR19();\n'
    '    MaybeInstallSfbHook();\n'
)

once(
    'startupAudioPolicy=first-check-inside-scrController-Start-baseline-silent-then-original',
    'startupAudioPolicy=disabled-r19-not-device-calibration-root'
)
once(
    'startupAudioMutation=runtime-baseline-only-no-PlayerPrefs',
    'startupAudioMutation=disabled-r19'
)
once(
    '        << "startupAudioAbiGuard=" << g_startupAudioAbiGuard.load() << \'\\n\'\n',
    '        << "calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero" << \'\\n\'\n'
    '        << "calibrationR19Source=authoritative-v240-Persistence.GetInputOffset" << \'\\n\'\n'
    '        << "calibrationR19Mutation=only-if-exact-999-preserve-existing" << \'\\n\'\n'
    '        << "calibrationR19AbiGuard=" << g_calibrationR19AbiGuard.load() << \'\\n\'\n'
    '        << "calibrationR19Attempted=" << g_calibrationR19Attempted.load() << \'\\n\'\n'
    '        << "calibrationR19UnsetDetected=" << g_calibrationR19UnsetDetected.load() << \'\\n\'\n'
    '        << "calibrationR19Repaired=" << g_calibrationR19Repaired.load() << \'\\n\'\n'
    '        << "calibrationR19SaveCalled=" << g_calibrationR19SaveCalled.load() << \'\\n\'\n'
    '        << "calibrationR19MarkerReady=" << g_calibrationR19MarkerReady.load() << \'\\n\'\n'
    '        << "calibrationR19RecoveryState=" << g_calibrationR19RecoveryState.load() << \'\\n\'\n'
    '        << "calibrationR19BeforeX100=" << g_calibrationR19BeforeX100.load() << \'\\n\'\n'
    '        << "calibrationR19AfterX100=" << g_calibrationR19AfterX100.load() << \'\\n\'\n'
    '        << "startupAudioAbiGuard=" << g_startupAudioAbiGuard.load() << \'\\n\'\n'
)
once('        << "stabilityRevision=18" << \'\\n\'\n',
     '        << "stabilityRevision=19" << \'\\n\'\n')

# r11's old report values came from a different dump. Correct diagnostics to the authoritative APK.
once('editorProbeMetadataHandleRva=0x1C3D700',
     'editorProbeMetadataHandleRva=0x22E3EB0')
once('editorProbeMetadataSelectRva=0x1C47C80',
     'editorProbeMetadataSelectRva=0x22E7DD0')
once(
    '        << "editorProbeMetadataSelectRva=0x22E7DD0\\n"\n',
    '        << "editorProbeMetadataSelectRva=0x22E7DD0\\n"\n'
    '        << "editorProbeMetadataObjectsRva=0x22E8DF4\\n"\n'
    '        << "editorProbeMetadataSource=authoritative-v240-apk\\n"\n'
)

for marker in (
    'stabilityRevision=19',
    'calibrationR19Policy=exact-playerprefs-offset-sentinel-999-to-zero',
    'calibrationR19Mutation=only-if-exact-999-preserve-existing',
    'playerPrefs.GetMethod("GetFloat", 2)',
    'playerPrefs.GetMethod("SetFloat", 2)',
    'playerPrefs.GetMethod("Save", 0)',
    'CreateMonoString("offset")',
    'kCalibrationR19Unset = 999.0f',
    'calibration-r19-playerprefs-write.pending',
    'MaybeRepairCalibrationR19();',
    'startupAudioPolicy=disabled-r19-not-device-calibration-root',
    'editorProbeMetadataHandleRva=0x22E3EB0',
    'editorProbeMetadataSelectRva=0x22E7DD0',
    'editorProbeMetadataObjectsRva=0x22E8DF4',
):
    if marker not in s:
        raise SystemExit(f"r19 marker missing after transform: {marker}")

for forbidden in (
    '    MaybeInstallStartupAudioBaselineHook();\n',
    'editorProbeMetadataHandleRva=0x1C3D700',
    'editorProbeMetadataSelectRva=0x1C47C80',
):
    if forbidden in s:
        raise SystemExit(f"r19 forbidden active/stale marker survived: {forbidden.strip()}")

if s.count('BasicHook(') != 9:
    raise SystemExit(f"r19 adds no hook; expected nine compiled sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
