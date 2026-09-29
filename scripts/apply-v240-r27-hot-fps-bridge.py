#!/usr/bin/env python3
"""r27: restore the parent mobile settings FPS JNI contract in the hot cache runtime.

Bootstrap3 intentionally loads only the hash-verified code_cache libv240fix.so. The stable
parent V240SettingsOverlay still calls nativeApply/nativeApplyTouchAssist, but the cache
runtime did not export those JNI methods. As a result Android display-mode preference could
change while Unity Application.targetFrameRate/vSync policy remained unconfigured.

This overlay adds only exact Application.set_targetFrameRate(int) and
QualitySettings.set_vSyncCount(int) hooks, preserves every game-requested value, makes the
policy reversible, and exports the parent JNI surface. UI/touch expansion is NOT guessed
here; nativeApplyTouchAssist explicitly returns false so the Java layer does not believe an
enhanced touch hook exists.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r27-hot-fps-bridge.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"r27 anchor expected once, found {count}: {old[:140]!r}")
    s = s.replace(old, new, 1)

# State is deliberately independent from the historical fixed-native runtime. Bootstrap3
# never hard-loads that embedded library, so the cache runtime must own this JNI contract.
globals_code = r'''
using FpsR27SetterFn = void (*)(int, IL2CPP::MethodInfo*);
FpsR27SetterFn g_oldFpsR27SetTarget = nullptr;
FpsR27SetterFn g_oldFpsR27SetVsync = nullptr;
IL2CPP::MethodInfo* g_fpsR27TargetMethodInfo = nullptr;
IL2CPP::MethodInfo* g_fpsR27VsyncMethodInfo = nullptr;
std::atomic<int> g_fpsR27HookInstalled{0};
std::atomic<int> g_fpsR27AbiGuard{0};
std::atomic<int> g_fpsR27InstallAttempted{0};
std::atomic<int> g_fpsR27MarkerReady{0};
std::atomic<int> g_fpsR27RecoveryState{0};
std::atomic<int> g_fpsR27TargetFirstCallProven{0};
std::atomic<int> g_fpsR27VsyncFirstCallProven{0};
std::atomic<int> g_fpsR27FirstApplyProven{0};
std::atomic<int> g_fpsR27Configured{0};
std::atomic<int> g_fpsR27NativeApplyCalls{0};
std::atomic<int> g_fpsR27TouchAssistCalls{0};
std::atomic<int> g_fpsR27DesiredTarget{0};
std::atomic<int> g_fpsR27DesiredUnlock{0};
std::atomic<int> g_fpsR27DesiredLowLatency{0};
std::atomic<int> g_fpsR27LastRequestedTarget{-1};
std::atomic<int> g_fpsR27LastRequestedVsync{1};
std::atomic<int> g_fpsR27LastAppliedTarget{-1};
std::atomic<int> g_fpsR27LastAppliedVsync{1};
std::atomic<int> g_fpsR27PolicyApplyCalls{0};
std::mutex g_fpsR27TargetFirstCallMutex;
std::mutex g_fpsR27VsyncFirstCallMutex;
std::mutex g_fpsR27FirstApplyMutex;
std::string g_fpsR27InstallMarker;
std::string g_fpsR27TargetCallMarker;
std::string g_fpsR27VsyncCallMarker;
std::string g_fpsR27ApplyMarker;

'''
once('std::atomic<int> g_sfbFolderReturns{0};\n', 'std::atomic<int> g_sfbFolderReturns{0};\n\n' + globals_code)

fps_code = r'''
bool PrepareFpsR27Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_fpsR27RecoveryState.store(5);
        return false;
    }
    g_fpsR27InstallMarker = dir + "/fps-r27-install.pending";
    g_fpsR27TargetCallMarker = dir + "/fps-r27-target-call.pending";
    g_fpsR27VsyncCallMarker = dir + "/fps-r27-vsync-call.pending";
    g_fpsR27ApplyMarker = dir + "/fps-r27-apply.pending";
    g_fpsR27MarkerReady.store(1);
    if (MarkerExists(g_fpsR27InstallMarker)) {
        g_fpsR27RecoveryState.store(1);
        return false;
    }
    if (MarkerExists(g_fpsR27TargetCallMarker)) {
        g_fpsR27RecoveryState.store(2);
        return false;
    }
    if (MarkerExists(g_fpsR27VsyncCallMarker)) {
        g_fpsR27RecoveryState.store(3);
        return false;
    }
    if (MarkerExists(g_fpsR27ApplyMarker)) {
        g_fpsR27RecoveryState.store(4);
        return false;
    }
    const std::string probe = dir + "/fps-r27-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_fpsR27MarkerReady.store(0);
        g_fpsR27RecoveryState.store(5);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void HookFpsR27SetTarget(int fps, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldFpsR27SetTarget) return;
    bool first = false;
    if (!g_fpsR27TargetFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_fpsR27TargetFirstCallMutex);
        if (!g_fpsR27TargetFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_fpsR27TargetCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldFpsR27SetTarget(fps, methodInfo);
                return;
            }
            first = true;
        }
    }

    g_fpsR27LastRequestedTarget.store(fps);
    int applied = fps;
    if (g_fpsR27Configured.load(std::memory_order_acquire) &&
        g_fpsR27DesiredUnlock.load() && g_fpsR27DesiredTarget.load() > 0) {
        applied = g_fpsR27DesiredTarget.load();
    }
    g_oldFpsR27SetTarget(applied, methodInfo);
    g_fpsR27LastAppliedTarget.store(applied);

    if (first) {
        ClearMarker(g_fpsR27TargetCallMarker);
        g_fpsR27TargetFirstCallProven.store(1, std::memory_order_release);
    }
}

void HookFpsR27SetVsync(int count, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldFpsR27SetVsync) return;
    bool first = false;
    if (!g_fpsR27VsyncFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_fpsR27VsyncFirstCallMutex);
        if (!g_fpsR27VsyncFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_fpsR27VsyncCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldFpsR27SetVsync(count, methodInfo);
                return;
            }
            first = true;
        }
    }

    g_fpsR27LastRequestedVsync.store(count);
    const int applied = g_fpsR27Configured.load(std::memory_order_acquire) &&
                        g_fpsR27DesiredLowLatency.load() ? 0 : count;
    g_oldFpsR27SetVsync(applied, methodInfo);
    g_fpsR27LastAppliedVsync.store(applied);

    if (first) {
        ClearMarker(g_fpsR27VsyncCallMarker);
        g_fpsR27VsyncFirstCallProven.store(1, std::memory_order_release);
    }
}

void ApplyFpsR27PolicyNow() {
    if (!g_fpsR27Configured.load(std::memory_order_acquire) ||
        !g_fpsR27HookInstalled.load(std::memory_order_acquire) ||
        !g_oldFpsR27SetTarget || !g_oldFpsR27SetVsync ||
        !g_fpsR27TargetMethodInfo || !g_fpsR27VsyncMethodInfo) return;

    bool first = false;
    if (!g_fpsR27FirstApplyProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_fpsR27FirstApplyMutex);
        if (!g_fpsR27FirstApplyProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_fpsR27ApplyMarker)) {
                g_markerWriteFailures.fetch_add(1);
                return;
            }
            first = true;
        }
    }

    const int requestedTarget = g_fpsR27LastRequestedTarget.load();
    const int desiredTarget = g_fpsR27DesiredTarget.load();
    const int appliedTarget = g_fpsR27DesiredUnlock.load() && desiredTarget > 0
            ? desiredTarget : requestedTarget;
    const int requestedVsync = g_fpsR27LastRequestedVsync.load();
    const int appliedVsync = g_fpsR27DesiredLowLatency.load() ? 0 : requestedVsync;

    g_oldFpsR27SetTarget(appliedTarget, g_fpsR27TargetMethodInfo);
    g_oldFpsR27SetVsync(appliedVsync, g_fpsR27VsyncMethodInfo);
    g_fpsR27LastAppliedTarget.store(appliedTarget);
    g_fpsR27LastAppliedVsync.store(appliedVsync);
    g_fpsR27PolicyApplyCalls.fetch_add(1);

    if (first) {
        ClearMarker(g_fpsR27ApplyMarker);
        g_fpsR27FirstApplyProven.store(1, std::memory_order_release);
    }
}

void MaybeInstallFpsR27() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_fpsR27HookInstalled.load() || g_fpsR27InstallAttempted.load() ||
        g_fpsR27RecoveryState.load() != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_fpsR27HookInstalled.load() || g_fpsR27InstallAttempted.load()) return;

    Class application("UnityEngine", "Application");
    Class quality("UnityEngine", "QualitySettings");
    Class intClass = Defaults::Get<int>();
    if (!application || !quality || !intClass) return;

    MethodBase setTarget = application.GetMethod("set_targetFrameRate", 1);
    MethodBase setVsync = quality.GetMethod("set_vSyncCount", 1);
    IL2CPP::MethodInfo* targetInfo = setTarget.IsValid() ? setTarget.GetInfo() : nullptr;
    IL2CPP::MethodInfo* vsyncInfo = setVsync.IsValid() ? setVsync.GetInfo() : nullptr;

    const bool targetAbi = targetInfo && targetInfo->methodPointer && setTarget._isStatic &&
            targetInfo->parameters_count == 1 && targetInfo->parameters &&
            targetInfo->parameters[0] && TypeByRef(targetInfo->parameters[0]) == 0 &&
            SameClass(Class(targetInfo->parameters[0]), intClass) &&
            targetInfo->return_type && TypeCode(targetInfo->return_type) == 1;
    const bool vsyncAbi = vsyncInfo && vsyncInfo->methodPointer && setVsync._isStatic &&
            vsyncInfo->parameters_count == 1 && vsyncInfo->parameters &&
            vsyncInfo->parameters[0] && TypeByRef(vsyncInfo->parameters[0]) == 0 &&
            SameClass(Class(vsyncInfo->parameters[0]), intClass) &&
            vsyncInfo->return_type && TypeCode(vsyncInfo->return_type) == 1;
    const bool abi = targetAbi && vsyncAbi;
    g_fpsR27AbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareFpsR27Fuse()) return;

    if (!WriteMarker(g_fpsR27InstallMarker)) {
        g_fpsR27MarkerReady.store(0);
        g_fpsR27RecoveryState.store(5);
        return;
    }

    g_fpsR27InstallAttempted.store(1);
    g_fpsR27TargetMethodInfo = targetInfo;
    g_fpsR27VsyncMethodInfo = vsyncInfo;
    BasicHook(setTarget, HookFpsR27SetTarget, g_oldFpsR27SetTarget);
    BasicHook(setVsync, HookFpsR27SetVsync, g_oldFpsR27SetVsync);
    const bool installed = g_oldFpsR27SetTarget && g_oldFpsR27SetVsync;
    g_fpsR27HookInstalled.store(installed ? 1 : 0);
    if (installed) {
        ClearMarker(g_fpsR27InstallMarker);
        ApplyFpsR27PolicyNow();
    } else {
        g_fpsR27RecoveryState.store(6);
    }
}

'''
once('void MaybeInstallSfbExtraHooks() {\n', fps_code + 'void MaybeInstallSfbExtraHooks() {\n')

once(
    '    MaybeInstallSfbExtraHooks();\n    MaybeInstallSfbHook();\n',
    '    MaybeInstallSfbExtraHooks();\n    MaybeInstallFpsR27();\n    MaybeInstallSfbHook();\n'
)

once(
    '                                        (g_sfbExtraHookInstalled.load() ? 3 : 0)) << \'\\n\'\n',
    '                                        (g_sfbExtraHookInstalled.load() ? 3 : 0) +\n'
    '                                        (g_fpsR27HookInstalled.load() ? 2 : 0)) << \'\\n\'\n'
)
once(
    '        << "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2\\n"\n',
    '        << "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2\\n"\n'
    '        << "fpsR27Revision=27\\n"\n'
    '        << "fpsR27Policy=hot-cache-Application-targetFrameRate-QualitySettings-vSync-reversible\\n"\n'
    '        << "fpsR27UiTouchPolicy=enhanced-touch-unavailable-return-false-no-false-claim\\n"\n'
    '        << "fpsR27AbiGuard=" << g_fpsR27AbiGuard.load() << \'\\n\'\n'
    '        << "fpsR27HookInstalled=" << g_fpsR27HookInstalled.load() << \'\\n\'\n'
    '        << "fpsR27InstallAttempted=" << g_fpsR27InstallAttempted.load() << \'\\n\'\n'
    '        << "fpsR27MarkerReady=" << g_fpsR27MarkerReady.load() << \'\\n\'\n'
    '        << "fpsR27RecoveryState=" << g_fpsR27RecoveryState.load() << \'\\n\'\n'
    '        << "fpsR27Configured=" << g_fpsR27Configured.load() << \'\\n\'\n'
    '        << "fpsR27NativeApplyCalls=" << g_fpsR27NativeApplyCalls.load() << \'\\n\'\n'
    '        << "fpsR27TouchAssistCalls=" << g_fpsR27TouchAssistCalls.load() << \'\\n\'\n'
    '        << "fpsR27DesiredTarget=" << g_fpsR27DesiredTarget.load() << \'\\n\'\n'
    '        << "fpsR27DesiredUnlock=" << g_fpsR27DesiredUnlock.load() << \'\\n\'\n'
    '        << "fpsR27DesiredLowLatency=" << g_fpsR27DesiredLowLatency.load() << \'\\n\'\n'
    '        << "fpsR27LastRequestedTarget=" << g_fpsR27LastRequestedTarget.load() << \'\\n\'\n'
    '        << "fpsR27LastRequestedVsync=" << g_fpsR27LastRequestedVsync.load() << \'\\n\'\n'
    '        << "fpsR27LastAppliedTarget=" << g_fpsR27LastAppliedTarget.load() << \'\\n\'\n'
    '        << "fpsR27LastAppliedVsync=" << g_fpsR27LastAppliedVsync.load() << \'\\n\'\n'
    '        << "fpsR27PolicyApplyCalls=" << g_fpsR27PolicyApplyCalls.load() << \'\\n\'\n'
)

jni_code = r'''
extern "C" JNIEXPORT void JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeApply(
        JNIEnv*, jclass,
        jfloat, jfloat, jfloat, jboolean,
        jint targetFps, jboolean unlockFps, jboolean lowLatency) {
    int fps = static_cast<int>(targetFps);
    if (fps > 0) {
        if (fps < 30) fps = 30;
        if (fps > 240) fps = 240;
    } else {
        fps = 0;
    }
    g_fpsR27DesiredTarget.store(fps);
    g_fpsR27DesiredUnlock.store(unlockFps == JNI_TRUE ? 1 : 0);
    g_fpsR27DesiredLowLatency.store(lowLatency == JNI_TRUE ? 1 : 0);
    g_fpsR27Configured.store(1, std::memory_order_release);
    g_fpsR27NativeApplyCalls.fetch_add(1);
    ApplyFpsR27PolicyNow();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(
        JNIEnv*, jclass, jboolean, jfloat) {
    // Do not claim the old EventSystem touch-expansion hook exists in Bootstrap3.
    // Tile selection has its separate exact r21/r23 physics-path repair.
    g_fpsR27TouchAssistCalls.fetch_add(1);
    return JNI_FALSE;
}

'''
once('extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {\n',
     jni_code + 'extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {\n')

for marker in (
    "fpsR27Revision=27",
    "fpsR27Policy=hot-cache-Application-targetFrameRate-QualitySettings-vSync-reversible",
    "Java_com_unity3d_player_V240SettingsOverlay_nativeApply(",
    "Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(",
    'application.GetMethod("set_targetFrameRate", 1)',
    'quality.GetMethod("set_vSyncCount", 1)',
    "BasicHook(setTarget, HookFpsR27SetTarget, g_oldFpsR27SetTarget)",
    "BasicHook(setVsync, HookFpsR27SetVsync, g_oldFpsR27SetVsync)",
    "fps-r27-install.pending",
    "fps-r27-target-call.pending",
    "fps-r27-vsync-call.pending",
    "fps-r27-apply.pending",
    "MaybeInstallFpsR27();",
    "activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2-fpsR27-2",
):
    if marker not in s:
        raise SystemExit(f"r27 marker missing: {marker}")

if s.count("BasicHook(") != 18:
    raise SystemExit(f"r27 expected 18 compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")

r28 = Path(__file__).with_name("apply-v240-r28-objects-scope-tile-sync.py")
if not r28.is_file():
    raise SystemExit(f"missing r28 overlay: {r28}")
__import__("subprocess").run([sys.executable, str(r28), str(path)], check=True)
