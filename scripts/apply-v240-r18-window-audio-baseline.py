#!/usr/bin/env python3
"""R18: restore the original full-width phone viewport and silence only the startup audio-output rebaseline.

Why this replaces the r17 tile mutation:
- The bootstrap3 APK forces LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER through V240WindowCompat.
- On the affected phone Unity reported a 2237x1080 surface while the device landscape panel is wider.
- The old tablet, which does not run this bootstrap3 window policy, selects editor tiles normally.
- v2.4 ObjectsAtMouse is world/Physics2D based, so preserving Unity's full landscape viewport is
  safer than changing tile raycasts or selection.

Startup calibration notification:
- Exact v2.4 code shows scrController.Start -> CheckForAudioOutputChange.
- That method calls scrConductor.HasAudioOutputChanged and displays a notification when Android's
  output identity settles after ADOStartup.LoadCalibration.
- Suppress only the CheckForAudioOutputChange invocation made from scrController.Start by silently
  rebasing scrConductor.UpdateCurrentAudioOutput. Every later check calls the original unchanged.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r18-window-audio-baseline.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r18 anchor must occur once, got {n}: {old[:140]!r}")
    s = s.replace(old, new, 1)

once(
    'std::string g_editorPhysicsCallMarker;\n',
    'std::string g_editorPhysicsCallMarker;\n'
    'using StartupAudioVoidFn = void (*)(IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);\n'
    'StartupAudioVoidFn g_oldControllerStart = nullptr;\n'
    'StartupAudioVoidFn g_oldCheckForAudioOutputChange = nullptr;\n'
    'Method<void> g_updateCurrentAudioOutput;\n'
    'thread_local int g_controllerStartDepth = 0;\n'
    'std::mutex g_startupAudioFirstCallMutex;\n'
    'std::atomic<int> g_startupAudioAbiGuard{0};\n'
    'std::atomic<int> g_startupAudioHookInstalled{0};\n'
    'std::atomic<int> g_startupAudioInstallAttempted{0};\n'
    'std::atomic<int> g_startupAudioMarkerReady{0};\n'
    'std::atomic<int> g_startupAudioRecoveryState{0};\n'
    'std::atomic<int> g_startupAudioStartCalls{0};\n'
    'std::atomic<int> g_startupAudioCheckCalls{0};\n'
    'std::atomic<int> g_startupAudioStartupChecks{0};\n'
    'std::atomic<int> g_startupAudioBaselineUpdates{0};\n'
    'std::atomic<int> g_startupAudioSuppressedNotifications{0};\n'
    'std::atomic<int> g_startupAudioFirstCallProven{0};\n'
    'std::string g_startupAudioInstallMarker;\n'
    'std::string g_startupAudioCallMarker;\n'
)

audio_code = r"""
bool PrepareStartupAudioFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_startupAudioRecoveryState.store(4); return false; }
    g_startupAudioInstallMarker = dir + "/startup-audio-r18-install.pending";
    g_startupAudioCallMarker = dir + "/startup-audio-r18-call.pending";
    g_startupAudioMarkerReady.store(1);
    if (MarkerExists(g_startupAudioInstallMarker)) {
        g_startupAudioRecoveryState.store(1);
        return false;
    }
    if (MarkerExists(g_startupAudioCallMarker)) {
        g_startupAudioRecoveryState.store(2);
        return false;
    }
    const std::string probe = dir + "/startup-audio-r18-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_startupAudioMarkerReady.store(0);
        g_startupAudioRecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void HookControllerStart(IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldControllerStart) return;
    g_startupAudioStartCalls.fetch_add(1);
    ++g_controllerStartDepth;
    g_oldControllerStart(self, methodInfo);
    --g_controllerStartDepth;
}

void HookCheckForAudioOutputChange(IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldCheckForAudioOutputChange) return;
    g_startupAudioCheckCalls.fetch_add(1);

    // Only the check nested inside scrController.Start is treated as startup baseline.
    // Later Update() checks remain byte-for-byte original behavior.
    if (g_controllerStartDepth > 0 &&
        !g_startupAudioFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> guard(g_startupAudioFirstCallMutex);
        if (!g_startupAudioFirstCallProven.load(std::memory_order_relaxed)) {
            g_startupAudioStartupChecks.fetch_add(1);
            if (!WriteMarker(g_startupAudioCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldCheckForAudioOutputChange(self, methodInfo);
                return;
            }
            // ADOStartup already establishes this baseline once. Repeating it here after
            // Android's audio backend settles avoids the false "output changed" calibration
            // notification without writing PlayerPrefs or altering calibration offsets.
            g_updateCurrentAudioOutput.Call();
            g_startupAudioBaselineUpdates.fetch_add(1);
            g_startupAudioSuppressedNotifications.fetch_add(1);
            ClearMarker(g_startupAudioCallMarker);
            g_startupAudioFirstCallProven.store(1, std::memory_order_release);
            return;
        }
    }
    g_oldCheckForAudioOutputChange(self, methodInfo);
}

void MaybeInstallStartupAudioBaselineHook() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_startupAudioHookInstalled.load() ||
        g_startupAudioInstallAttempted.load() ||
        g_startupAudioRecoveryState.load() != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_startupAudioHookInstalled.load() || g_startupAudioInstallAttempted.load()) return;

    Class controller("", "scrController");
    Class conductor("", "scrConductor");
    if (!controller || !conductor) return;

    MethodBase start = controller.GetMethod("Start", 0);
    MethodBase check = controller.GetMethod("CheckForAudioOutputChange", 0);
    MethodBase update = conductor.GetMethod("UpdateCurrentAudioOutput", 0);
    IL2CPP::MethodInfo* startInfo = start.IsValid() ? start.GetInfo() : nullptr;
    IL2CPP::MethodInfo* checkInfo = check.IsValid() ? check.GetInfo() : nullptr;
    IL2CPP::MethodInfo* updateInfo = update.IsValid() ? update.GetInfo() : nullptr;

    const bool startAbi = startInfo && startInfo->methodPointer && !start._isStatic &&
            startInfo->parameters_count == 0 && startInfo->return_type &&
            TypeCode(startInfo->return_type) == 1;
    const bool checkAbi = checkInfo && checkInfo->methodPointer && !check._isStatic &&
            checkInfo->parameters_count == 0 && checkInfo->return_type &&
            TypeCode(checkInfo->return_type) == 1;
    const bool updateAbi = updateInfo && updateInfo->methodPointer && update._isStatic &&
            updateInfo->parameters_count == 0 && updateInfo->return_type &&
            TypeCode(updateInfo->return_type) == 1;
    const bool abi = startAbi && checkAbi && updateAbi;
    g_startupAudioAbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareStartupAudioFuse()) return;

    if (!WriteMarker(g_startupAudioInstallMarker)) {
        g_startupAudioMarkerReady.store(0);
        g_startupAudioRecoveryState.store(4);
        return;
    }
    g_startupAudioInstallAttempted.store(1);
    g_updateCurrentAudioOutput = Method<void>(update);

    // Install Check first. A partial install cannot suppress anything unless Start is also hooked.
    BasicHook(check, HookCheckForAudioOutputChange, g_oldCheckForAudioOutputChange);
    BasicHook(start, HookControllerStart, g_oldControllerStart);
    const bool installed = g_oldCheckForAudioOutputChange != nullptr &&
            g_oldControllerStart != nullptr && g_updateCurrentAudioOutput.IsValid();
    g_startupAudioHookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_startupAudioInstallMarker);
    else g_startupAudioRecoveryState.store(5);
}

"""
once('void MaybeInstallEditorPhysicsSync() {\n', audio_code + 'void MaybeInstallEditorPhysicsSync() {\n')

# The final tile path is intentionally the original game path. r11/r16/r17 remain compiled
# forensic evidence only; none of those editor probes or the speculative physics mutation installs.
for call in (
    '    MaybeInstallEditorProbe();\n',
    '    MaybeInstallEditorObjectsProbe();\n',
    '    MaybeInstallEditorPhysicsSync();\n',
):
    once(call, '')

# Install only the exact startup-audio baseline hook plus the independently proven SFB hook.
once(
    '    MaybeInstallSfbHook();\n',
    '    MaybeInstallStartupAudioBaselineHook();\n'
    '    MaybeInstallSfbHook();\n'
)

once(
    'editorProbePolicy=scnEditor-pass-through-observe-only',
    'editorProbePolicy=disabled-r18-original-editor-path-window-viewport-fix'
)
once(
    'editorObjectsPolicy=ObjectsAtMouse-pass-through-observe-only',
    'editorObjectsPolicy=disabled-r18-original-editor-path-window-viewport-fix'
)
once(
    'editorPhysicsPolicy=touch-editor-objects-raycast-sync-once-per-frame',
    'editorPhysicsPolicy=disabled-r18-window-viewport-fix'
)

# Keep the historical nativeProbe/abiProbeRevision fields for bundle compatibility; expose the
# actually active stability revision explicitly so old build contracts do not masquerade as active.
once(
    '        << "editorPhysicsOriginalCaptured=" << (g_oldEditorRaycast ? 1 : 0) << \'\\n\'\n',
    '        << "editorPhysicsOriginalCaptured=" << (g_oldEditorRaycast ? 1 : 0) << \'\\n\'\n'
    '        << "stabilityRevision=18" << \'\\n\'\n'
    '        << "activeTilePolicy=original-v240-editor-path-plus-full-width-window" << \'\\n\'\n'
    '        << "windowViewportPolicy=short-edges-full-width" << \'\\n\'\n'
    '        << "startupAudioPolicy=first-check-inside-scrController-Start-baseline-silent-then-original" << \'\\n\'\n'
    '        << "startupAudioMutation=runtime-baseline-only-no-PlayerPrefs" << \'\\n\'\n'
    '        << "startupAudioAbiGuard=" << g_startupAudioAbiGuard.load() << \'\\n\'\n'
    '        << "startupAudioHookInstalled=" << g_startupAudioHookInstalled.load() << \'\\n\'\n'
    '        << "startupAudioInstallAttempted=" << g_startupAudioInstallAttempted.load() << \'\\n\'\n'
    '        << "startupAudioMarkerReady=" << g_startupAudioMarkerReady.load() << \'\\n\'\n'
    '        << "startupAudioRecoveryState=" << g_startupAudioRecoveryState.load() << \'\\n\'\n'
    '        << "startupAudioStartCalls=" << g_startupAudioStartCalls.load() << \'\\n\'\n'
    '        << "startupAudioCheckCalls=" << g_startupAudioCheckCalls.load() << \'\\n\'\n'
    '        << "startupAudioStartupChecks=" << g_startupAudioStartupChecks.load() << \'\\n\'\n'
    '        << "startupAudioBaselineUpdates=" << g_startupAudioBaselineUpdates.load() << \'\\n\'\n'
    '        << "startupAudioSuppressedNotifications=" << g_startupAudioSuppressedNotifications.load() << \'\\n\'\n'
    '        << "startupAudioFirstCallProven=" << g_startupAudioFirstCallProven.load() << \'\\n\'\n'
)

for marker in (
    'stabilityRevision=18',
    'activeTilePolicy=original-v240-editor-path-plus-full-width-window',
    'windowViewportPolicy=short-edges-full-width',
    'startupAudioPolicy=first-check-inside-scrController-Start-baseline-silent-then-original',
    'scrController',
    'CheckForAudioOutputChange',
    'UpdateCurrentAudioOutput',
    'startup-audio-r18-install.pending',
    'startup-audio-r18-call.pending',
    'BasicHook(check, HookCheckForAudioOutputChange, g_oldCheckForAudioOutputChange)',
    'BasicHook(start, HookControllerStart, g_oldControllerStart)',
    'editorPhysicsPolicy=disabled-r18-window-viewport-fix',
):
    if marker not in s:
        raise SystemExit(f"missing r18 marker: {marker}")

for forbidden_call in (
    '    MaybeInstallEditorProbe();\n',
    '    MaybeInstallEditorObjectsProbe();\n',
    '    MaybeInstallEditorPhysicsSync();\n',
    '    MaybeNeutralizeUnsetCalibration();\n',
):
    if forbidden_call in s:
        raise SystemExit(f"r18 active forbidden call survived: {forbidden_call.strip()}")

if s.count('BasicHook(') != 9:
    raise SystemExit(f"r18 expected nine compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
