#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r10-raycast-probe.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor, found {count}: {old[:96]!r}")
    s = s.replace(old, new, 1)


# Add a read-only EventSystem.RaycastAll observation surface. The r9 wrapper hook is left
# compiled for forensic continuity but is no longer installed: real-device evidence showed
# uiHitHookInstalled=1 with uiHitCalls=0 while tile taps still failed.
replace_once(
    "Method<void> g_raycastAll;\n",
    "Method<void> g_raycastAll;\n"
    "Method<int> g_getTouchCount;\n\n"
    "using RaycastProbeFn = void (*)(IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*,\n"
    "        IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);\n"
    "RaycastProbeFn g_oldRaycastProbe = nullptr;\n"
    "std::mutex g_raycastFirstCallMutex;\n"
    "std::atomic<bool> g_raycastHookInstalled{false};\n"
    "std::atomic<bool> g_raycastInstallAttempted{false};\n"
    "std::atomic<bool> g_raycastMarkerReady{false};\n"
    "std::atomic<bool> g_raycastFirstCallProven{false};\n"
    "std::atomic<int> g_raycastRecoveryState{0};\n"
    "std::atomic<int> g_raycastGuardValue{0};\n"
    "std::atomic<int> g_raycastCalls{0};\n"
    "std::atomic<int> g_raycastTouchActiveCalls{0};\n"
    "std::atomic<int> g_raycastCallsWithResults{0};\n"
    "std::atomic<int> g_raycastLastTouchCount{-1};\n"
    "std::atomic<int> g_raycastLastX100{0};\n"
    "std::atomic<int> g_raycastLastY100{0};\n"
    "std::atomic<int> g_raycastLastResultCount{-1};\n"
    "std::string g_raycastInstallMarker;\n"
    "std::string g_raycastCallMarker;\n"
)

replace_once(
    "bool GetEnv(JNIEnv** env, bool* attached) {\n",
    "bool PrepareRaycastFuse() {\n"
    "    const std::string dir = RuntimeDir();\n"
    "    if (dir.empty()) { g_raycastRecoveryState.store(3); return false; }\n"
    "    g_raycastInstallMarker = dir + \"/raycast-r10-install.pending\";\n"
    "    g_raycastCallMarker = dir + \"/raycast-r10-call.pending\";\n"
    "    g_raycastMarkerReady.store(true);\n"
    "    if (MarkerExists(g_raycastInstallMarker)) { g_raycastRecoveryState.store(1); return false; }\n"
    "    if (MarkerExists(g_raycastCallMarker)) { g_raycastRecoveryState.store(2); return false; }\n"
    "    const std::string probe = dir + \"/raycast-r10-probe.tmp\";\n"
    "    ClearMarker(probe);\n"
    "    if (!WriteMarker(probe)) {\n"
    "        g_raycastMarkerReady.store(false);\n"
    "        g_raycastRecoveryState.store(3);\n"
    "        return false;\n"
    "    }\n"
    "    ClearMarker(probe);\n"
    "    return true;\n"
    "}\n\n"
    "bool GetEnv(JNIEnv** env, bool* attached) {\n"
)

raycast_code = r'''
void HookRaycastProbe(IL2CPP::Il2CppObject* eventSystem,
                      IL2CPP::Il2CppObject* eventData,
                      IL2CPP::Il2CppObject* results,
                      IL2CPP::MethodInfo* methodInfo) {
    if (g_oldRaycastProbe == nullptr) return;

    bool firstCanary = false;
    if (!g_raycastFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_raycastFirstCallMutex);
        if (!g_raycastFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_raycastCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldRaycastProbe(eventSystem, eventData, results, methodInfo);
                return;
            }
            firstCanary = true;
        }
    }

    g_raycastCalls.fetch_add(1);
    int touchCount = -1;
    if (g_getTouchCount.IsValid()) touchCount = g_getTouchCount.Call();
    g_raycastLastTouchCount.store(touchCount);

    const bool activeTouch = touchCount > 0;
    if (activeTouch) {
        g_raycastTouchActiveCalls.fetch_add(1);
        if (eventData != nullptr && g_pointerPosition.IsValid()) {
            const Vector2 position = g_pointerPosition[eventData].Get();
            g_raycastLastX100.store(static_cast<int>(position.x * 100.0f));
            g_raycastLastY100.store(static_cast<int>(position.y * 100.0f));
        }
    }

    // Observation-only: forward the exact original arguments and result container unchanged.
    g_oldRaycastProbe(eventSystem, eventData, results, methodInfo);

    if (activeTouch && results != nullptr && g_listCount.IsValid()) {
        const int count = g_listCount[results].Get();
        g_raycastLastResultCount.store(count);
        if (count > 0) g_raycastCallsWithResults.fetch_add(1);
    }

    if (firstCanary) {
        ClearMarker(g_raycastCallMarker);
        g_raycastFirstCallProven.store(true, std::memory_order_release);
    }
}

bool ResolveRaycastProbe(MethodBase* raycastMethod) {
    Class eventSystem("UnityEngine.EventSystems", "EventSystem");
    Class pointerEventData("UnityEngine.EventSystems", "PointerEventData");
    Class raycastResult("UnityEngine.EventSystems", "RaycastResult");
    Class list("System.Collections.Generic", "List`1");
    Class input("UnityEngine", "Input");
    if (!eventSystem || !pointerEventData || !raycastResult || !list || !input) return false;

    Class listRaycast = list.GetGeneric({raycastResult.GetCompileTimeClass()});
    if (!listRaycast) return false;
    MethodBase method = eventSystem.GetMethod("RaycastAll", 2);
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    const bool params2 = info != nullptr && info->parameters_count == 2 && info->parameters;
    const IL2CPP::Il2CppType* p0 = params2 ? info->parameters[0] : nullptr;
    const IL2CPP::Il2CppType* p1 = params2 ? info->parameters[1] : nullptr;
    const bool abi = method.IsValid() && info && info->methodPointer && !method._isStatic &&
            params2 && SameClass(Class(p0), pointerEventData) &&
            SameClass(Class(p1), listRaycast) && TypeByRef(p0) == 0 && TypeByRef(p1) == 0;
    g_raycastGuardValue.store(abi ? 1 : 0);
    if (!abi) return false;

    g_pointerPosition = pointerEventData.GetProperty("position");
    g_listCount = listRaycast.GetProperty("Count");
    g_getTouchCount = input.GetMethod("get_touchCount", 0);
    const bool surface = g_pointerPosition.IsValid() && g_listCount.IsValid() &&
            g_getTouchCount.IsValid();
    if (surface && raycastMethod != nullptr) *raycastMethod = method;
    return surface;
}

void MaybeInstallRaycastProbe() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_raycastHookInstalled.load(std::memory_order_acquire) ||
        g_raycastRecoveryState.load() != 0) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_raycastHookInstalled.load() || g_raycastInstallAttempted.load()) return;

    MethodBase raycastMethod;
    if (!ResolveRaycastProbe(&raycastMethod) || !PrepareRaycastFuse()) return;
    if (!WriteMarker(g_raycastInstallMarker)) {
        g_raycastMarkerReady.store(false);
        g_raycastRecoveryState.store(3);
        return;
    }
    g_raycastInstallAttempted.store(true);
    BasicHook(raycastMethod, HookRaycastProbe, g_oldRaycastProbe);
    const bool installed = g_oldRaycastProbe != nullptr;
    g_raycastHookInstalled.store(installed);
    if (installed) ClearMarker(g_raycastInstallMarker);
    else g_raycastRecoveryState.store(4);
}

'''
replace_once("void MaybeInstallUiHook() {\n", raycast_code + "void MaybeInstallUiHook() {\n")

# r9's scrController wrapper was proven absent from the tile-tap path on the real device.
# Do not install it in r10; observe the lower EventSystem.RaycastAll surface instead.
replace_once(
    "void ReconcileInstallState() {\n    MaybeInstallUiHook();\n    MaybeInstallSfbHook();\n    BuildStaticReport();\n}\n",
    "void ReconcileInstallState() {\n    MaybeInstallRaycastProbe();\n    MaybeInstallSfbHook();\n    BuildStaticReport();\n}\n"
)

replace_once(
    'nativeProbe=cache-post-bnm-sfb-dynamic-import-uihit-v1\\n',
    'nativeProbe=cache-post-bnm-eventsystem-raycast-observe-v1\\n'
)
replace_once(
    'nativeStage=post-bnm-dynamic-document-and-uihit\\n',
    'nativeStage=post-bnm-eventsystem-raycast-pass-through\\n'
)
replace_once('abiProbeRevision=9\\n', 'abiProbeRevision=10\\n')
replace_once(
    '<< "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +\n                                        (g_uiHookInstalled.load() ? 1 : 0)) << \'\\n\'\n',
    '<< "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +\n                                        (g_raycastHookInstalled.load() ? 1 : 0)) << \'\\n\'\n'
)
replace_once(
    '<< "uiHitHookInstalled=" << (g_uiHookInstalled.load() ? 1 : 0) << \'\\n\'\n',
    '<< "raycastProbeHookInstalled=" << (g_raycastHookInstalled.load() ? 1 : 0) << \'\\n\'\n'
    '        << "raycastProbeAbiGuard=" << g_raycastGuardValue.load() << \'\\n\'\n'
    '        << "raycastProbePolicy=pass-through-observe-only\\n"\n'
    '        << "raycastProbeMutation=0\\n"\n'
    '        << "raycastProbeTouchCountRead=1\\n"\n'
    '        << "raycastProbeSelfFuse=1\\n"\n'
    '        << "raycastProbeMarkerReady=" << (g_raycastMarkerReady.load() ? 1 : 0) << \'\\n\'\n'
    '        << "raycastProbeRecoveryState=" << g_raycastRecoveryState.load() << \'\\n\'\n'
    '        << "raycastProbeInstallAttempted=" << (g_raycastInstallAttempted.load() ? 1 : 0) << \'\\n\'\n'
    '        << "raycastProbeOriginalCaptured=" << (g_oldRaycastProbe ? 1 : 0) << \'\\n\'\n'
    '        << "uiHitHookInstalled=" << (g_uiHookInstalled.load() ? 1 : 0) << \'\\n\'\n'
)
replace_once(
    '<< "uiHitPolicy=pinned-upstream-eventsystem-raycast\\n"\n',
    '<< "uiHitPolicy=disabled-r9-device-proven-not-on-tile-path\\n"\n'
)

replace_once(
    'out << "uiHitCalls=" << g_uiHitCalls.load() << \'\\n\'\n',
    'out << "raycastProbeCalls=" << g_raycastCalls.load() << \'\\n\'\n'
    '        << "raycastProbeTouchActiveCalls=" << g_raycastTouchActiveCalls.load() << \'\\n\'\n'
    '        << "raycastProbeCallsWithResults=" << g_raycastCallsWithResults.load() << \'\\n\'\n'
    '        << "raycastProbeFirstCallProven=" << (g_raycastFirstCallProven.load() ? 1 : 0) << \'\\n\'\n'
    '        << "raycastProbeLastTouchCount=" << g_raycastLastTouchCount.load() << \'\\n\'\n'
    '        << "raycastProbeLastX100=" << g_raycastLastX100.load() << \'\\n\'\n'
    '        << "raycastProbeLastY100=" << g_raycastLastY100.load() << \'\\n\'\n'
    '        << "raycastProbeLastResultCount=" << g_raycastLastResultCount.load() << \'\\n\'\n'
    '        << "uiHitCalls=" << g_uiHitCalls.load() << \'\\n\'\n'
)

# Compile-time/build-copy contract: exactly one new active observation hook is added, the
# r9 UI wrapper remains compiled but is no longer reached from ReconcileInstallState.
for marker in (
    "raycast-r10-install.pending",
    "raycast-r10-call.pending",
    "HookRaycastProbe",
    "MaybeInstallRaycastProbe",
    "BasicHook(raycastMethod, HookRaycastProbe, g_oldRaycastProbe)",
    "raycastProbePolicy=pass-through-observe-only",
    "raycastProbeMutation=0",
    "abiProbeRevision=10",
    "nativeProbe=cache-post-bnm-eventsystem-raycast-observe-v1",
    "g_oldRaycastProbe(eventSystem, eventData, results, methodInfo)",
):
    if marker not in s:
        raise SystemExit(f"missing r10 marker after transform: {marker}")
if "MaybeInstallUiHook();" in s:
    raise SystemExit("r9 UI wrapper must not be installed in r10")
if s.count("BasicHook(") != 3:
    raise SystemExit(f"expected three compiled BasicHook sites after r10 overlay, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
