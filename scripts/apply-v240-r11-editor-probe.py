#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r11-editor-probe.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor, found {count}: {old[:120]!r}")
    s = s.replace(old, new, 1)


# r11 owns two independent fixes:
# 1) discover the hot-swapped DirectDocumentBridge through the thread context classloader from
#    the parent-loaded native report method (the r9 child-native registration path never linked),
# 2) observe the exact v2.4 editor tile path via scnEditor.HandleMouseActions/SelectFloor without
#    changing coordinates, arguments, return values, or selection state.

replace_once(
    "std::string g_raycastCallMarker;\n",
    "std::string g_raycastCallMarker;\n\n"
    "std::atomic<int> g_dynamicBridgeWakeCalls{0};\n"
    "std::atomic<int> g_dynamicBridgeContextLoaderSeen{0};\n"
    "std::atomic<int> g_dynamicBridgeLoadClass{0};\n"
    "std::atomic<int> g_dynamicBridgeMethodResolution{0};\n\n"
    "using EditorMouseFn = void (*)(IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);\n"
    "using SelectFloorFn = void (*)(IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*, bool, IL2CPP::MethodInfo*);\n"
    "EditorMouseFn g_oldEditorMouse = nullptr;\n"
    "SelectFloorFn g_oldSelectFloor = nullptr;\n"
    "Method<Vector3> g_editorGetMousePosition;\n"
    "Method<int> g_editorGetTouchCount;\n"
    "Method<int> g_editorGetScreenWidth;\n"
    "Method<int> g_editorGetScreenHeight;\n"
    "std::mutex g_editorFirstCallMutex;\n"
    "std::mutex g_selectFirstCallMutex;\n"
    "std::atomic<bool> g_editorProbeInstalled{false};\n"
    "std::atomic<bool> g_editorInstallAttempted{false};\n"
    "std::atomic<bool> g_editorMarkerReady{false};\n"
    "std::atomic<bool> g_editorFirstCallProven{false};\n"
    "std::atomic<bool> g_selectFirstCallProven{false};\n"
    "std::atomic<int> g_editorRecoveryState{0};\n"
    "std::atomic<int> g_editorHandleAbiGuard{0};\n"
    "std::atomic<int> g_editorSelectAbiGuard{0};\n"
    "std::atomic<int> g_editorMouseCalls{0};\n"
    "std::atomic<int> g_editorTouchFrames{0};\n"
    "std::atomic<int> g_editorSelectFloorCalls{0};\n"
    "std::atomic<int> g_editorLastMouseX100{0};\n"
    "std::atomic<int> g_editorLastMouseY100{0};\n"
    "std::atomic<int> g_editorLastTouchCount{0};\n"
    "std::atomic<int> g_editorScreenWidth{0};\n"
    "std::atomic<int> g_editorScreenHeight{0};\n"
    "std::atomic<int> g_editorLastFloorNonNull{0};\n"
    "std::atomic<int> g_editorLastCameraJump{0};\n"
    "std::string g_editorInstallMarker;\n"
    "std::string g_editorCallMarker;\n"
    "std::string g_selectCallMarker;\n"
)

bridge_code = r'''
bool RegisterDynamicBridgeClass(JNIEnv* env, jclass bridgeClass) {
    if (env == nullptr || bridgeClass == nullptr) return false;
    std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
    if (g_dynamicBridgeClass != nullptr && g_dynamicBegin != nullptr &&
        g_dynamicAwait != nullptr && g_dynamicDiagnosticsMethod != nullptr) {
        g_dynamicBridgeReady.store(true, std::memory_order_release);
        g_dynamicBridgeMethodResolution.store(1);
        return true;
    }

    jmethodID begin = env->GetStaticMethodID(
            bridgeClass, "begin", "(Ljava/lang/String;Ljava/lang/String;Z)I");
    jmethodID await = env->GetStaticMethodID(
            bridgeClass, "await", "(IJ)Ljava/lang/String;");
    jmethodID diagnostics = env->GetStaticMethodID(
            bridgeClass, "diagnostics", "()Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (begin == nullptr || await == nullptr || diagnostics == nullptr) return false;

    jclass global = reinterpret_cast<jclass>(env->NewGlobalRef(bridgeClass));
    if (global == nullptr) return false;
    g_dynamicBridgeClass = global;
    g_dynamicBegin = begin;
    g_dynamicAwait = await;
    g_dynamicDiagnosticsMethod = diagnostics;
    g_dynamicBridgeReady.store(true, std::memory_order_release);
    g_dynamicBridgeMethodResolution.store(1);
    return true;
}

void DiscoverDynamicBridgeFromContextLoader(JNIEnv* env) {
    if (env == nullptr || DynamicBridgeReady()) return;
    g_dynamicBridgeWakeCalls.fetch_add(1);

    jclass threadClass = env->FindClass("java/lang/Thread");
    if (threadClass == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        return;
    }
    jmethodID currentThread = env->GetStaticMethodID(
            threadClass, "currentThread", "()Ljava/lang/Thread;");
    jmethodID getContextClassLoader = env->GetMethodID(
            threadClass, "getContextClassLoader", "()Ljava/lang/ClassLoader;");
    if (env->ExceptionCheck() || currentThread == nullptr || getContextClassLoader == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(threadClass);
        return;
    }

    jobject thread = env->CallStaticObjectMethod(threadClass, currentThread);
    jobject loader = thread == nullptr ? nullptr :
            env->CallObjectMethod(thread, getContextClassLoader);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        loader = nullptr;
    }
    if (loader != nullptr) g_dynamicBridgeContextLoaderSeen.store(1);

    jclass bridgeClass = nullptr;
    if (loader != nullptr) {
        jclass loaderClass = env->GetObjectClass(loader);
        jmethodID loadClass = loaderClass == nullptr ? nullptr : env->GetMethodID(
                loaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
        jstring className = env->NewStringUTF(
                "dev.hoonex.adofai.v240.dynamic.DirectDocumentBridge");
        if (!env->ExceptionCheck() && loadClass != nullptr && className != nullptr) {
            bridgeClass = reinterpret_cast<jclass>(
                    env->CallObjectMethod(loader, loadClass, className));
        }
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            bridgeClass = nullptr;
        }
        if (bridgeClass != nullptr) g_dynamicBridgeLoadClass.store(1);
        if (className != nullptr) env->DeleteLocalRef(className);
        if (loaderClass != nullptr) env->DeleteLocalRef(loaderClass);
    }

    if (bridgeClass != nullptr) RegisterDynamicBridgeClass(env, bridgeClass);
    if (bridgeClass != nullptr) env->DeleteLocalRef(bridgeClass);
    if (loader != nullptr) env->DeleteLocalRef(loader);
    if (thread != nullptr) env->DeleteLocalRef(thread);
    env->DeleteLocalRef(threadClass);
}

'''
replace_once("bool DynamicBridgeReady() {\n", bridge_code + "bool DynamicBridgeReady() {\n")

# RegisterDynamicBridgeClass references DynamicBridgeReady through Discover... after its definition in
# source order. Add a small forward declaration before the helper block.
replace_once(
    "bool RegisterDynamicBridgeClass(JNIEnv* env, jclass bridgeClass) {\n",
    "bool DynamicBridgeReady();\n\nbool RegisterDynamicBridgeClass(JNIEnv* env, jclass bridgeClass) {\n"
)

replace_once(
    "bool GetEnv(JNIEnv** env, bool* attached) {\n",
    "bool PrepareEditorFuse() {\n"
    "    const std::string dir = RuntimeDir();\n"
    "    if (dir.empty()) { g_editorRecoveryState.store(4); return false; }\n"
    "    g_editorInstallMarker = dir + \"/editor-r11-install.pending\";\n"
    "    g_editorCallMarker = dir + \"/editor-r11-handle.pending\";\n"
    "    g_selectCallMarker = dir + \"/editor-r11-select.pending\";\n"
    "    g_editorMarkerReady.store(true);\n"
    "    if (MarkerExists(g_editorInstallMarker)) { g_editorRecoveryState.store(1); return false; }\n"
    "    if (MarkerExists(g_editorCallMarker)) { g_editorRecoveryState.store(2); return false; }\n"
    "    if (MarkerExists(g_selectCallMarker)) { g_editorRecoveryState.store(3); return false; }\n"
    "    const std::string probe = dir + \"/editor-r11-probe.tmp\";\n"
    "    ClearMarker(probe);\n"
    "    if (!WriteMarker(probe)) {\n"
    "        g_editorMarkerReady.store(false);\n"
    "        g_editorRecoveryState.store(4);\n"
    "        return false;\n"
    "    }\n"
    "    ClearMarker(probe);\n"
    "    return true;\n"
    "}\n\n"
    "bool GetEnv(JNIEnv** env, bool* attached) {\n"
)

editor_code = r'''
void SnapshotEditorInput() {
    if (!g_editorGetTouchCount.IsValid()) return;
    const int touchCount = g_editorGetTouchCount.Call();
    if (touchCount <= 0) return;
    g_editorTouchFrames.fetch_add(1);
    g_editorLastTouchCount.store(touchCount);
    if (g_editorGetMousePosition.IsValid()) {
        const Vector3 pos = g_editorGetMousePosition.Call();
        g_editorLastMouseX100.store(static_cast<int>(pos.x * 100.0f));
        g_editorLastMouseY100.store(static_cast<int>(pos.y * 100.0f));
    }
    if (g_editorGetScreenWidth.IsValid()) g_editorScreenWidth.store(g_editorGetScreenWidth.Call());
    if (g_editorGetScreenHeight.IsValid()) g_editorScreenHeight.store(g_editorGetScreenHeight.Call());
}

void HookEditorMouse(IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (g_oldEditorMouse == nullptr) return;
    bool firstCanary = false;
    if (!g_editorFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_editorFirstCallMutex);
        if (!g_editorFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_editorCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldEditorMouse(self, methodInfo);
                return;
            }
            firstCanary = true;
        }
    }

    g_editorMouseCalls.fetch_add(1);
    SnapshotEditorInput();
    g_oldEditorMouse(self, methodInfo);

    if (firstCanary) {
        ClearMarker(g_editorCallMarker);
        g_editorFirstCallProven.store(true, std::memory_order_release);
    }
}

void HookSelectFloor(IL2CPP::Il2CppObject* self, IL2CPP::Il2CppObject* floor,
                     bool cameraJump, IL2CPP::MethodInfo* methodInfo) {
    if (g_oldSelectFloor == nullptr) return;
    bool firstCanary = false;
    if (!g_selectFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_selectFirstCallMutex);
        if (!g_selectFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_selectCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                g_oldSelectFloor(self, floor, cameraJump, methodInfo);
                return;
            }
            firstCanary = true;
        }
    }

    g_editorSelectFloorCalls.fetch_add(1);
    g_editorLastFloorNonNull.store(floor != nullptr ? 1 : 0);
    g_editorLastCameraJump.store(cameraJump ? 1 : 0);
    g_oldSelectFloor(self, floor, cameraJump, methodInfo);

    if (firstCanary) {
        ClearMarker(g_selectCallMarker);
        g_selectFirstCallProven.store(true, std::memory_order_release);
    }
}

bool ResolveEditorProbe(MethodBase* handleOut, MethodBase* selectOut) {
    Class editor("", "scnEditor");
    Class floor("", "scrFloor");
    Class boolClass = Defaults::Get<bool>();
    Class input("UnityEngine", "Input");
    Class screen("UnityEngine", "Screen");
    if (!editor || !floor || !boolClass || !input || !screen) return false;

    MethodBase handle = editor.GetMethod("HandleMouseActions", 0);
    IL2CPP::MethodInfo* handleInfo = handle.IsValid() ? handle.GetInfo() : nullptr;
    const bool handleAbi = handle.IsValid() && handleInfo && handleInfo->methodPointer &&
            !handle._isStatic && handleInfo->parameters_count == 0;
    g_editorHandleAbiGuard.store(handleAbi ? 1 : 0);

    MethodBase select = editor.GetMethod("SelectFloor", 2);
    IL2CPP::MethodInfo* selectInfo = select.IsValid() ? select.GetInfo() : nullptr;
    const bool params2 = selectInfo != nullptr && selectInfo->parameters_count == 2 &&
            selectInfo->parameters != nullptr;
    const IL2CPP::Il2CppType* p0 = params2 ? selectInfo->parameters[0] : nullptr;
    const IL2CPP::Il2CppType* p1 = params2 ? selectInfo->parameters[1] : nullptr;
    const bool selectAbi = select.IsValid() && selectInfo && selectInfo->methodPointer &&
            !select._isStatic && params2 && SameClass(Class(p0), floor) &&
            SameClass(Class(p1), boolClass) && TypeByRef(p0) == 0 && TypeByRef(p1) == 0;
    g_editorSelectAbiGuard.store(selectAbi ? 1 : 0);
    if (!handleAbi || !selectAbi) return false;

    g_editorGetMousePosition = input.GetMethod("get_mousePosition", 0);
    g_editorGetTouchCount = input.GetMethod("get_touchCount", 0);
    g_editorGetScreenWidth = screen.GetMethod("get_width", 0);
    g_editorGetScreenHeight = screen.GetMethod("get_height", 0);
    if (!g_editorGetMousePosition.IsValid() || !g_editorGetTouchCount.IsValid() ||
        !g_editorGetScreenWidth.IsValid() || !g_editorGetScreenHeight.IsValid()) return false;

    if (handleOut != nullptr) *handleOut = handle;
    if (selectOut != nullptr) *selectOut = select;
    return true;
}

void MaybeInstallEditorProbe() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_editorProbeInstalled.load(std::memory_order_acquire) ||
        g_editorRecoveryState.load() != 0) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_editorProbeInstalled.load() || g_editorInstallAttempted.load()) return;

    MethodBase handleMethod;
    MethodBase selectMethod;
    if (!ResolveEditorProbe(&handleMethod, &selectMethod) || !PrepareEditorFuse()) return;
    if (!WriteMarker(g_editorInstallMarker)) {
        g_editorMarkerReady.store(false);
        g_editorRecoveryState.store(4);
        return;
    }
    g_editorInstallAttempted.store(true);
    BasicHook(handleMethod, HookEditorMouse, g_oldEditorMouse);
    BasicHook(selectMethod, HookSelectFloor, g_oldSelectFloor);
    const bool installed = g_oldEditorMouse != nullptr && g_oldSelectFloor != nullptr;
    g_editorProbeInstalled.store(installed);
    if (installed) ClearMarker(g_editorInstallMarker);
    else g_editorRecoveryState.store(5);
}

'''
replace_once("void MaybeInstallUiHook() {\n", editor_code + "void MaybeInstallUiHook() {\n")

# r10's broad EventSystem probe is superseded by the exact v2.4 editor input/selection methods.
replace_once(
    "void ReconcileInstallState() {\n    MaybeInstallRaycastProbe();\n    MaybeInstallSfbHook();\n    BuildStaticReport();\n}\n",
    "void ReconcileInstallState() {\n    MaybeInstallEditorProbe();\n    MaybeInstallSfbHook();\n    BuildStaticReport();\n}\n"
)

replace_once(
    'nativeProbe=cache-post-bnm-eventsystem-raycast-observe-v1\\n',
    'nativeProbe=cache-post-bnm-scneditor-input-observe-v1\\n'
)
replace_once(
    'nativeStage=post-bnm-eventsystem-raycast-pass-through\\n',
    'nativeStage=post-bnm-scneditor-handlemouse-selectfloor\\n'
)
replace_once('abiProbeRevision=10\\n', 'abiProbeRevision=11\\n')
replace_once(
    '<< "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +\n                                        (g_raycastHookInstalled.load() ? 1 : 0)) << \'\\n\'\n',
    '<< "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +\n                                        (g_editorProbeInstalled.load() ? 2 : 0)) << \'\\n\'\n'
)
replace_once(
    '<< "sfbDynamicBridgeReady=" << (g_dynamicBridgeReady.load() ? 1 : 0) << \'\\n\'\n',
    '<< "sfbDynamicBridgeReady=" << (g_dynamicBridgeReady.load() ? 1 : 0) << \'\\n\'\n'
    '        << "dynamicBridgeRegistrationPath=context-classloader-parent-native\\n"\n'
    '        << "dynamicBridgeWakeCalls=" << g_dynamicBridgeWakeCalls.load() << \'\\n\'\n'
    '        << "dynamicBridgeContextLoaderSeen=" << g_dynamicBridgeContextLoaderSeen.load() << \'\\n\'\n'
    '        << "dynamicBridgeLoadClass=" << g_dynamicBridgeLoadClass.load() << \'\\n\'\n'
    '        << "dynamicBridgeMethodResolution=" << g_dynamicBridgeMethodResolution.load() << \'\\n\'\n'
)
replace_once(
    '<< "raycastProbePolicy=pass-through-observe-only\\n"\n',
    '<< "raycastProbePolicy=disabled-r10-superseded-by-scnEditor\\n"\n'
)
replace_once(
    '<< "uiHitHookInstalled=" << (g_uiHookInstalled.load() ? 1 : 0) << \'\\n\'\n',
    '<< "editorProbeHookInstalled=" << (g_editorProbeInstalled.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorProbeHookCount=" << (g_editorProbeInstalled.load() ? 2 : 0) << \'\\n\'\n'
    '        << "editorHandleAbiGuard=" << g_editorHandleAbiGuard.load() << \'\\n\'\n'
    '        << "editorSelectAbiGuard=" << g_editorSelectAbiGuard.load() << \'\\n\'\n'
    '        << "editorProbePolicy=scnEditor-pass-through-observe-only\\n"\n'
    '        << "editorProbeMutation=0\\n"\n'
    '        << "editorProbeMetadataHandleRva=0x1C3D700\\n"\n'
    '        << "editorProbeMetadataSelectRva=0x1C47C80\\n"\n'
    '        << "editorProbeSelfFuse=1\\n"\n'
    '        << "editorProbeMarkerReady=" << (g_editorMarkerReady.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorProbeRecoveryState=" << g_editorRecoveryState.load() << \'\\n\'\n'
    '        << "editorProbeInstallAttempted=" << (g_editorInstallAttempted.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorHandleOriginalCaptured=" << (g_oldEditorMouse ? 1 : 0) << \'\\n\'\n'
    '        << "editorSelectOriginalCaptured=" << (g_oldSelectFloor ? 1 : 0) << \'\\n\'\n'
    '        << "uiHitHookInstalled=" << (g_uiHookInstalled.load() ? 1 : 0) << \'\\n\'\n'
)
replace_once(
    'out << "raycastProbeCalls=" << g_raycastCalls.load() << \'\\n\'\n',
    'out << "editorMouseCalls=" << g_editorMouseCalls.load() << \'\\n\'\n'
    '        << "editorTouchFrames=" << g_editorTouchFrames.load() << \'\\n\'\n'
    '        << "editorSelectFloorCalls=" << g_editorSelectFloorCalls.load() << \'\\n\'\n'
    '        << "editorHandleFirstCallProven=" << (g_editorFirstCallProven.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorSelectFirstCallProven=" << (g_selectFirstCallProven.load() ? 1 : 0) << \'\\n\'\n'
    '        << "editorLastMouseX100=" << g_editorLastMouseX100.load() << \'\\n\'\n'
    '        << "editorLastMouseY100=" << g_editorLastMouseY100.load() << \'\\n\'\n'
    '        << "editorLastTouchCount=" << g_editorLastTouchCount.load() << \'\\n\'\n'
    '        << "editorScreenWidth=" << g_editorScreenWidth.load() << \'\\n\'\n'
    '        << "editorScreenHeight=" << g_editorScreenHeight.load() << \'\\n\'\n'
    '        << "editorLastFloorNonNull=" << g_editorLastFloorNonNull.load() << \'\\n\'\n'
    '        << "editorLastCameraJump=" << g_editorLastCameraJump.load() << \'\\n\'\n'
    '        << "raycastProbeCalls=" << g_raycastCalls.load() << \'\\n\'\n'
)

# The parent-loaded report JNI method is a reliable bridge back into this native library. The
# child RuntimeEntry temporarily exposes its DexClassLoader through Thread.contextClassLoader.
replace_once(
    "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(JNIEnv* env, jclass) {\n"
    "    if (env == nullptr) return nullptr;\n"
    "    ReconcileInstallState();\n",
    "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(JNIEnv* env, jclass) {\n"
    "    if (env == nullptr) return nullptr;\n"
    "    DiscoverDynamicBridgeFromContextLoader(env);\n"
    "    ReconcileInstallState();\n"
)

for marker in (
    "abiProbeRevision=11",
    "nativeProbe=cache-post-bnm-scneditor-input-observe-v1",
    "dynamicBridgeRegistrationPath=context-classloader-parent-native",
    "DiscoverDynamicBridgeFromContextLoader(env)",
    "editor-r11-install.pending",
    "editor-r11-handle.pending",
    "editor-r11-select.pending",
    "HandleMouseActions",
    "SelectFloor",
    "BasicHook(handleMethod, HookEditorMouse, g_oldEditorMouse)",
    "BasicHook(selectMethod, HookSelectFloor, g_oldSelectFloor)",
    "editorProbePolicy=scnEditor-pass-through-observe-only",
    "editorProbeMutation=0",
    "raycastProbePolicy=disabled-r10-superseded-by-scnEditor",
):
    if marker not in s:
        raise SystemExit(f"missing r11 marker after transform: {marker}")
if "MaybeInstallUiHook();" in s:
    raise SystemExit("r9 UI wrapper must stay disabled in r11")
if "MaybeInstallRaycastProbe();" in s:
    raise SystemExit("r10 broad raycast probe must stay disabled in r11")
if s.count("BasicHook(") != 5:
    raise SystemExit(f"expected five compiled hook sites after r11 overlay, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
