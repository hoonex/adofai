#include <jni.h>
#include <atomic>
#include <mutex>
#include <sstream>
#include <string>

#include "universe.h"

using namespace BNM;

namespace {
std::atomic<bool> g_bnmLoadRequested{false};
std::atomic<bool> g_bnmLoadedCallback{false};
std::atomic<bool> g_probeComplete{false};
std::mutex g_reportMutex;
std::string g_report =
        "nativeProbe=cache-post-bnm-probe-only-v2\n"
        "nativeStage=post-bnm-read-only-abi\n"
        "bnmLoadRequested=0\n"
        "bnmLoadedCallback=0\n"
        "probeComplete=0\n"
        "gameHooksInstalled=0\n"
        "sfbOpenFiltersHookInstalled=0\n"
        "sfbFilterMemoryRead=0\n"
        "sfbHookPolicy=disabled-unproven-call-abi\n";

void RunReadOnlyAbiProbe() {
    // Evidence-only recovery payload. Resolve metadata only after BNM is loaded.
    // Never invoke managed code, write managed state, create objects, or install hooks.
    // In particular, method/field existence does not prove the native call ABI for
    // SFB.ExtensionFilter[] and therefore is not enough evidence to activate BasicHook.
    Class browser("SFB", "StandaloneFileBrowser");
    Class extensionFilter("SFB", "ExtensionFilter");
    Class canvasScaler("UnityEngine.UI", "CanvasScaler");
    Class input("UnityEngine", "Input");
    Class controller("", "scrController");
    Class eventSystem("UnityEngine.EventSystems", "EventSystem");
    Class pointerEventData("UnityEngine.EventSystems", "PointerEventData");
    Class raycastResult("UnityEngine.EventSystems", "RaycastResult");
    Class genericList("System.Collections.Generic", "List`1");
    Class application("UnityEngine", "Application");
    Class qualitySettings("UnityEngine", "QualitySettings");
    Class adoBase("", "ADOBase");
    Class scrCamera("", "scrCamera");
    Class levelEvent("ADOFAI", "LevelEvent");
    Class scrPlanet("", "scrPlanet");
    Class ffxCallMethod("", "ffxCallMethod");
    Class pauseMenu("", "PauseMenu");

    const bool openFile4 = browser && browser.GetMethod("OpenFilePanel", 4).IsValid();
    const bool openFiltersExact = browser && browser.GetMethod(
            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"}).IsValid();
    const bool saveFile4 = browser && browser.GetMethod("SaveFilePanel", 4).IsValid();
    const bool openFolder3 = browser && browser.GetMethod("OpenFolderPanel", 3).IsValid();
    const bool openFileAsync5 = browser && browser.GetMethod("OpenFilePanelAsync", 5).IsValid();
    const bool saveFileAsync5 = browser && browser.GetMethod("SaveFilePanelAsync", 5).IsValid();
    const bool openFolderAsync4 = browser && browser.GetMethod("OpenFolderPanelAsync", 4).IsValid();

    const bool filterNameField = extensionFilter && extensionFilter.GetField("Name").IsValid();
    const bool filterExtensionsField = extensionFilter && extensionFilter.GetField("Extensions").IsValid();

    const bool setScaleFactor1 = canvasScaler && canvasScaler.GetMethod("SetScaleFactor", 1).IsValid();
    const bool getAxis1 = input && input.GetMethod("GetAxis", 1).IsValid();
    const bool getAxisRaw1 = input && input.GetMethod("GetAxisRaw", 1).IsValid();
    const bool insideUi1 = controller
            && controller.GetMethod("IsScreenPointInsideUIElements", 1).IsValid();

    Class listRaycastResult = (genericList && raycastResult)
            ? genericList.GetGeneric({raycastResult.GetCompileTimeClass()}) : Class{};
    const bool eventCurrent = eventSystem && eventSystem.GetProperty("current").IsValid();
    const bool raycastAll = eventSystem && eventSystem.GetMethod("RaycastAll").IsValid();
    const bool pointerPosition = pointerEventData
            && pointerEventData.GetProperty("position").IsValid();
    const bool listCount = listRaycastResult
            && listRaycastResult.GetProperty("Count").IsValid();
    const bool listClear = listRaycastResult
            && listRaycastResult.GetMethod("Clear", 0).IsValid();
    const bool touchCount = input && input.GetMethod("get_touchCount", 0).IsValid();
    const bool sceneName = adoBase && adoBase.GetMethod("get_sceneName").IsValid();

    const bool setTargetFrameRate1 = application
            && application.GetMethod("set_targetFrameRate", 1).IsValid();
    const bool setVSyncCount1 = qualitySettings
            && qualitySettings.GetMethod("set_vSyncCount", 1).IsValid();

    const bool setCustomFrameRateBoolInt = scrCamera && scrCamera.GetMethod(
            "SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}).IsValid();
    const bool callMethodName = ffxCallMethod && ffxCallMethod.GetField("methodName").IsValid();
    const bool callMethodDecode = ffxCallMethod && levelEvent && ffxCallMethod.GetMethod(
            "Decode", {levelEvent.GetCompileTimeClass()}).IsValid();
    const bool callMethodStartEffect = ffxCallMethod && scrPlanet && ffxCallMethod.GetMethod(
            "StartEffect", {scrPlanet.GetCompileTimeClass()}).IsValid();

    const bool pauseMenuClass = static_cast<bool>(pauseMenu);
    const bool pauseMenuShowSettingsMenu0 = pauseMenu
            && pauseMenu.GetMethod("ShowSettingsMenu", 0).IsValid();

    std::ostringstream out;
    out << "nativeProbe=cache-post-bnm-probe-only-v2\n"
        << "nativeStage=post-bnm-read-only-abi\n"
        << "bnmLoadRequested=1\n"
        << "bnmLoadedCallback=1\n"
        << "probeComplete=1\n"
        << "gameHooksInstalled=0\n"
        << "sfbOpenFiltersHookInstalled=0\n"
        << "sfbFilterMemoryRead=0\n"
        << "sfbHookPolicy=disabled-unproven-call-abi\n"
        << "abi.SFB.class=" << (browser ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel4=" << (openFile4 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.filtersExact=" << (openFiltersExact ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.class=" << (extensionFilter ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.Name=" << (filterNameField ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.Extensions=" << (filterExtensionsField ? 1 : 0) << '\n'
        << "abi.SFB.SaveFilePanel4=" << (saveFile4 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFolderPanel3=" << (openFolder3 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanelAsync5=" << (openFileAsync5 ? 1 : 0) << '\n'
        << "abi.SFB.SaveFilePanelAsync5=" << (saveFileAsync5 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFolderPanelAsync4=" << (openFolderAsync4 ? 1 : 0) << '\n'
        << "abi.Mobile.CanvasScaler.SetScaleFactor1=" << (setScaleFactor1 ? 1 : 0) << '\n'
        << "abi.Mobile.Input.GetAxis1=" << (getAxis1 ? 1 : 0) << '\n'
        << "abi.Mobile.Input.GetAxisRaw1=" << (getAxisRaw1 ? 1 : 0) << '\n'
        << "abi.Mobile.scrController.IsScreenPointInsideUIElements1=" << (insideUi1 ? 1 : 0) << '\n'
        << "abi.Touch.EventSystem.current=" << (eventCurrent ? 1 : 0) << '\n'
        << "abi.Touch.EventSystem.RaycastAll=" << (raycastAll ? 1 : 0) << '\n'
        << "abi.Touch.PointerEventData.position=" << (pointerPosition ? 1 : 0) << '\n'
        << "abi.Touch.ListRaycastResult=" << (listRaycastResult ? 1 : 0) << '\n'
        << "abi.Touch.ListRaycastResult.Count=" << (listCount ? 1 : 0) << '\n'
        << "abi.Touch.ListRaycastResult.Clear=" << (listClear ? 1 : 0) << '\n'
        << "abi.Touch.Input.touchCount=" << (touchCount ? 1 : 0) << '\n'
        << "abi.Mobile.ADOBase.sceneName=" << (sceneName ? 1 : 0) << '\n'
        << "abi.FPS.Application.setTargetFrameRate1=" << (setTargetFrameRate1 ? 1 : 0) << '\n'
        << "abi.FPS.QualitySettings.setVSyncCount1=" << (setVSyncCount1 ? 1 : 0) << '\n'
        << "abi.Event.scrCamera.SetCustomFrameRateBoolInt="
        << (setCustomFrameRateBoolInt ? 1 : 0) << '\n'
        << "abi.Event.ffxCallMethod.methodName=" << (callMethodName ? 1 : 0) << '\n'
        << "abi.Event.ffxCallMethod.DecodeLevelEvent=" << (callMethodDecode ? 1 : 0) << '\n'
        << "abi.Event.ffxCallMethod.StartEffectPlanet=" << (callMethodStartEffect ? 1 : 0) << '\n'
        << "abi.Settings.PauseMenu.class=" << (pauseMenuClass ? 1 : 0) << '\n'
        << "abi.Settings.PauseMenu.ShowSettingsMenu0=" << (pauseMenuShowSettingsMenu0 ? 1 : 0) << '\n';

    {
        std::lock_guard<std::mutex> lock(g_reportMutex);
        g_report = out.str();
    }
    g_probeComplete.store(true, std::memory_order_release);
}

std::string CurrentReport() {
    if (!g_bnmLoadRequested.load(std::memory_order_acquire)) {
        return "nativeProbe=cache-post-bnm-probe-only-v2\n"
               "nativeStage=post-bnm-read-only-abi\n"
               "bnmLoadRequested=0\n"
               "bnmLoadedCallback=0\n"
               "probeComplete=0\n"
               "gameHooksInstalled=0\n"
               "sfbOpenFiltersHookInstalled=0\n"
               "sfbFilterMemoryRead=0\n"
               "sfbHookPolicy=disabled-unproven-call-abi\n";
    }
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire)) {
        return "nativeProbe=cache-post-bnm-probe-only-v2\n"
               "nativeStage=post-bnm-read-only-abi\n"
               "bnmLoadRequested=1\n"
               "bnmLoadedCallback=0\n"
               "probeComplete=0\n"
               "gameHooksInstalled=0\n"
               "sfbOpenFiltersHookInstalled=0\n"
               "sfbFilterMemoryRead=0\n"
               "sfbHookPolicy=disabled-unproven-call-abi\n";
    }
    std::lock_guard<std::mutex> lock(g_reportMutex);
    return g_report;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm == nullptr ||
        vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK ||
        env == nullptr) {
        return JNI_ERR;
    }
    g_bnmLoadRequested.store(true, std::memory_order_release);
    Loading::TryLoadByJNI(env);
    Loading::AddOnLoadedEvent([]() {
        g_bnmLoadedCallback.store(true, std::memory_order_release);
        RunReadOnlyAbiProbe();
    });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (env == nullptr) return nullptr;
    const std::string report = CurrentReport();
    return env->NewStringUTF(report.c_str());
}
