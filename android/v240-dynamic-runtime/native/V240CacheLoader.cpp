#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "universe.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

namespace {
constexpr const char* kLogTag = "ADOFAI.V240Cache";
constexpr char kPickerPathSeparator = '\x1f';
constexpr std::size_t kMaxExtensionFilters = 32;
constexpr std::size_t kMaxExtensionsPerFilter = 64;
constexpr int kPickerPollCount = 18000;
constexpr int kPickerPollMs = 50;

JavaVM* g_vm = nullptr;
jclass g_selectorClass = nullptr;
jmethodID g_selectFile = nullptr;
jmethodID g_getFilePath = nullptr;
jfieldID g_isDone = nullptr;
std::mutex g_pickerCallMutex;
Class g_stringClass;

std::atomic<bool> g_bnmLoadRequested{false};
std::atomic<bool> g_bnmLoadedCallback{false};
std::atomic<bool> g_probeComplete{false};
std::atomic<bool> g_sfbOpenFiltersHookInstalled{false};
std::mutex g_reportMutex;
std::string g_report =
        "nativeProbe=cache-post-bnm-narrow-fix\n"
        "nativeStage=post-bnm-narrow-sfb-fix\n"
        "bnmLoadRequested=0\n"
        "bnmLoadedCallback=0\n"
        "probeComplete=0\n"
        "gameHooksInstalled=0\n"
        "sfbOpenFiltersHookInstalled=0\n";

struct ExtensionFilterValue {
    String* Name;
    Array<String*>* Extensions;
};

void LogWarn(const char* message) {
    __android_log_print(ANDROID_LOG_WARN, kLogTag, "%s", message);
}

jclass LoadAppClass(JNIEnv* env, const char* slashName, const char* dotName) {
    jclass direct = env->FindClass(slashName);
    if (direct != nullptr && !env->ExceptionCheck()) return direct;
    if (env->ExceptionCheck()) env->ExceptionClear();

    jclass activityThread = env->FindClass("android/app/ActivityThread");
    if (activityThread == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        return nullptr;
    }
    jmethodID currentApplication = env->GetStaticMethodID(
            activityThread, "currentApplication", "()Landroid/app/Application;");
    if (currentApplication == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(activityThread);
        return nullptr;
    }
    jobject app = env->CallStaticObjectMethod(activityThread, currentApplication);
    if (app == nullptr || env->ExceptionCheck()) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(activityThread);
        return nullptr;
    }

    jclass appClass = env->GetObjectClass(app);
    jmethodID getClassLoader = appClass == nullptr ? nullptr :
            env->GetMethodID(appClass, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject loader = getClassLoader == nullptr ? nullptr :
            env->CallObjectMethod(app, getClassLoader);
    if (loader == nullptr || env->ExceptionCheck()) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (appClass != nullptr) env->DeleteLocalRef(appClass);
        env->DeleteLocalRef(app);
        env->DeleteLocalRef(activityThread);
        return nullptr;
    }

    jclass classLoaderClass = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClass = classLoaderClass == nullptr ? nullptr :
            env->GetMethodID(classLoaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    jclass result = nullptr;
    if (loadClass != nullptr) {
        jstring name = env->NewStringUTF(dotName);
        jobject loaded = name == nullptr ? nullptr : env->CallObjectMethod(loader, loadClass, name);
        if (name != nullptr) env->DeleteLocalRef(name);
        if (!env->ExceptionCheck() && loaded != nullptr) {
            result = reinterpret_cast<jclass>(loaded);
        } else if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    } else if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }

    if (classLoaderClass != nullptr) env->DeleteLocalRef(classLoaderClass);
    env->DeleteLocalRef(loader);
    if (appClass != nullptr) env->DeleteLocalRef(appClass);
    env->DeleteLocalRef(app);
    env->DeleteLocalRef(activityThread);
    return result;
}

bool InitSelector(JNIEnv* env) {
    if (g_selectorClass != nullptr) return true;
    jclass local = LoadAppClass(
            env, "com/unity3d/player/FileSelector", "com.unity3d.player.FileSelector");
    if (local == nullptr) {
        LogWarn("FileSelector class unavailable");
        return false;
    }

    jmethodID selectFile = env->GetStaticMethodID(
            local, "selectFile", "(Ljava/lang/String;Z)V");
    jmethodID getFilePath = env->GetStaticMethodID(
            local, "getFilePath", "()Ljava/lang/String;");
    jfieldID isDone = env->GetStaticFieldID(local, "isDone", "Z");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (selectFile == nullptr || getFilePath == nullptr || isDone == nullptr) {
        env->DeleteLocalRef(local);
        LogWarn("FileSelector ABI incomplete");
        return false;
    }

    jclass global = reinterpret_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    if (global == nullptr) return false;
    g_selectorClass = global;
    g_selectFile = selectFile;
    g_getFilePath = getFilePath;
    g_isDone = isDone;
    return true;
}

std::string NormalizeExtension(String* value) {
    if (value == nullptr) return "";
    std::string extension = value->str();
    while (!extension.empty() && (extension[0] == '.' || extension[0] == '*')) {
        extension.erase(extension.begin());
    }
    return extension;
}

void AppendUniqueExtension(std::vector<std::string>& values, String* extension) {
    std::string normalized = NormalizeExtension(extension);
    if (normalized.empty()) return;
    if (std::find(values.begin(), values.end(), normalized) == values.end()) {
        values.push_back(normalized);
    }
}

std::string PickerExtensions(void* rawFilters) {
    std::vector<std::string> values;
    if (rawFilters != nullptr) {
        auto* filters = reinterpret_cast<Array<ExtensionFilterValue>*>(rawFilters);
        const std::size_t filterCount = static_cast<std::size_t>(filters->capacity);
        if (filterCount > 0 && filterCount <= kMaxExtensionFilters) {
            for (std::size_t i = 0; i < filterCount; ++i) {
                Array<String*>* extensions = filters->m_Items[i].Extensions;
                if (extensions == nullptr) continue;
                const std::size_t count = static_cast<std::size_t>(extensions->capacity);
                if (count > kMaxExtensionsPerFilter) continue;
                for (std::size_t j = 0; j < count; ++j) {
                    AppendUniqueExtension(values, extensions->m_Items[j]);
                }
            }
        }
    }

    std::string joined;
    for (const std::string& value : values) {
        if (value.empty()) continue;
        if (!joined.empty()) joined += ',';
        joined += value;
    }
    return joined.empty() ? "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg" : joined;
}

std::string RunOpenPicker(const std::string& extensions, bool multiselect) {
    std::lock_guard<std::mutex> serialized(g_pickerCallMutex);
    if (g_vm == nullptr) return "";

    JNIEnv* env = nullptr;
    bool attached = false;
    jint state = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (state == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) return "";
        attached = true;
    } else if (state != JNI_OK || env == nullptr) {
        return "";
    }

    if (!InitSelector(env)) {
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    jstring filter = env->NewStringUTF(extensions.c_str());
    if (filter == nullptr) {
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    env->CallStaticVoidMethod(
            g_selectorClass, g_selectFile, filter, multiselect ? JNI_TRUE : JNI_FALSE);
    env->DeleteLocalRef(filter);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    bool done = false;
    for (int i = 0; i < kPickerPollCount; ++i) {
        jboolean value = env->GetStaticBooleanField(g_selectorClass, g_isDone);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            break;
        }
        if (value == JNI_TRUE) {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPickerPollMs));
    }

    std::string result;
    if (done) {
        jstring path = reinterpret_cast<jstring>(
                env->CallStaticObjectMethod(g_selectorClass, g_getFilePath));
        if (!env->ExceptionCheck() && path != nullptr) {
            const char* chars = env->GetStringUTFChars(path, nullptr);
            if (chars != nullptr) {
                result.assign(chars);
                env->ReleaseStringUTFChars(path, chars);
            }
            env->DeleteLocalRef(path);
        } else if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }

    if (attached) g_vm->DetachCurrentThread();
    return result;
}

std::vector<std::string> SplitPickerPaths(const std::string& encoded) {
    std::vector<std::string> paths;
    if (encoded.empty()) return paths;
    std::size_t start = 0;
    while (start <= encoded.size()) {
        const std::size_t end = encoded.find(kPickerPathSeparator, start);
        std::string value = encoded.substr(
                start, end == std::string::npos ? std::string::npos : end - start);
        if (!value.empty()) paths.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return paths;
}

Array<String*>* ToStringArray(const std::string& encodedPaths) {
    std::vector<std::string> paths = SplitPickerPaths(encodedPaths);
    auto array = g_stringClass.NewArray<String*>(paths.size());
    if (array == nullptr) return nullptr;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        array->m_Items[i] = CreateMonoString(paths[i]);
    }
    return array;
}

Array<String*>* HookOpenFilePanelFilters(
        String*, String*, void* filters, bool multiselect) {
    return ToStringArray(RunOpenPicker(PickerExtensions(filters), multiselect));
}

bool InstallExactOpenFiltersHook(Class& browser, bool filterLayoutCompatible) {
    if (!browser || !filterLayoutCompatible) return false;
    auto method = browser.GetMethod(
            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"});
    if (!method.IsValid()) return false;
    g_stringClass = Defaults::Get<String*>();
    if (!g_stringClass) return false;
    BasicHook(method, HookOpenFilePanelFilters, (void*)nullptr);
    g_sfbOpenFiltersHookInstalled.store(true, std::memory_order_release);
    return true;
}

void RunProbeAndInstallNarrowFix() {
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
    const bool filterLayoutCompatible = extensionFilter && filterNameField && filterExtensionsField;

    const bool setScaleFactor1 = canvasScaler && canvasScaler.GetMethod("SetScaleFactor", 1).IsValid();
    const bool getAxis1 = input && input.GetMethod("GetAxis", 1).IsValid();
    const bool getAxisRaw1 = input && input.GetMethod("GetAxisRaw", 1).IsValid();
    const bool insideUi1 = controller &&
            controller.GetMethod("IsScreenPointInsideUIElements", 1).IsValid();

    Class listRaycastResult = (genericList && raycastResult)
            ? genericList.GetGeneric({raycastResult.GetCompileTimeClass()}) : Class{};
    const bool eventCurrent = eventSystem && eventSystem.GetProperty("current").IsValid();
    const bool raycastAll = eventSystem && eventSystem.GetMethod("RaycastAll").IsValid();
    const bool pointerPosition = pointerEventData && pointerEventData.GetProperty("position").IsValid();
    const bool listCount = listRaycastResult && listRaycastResult.GetProperty("Count").IsValid();
    const bool listClear = listRaycastResult && listRaycastResult.GetMethod("Clear", 0).IsValid();
    const bool touchCount = input && input.GetMethod("get_touchCount", 0).IsValid();
    const bool sceneName = adoBase && adoBase.GetMethod("get_sceneName").IsValid();

    const bool setTargetFrameRate1 = application &&
            application.GetMethod("set_targetFrameRate", 1).IsValid();
    const bool setVSyncCount1 = qualitySettings &&
            qualitySettings.GetMethod("set_vSyncCount", 1).IsValid();
    const bool setCustomFrameRateBoolInt = scrCamera && scrCamera.GetMethod(
            "SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}).IsValid();
    const bool callMethodName = ffxCallMethod && ffxCallMethod.GetField("methodName").IsValid();
    const bool callMethodDecode = ffxCallMethod && levelEvent && ffxCallMethod.GetMethod(
            "Decode", {levelEvent.GetCompileTimeClass()}).IsValid();
    const bool callMethodStartEffect = ffxCallMethod && scrPlanet && ffxCallMethod.GetMethod(
            "StartEffect", {scrPlanet.GetCompileTimeClass()}).IsValid();

    const bool pauseMenuClass = static_cast<bool>(pauseMenu);
    const bool pauseMenuShowSettingsMenu0 = pauseMenu &&
            pauseMenu.GetMethod("ShowSettingsMenu", 0).IsValid();

    bool selectorReady = false;
    JNIEnv* env = nullptr;
    if (g_vm != nullptr &&
        g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK &&
        env != nullptr) {
        selectorReady = InitSelector(env);
    }

    const bool hookInstalled = selectorReady && openFiltersExact &&
            InstallExactOpenFiltersHook(browser, filterLayoutCompatible);

    std::ostringstream out;
    out << "nativeProbe=cache-post-bnm-narrow-fix\n"
        << "nativeStage=post-bnm-narrow-sfb-fix\n"
        << "bnmLoadRequested=1\n"
        << "bnmLoadedCallback=1\n"
        << "probeComplete=1\n"
        << "gameHooksInstalled=" << (hookInstalled ? 1 : 0) << '\n'
        << "sfbOpenFiltersHookInstalled=" << (hookInstalled ? 1 : 0) << '\n'
        << "abi.SFB.class=" << (browser ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel4=" << (openFile4 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.filtersExact=" << (openFiltersExact ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.class=" << (extensionFilter ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.Name=" << (filterNameField ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.Extensions=" << (filterExtensionsField ? 1 : 0) << '\n'
        << "abi.SFB.FileSelector.ready=" << (selectorReady ? 1 : 0) << '\n'
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
        << "abi.Event.scrCamera.SetCustomFrameRateBoolInt=" << (setCustomFrameRateBoolInt ? 1 : 0) << '\n'
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
        return "nativeProbe=cache-post-bnm-narrow-fix\n"
               "nativeStage=post-bnm-narrow-sfb-fix\n"
               "bnmLoadRequested=0\n"
               "bnmLoadedCallback=0\n"
               "probeComplete=0\n"
               "gameHooksInstalled=0\n"
               "sfbOpenFiltersHookInstalled=0\n";
    }
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire)) {
        return "nativeProbe=cache-post-bnm-narrow-fix\n"
               "nativeStage=post-bnm-narrow-sfb-fix\n"
               "bnmLoadRequested=1\n"
               "bnmLoadedCallback=0\n"
               "probeComplete=0\n"
               "gameHooksInstalled=0\n"
               "sfbOpenFiltersHookInstalled=0\n";
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
    g_vm = vm;
    g_bnmLoadRequested.store(true, std::memory_order_release);
    Loading::TryLoadByJNI(env);
    Loading::AddOnLoadedEvent([]() {
        g_bnmLoadedCallback.store(true, std::memory_order_release);
        RunProbeAndInstallNarrowFix();
    });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void*) {
    if (vm == nullptr || g_selectorClass == nullptr) return;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK && env != nullptr) {
        env->DeleteGlobalRef(g_selectorClass);
    }
    g_selectorClass = nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (env == nullptr) return nullptr;
    const std::string report = CurrentReport();
    return env->NewStringUTF(report.c_str());
}
