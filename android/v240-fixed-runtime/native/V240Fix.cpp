#include <jni.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "universe.h"
#include "Logger.h"

using namespace BNM;
using namespace BNM::Structures::Mono;
using namespace BNM::Structures::Unity;

namespace {
JavaVM* g_vm = nullptr;

jclass g_selectorClass = nullptr;
jmethodID g_selectFile = nullptr;
jmethodID g_saveAs = nullptr;
jmethodID g_selectFolder = nullptr;
jmethodID g_getFilePath = nullptr;
jmethodID g_getFolderPath = nullptr;
jfieldID g_isDone = nullptr;
std::mutex g_pickerCallMutex;

Class g_stringClass;
Method<String*> g_getSceneName;

std::atomic<float> g_uiScale{1.15f};
std::atomic<float> g_touchScale{1.25f};
std::atomic<float> g_dragScale{1.0f};
std::atomic<bool> g_touchAssist{true};
std::atomic<int> g_targetFps{0};
std::atomic<bool> g_unlockFps{false};
std::atomic<bool> g_lowLatency{false};
std::atomic<bool> g_framePolicyConfigured{false};
std::atomic<int> g_lastRequestedFps{-1};
std::atomic<int> g_lastRequestedVSync{1};

constexpr int64_t kEditorSceneCacheNs = 250000000LL;
constexpr char kPickerPathSeparator = '\x1f';
constexpr std::size_t kMaxExtensionFilters = 32;
constexpr std::size_t kMaxExtensionsPerFilter = 64;
std::atomic<int64_t> g_editorSceneCacheAtNs{0};
std::atomic<bool> g_editorSceneCacheValue{false};

void (*g_oldCanvasSetScaleFactor)(IL2CPP::Il2CppObject*, float) = nullptr;
float (*g_oldGetAxis)(String*) = nullptr;
float (*g_oldGetAxisRaw)(String*) = nullptr;
bool (*g_oldInsideUI)(IL2CPP::Il2CppObject*, Vector2) = nullptr;
void (*g_oldSetTargetFrameRate)(int) = nullptr;
void (*g_oldSetVSyncCount)(int) = nullptr;

Property<IL2CPP::Il2CppObject*> g_eventSystemCurrent;
Class g_pointerEventDataClass;
Class g_raycastResultClass;
Class g_listRaycastResultClass;
Method<void> g_raycastAll;
Method<void> g_listClear;
Property<Vector2> g_pointerPosition;
Property<int> g_listCount;

enum class PickerMode { Open, Save, Folder };

// SFB.ExtensionFilter is a value type with exactly two managed-reference fields in the
// upstream package used by ADOFAI: string Name; string[] Extensions.
struct ExtensionFilterValue {
    String* Name;
    Array<String*>* Extensions;
};

int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool MonoStringEqualsAscii(const String* value, const char* ascii, int length) {
    if (!value || !ascii || value->length != length) return false;
    for (int i = 0; i < length; ++i) {
        const auto expected = static_cast<IL2CPP::Il2CppChar>(static_cast<unsigned char>(ascii[i]));
        if (value->chars[i] != expected) return false;
    }
    return true;
}

bool MonoStringStartsWithAscii(const String* value, const char* ascii, int length) {
    if (!value || !ascii || value->length < length) return false;
    for (int i = 0; i < length; ++i) {
        const auto expected = static_cast<IL2CPP::Il2CppChar>(static_cast<unsigned char>(ascii[i]));
        if (value->chars[i] != expected) return false;
    }
    return true;
}

bool IsEditorScene() {
    const int64_t now = SteadyNowNs();
    const int64_t cachedAt = g_editorSceneCacheAtNs.load(std::memory_order_acquire);
    if (cachedAt > 0 && now >= cachedAt && now - cachedAt < kEditorSceneCacheNs) {
        return g_editorSceneCacheValue.load(std::memory_order_relaxed);
    }

    bool isEditor = false;
    if (g_getSceneName.IsValid()) {
        isEditor = MonoStringStartsWithAscii(g_getSceneName.Call(), "scnEditor", 9);
    }
    g_editorSceneCacheValue.store(isEditor, std::memory_order_relaxed);
    g_editorSceneCacheAtNs.store(now, std::memory_order_release);
    return isEditor;
}

jclass LoadAppClass(JNIEnv* env, const char* slashName, const char* dotName) {
    jclass direct = env->FindClass(slashName);
    if (direct && !env->ExceptionCheck()) return direct;
    if (env->ExceptionCheck()) env->ExceptionClear();

    jclass activityThread = env->FindClass("android/app/ActivityThread");
    if (!activityThread) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }
    jmethodID currentApplication = env->GetStaticMethodID(
            activityThread, "currentApplication", "()Landroid/app/Application;");
    if (!currentApplication) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }
    jobject app = env->CallStaticObjectMethod(activityThread, currentApplication);
    if (!app || env->ExceptionCheck()) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }

    jclass appClass = env->GetObjectClass(app);
    jmethodID getClassLoader = env->GetMethodID(appClass, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject loader = getClassLoader ? env->CallObjectMethod(app, getClassLoader) : nullptr;
    if (!loader || env->ExceptionCheck()) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }

    jclass classLoaderClass = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClass = classLoaderClass
            ? env->GetMethodID(classLoaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;")
            : nullptr;
    if (!loadClass) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }
    jstring name = env->NewStringUTF(dotName);
    jobject clazz = env->CallObjectMethod(loader, loadClass, name);
    env->DeleteLocalRef(name);
    if (!clazz || env->ExceptionCheck()) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }
    return reinterpret_cast<jclass>(clazz);
}

bool InitSelector(JNIEnv* env) {
    if (g_selectorClass) return true;
    jclass local = LoadAppClass(env, "com/unity3d/player/FileSelector", "com.unity3d.player.FileSelector");
    if (!local) {
        LOGE("V240: FileSelector class unavailable");
        return false;
    }
    g_selectFile = env->GetStaticMethodID(local, "selectFile", "(Ljava/lang/String;Z)V");
    g_saveAs = env->GetStaticMethodID(local, "saveAs", "(Ljava/lang/String;)V");
    g_selectFolder = env->GetStaticMethodID(local, "selectFolder", "()V");
    g_getFilePath = env->GetStaticMethodID(local, "getFilePath", "()Ljava/lang/String;");
    g_getFolderPath = env->GetStaticMethodID(local, "getFolderPath", "()Ljava/lang/String;");
    g_isDone = env->GetStaticFieldID(local, "isDone", "Z");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (!g_selectFile || !g_saveAs || !g_selectFolder || !g_getFilePath || !g_getFolderPath || !g_isDone) {
        LOGE("V240: FileSelector ABI incomplete");
        return false;
    }
    g_selectorClass = reinterpret_cast<jclass>(env->NewGlobalRef(local));
    LOGD("V240: FileSelector bridge initialized");
    return g_selectorClass != nullptr;
}

std::string NormalizeExtension(String* value) {
    if (!value) return "";
    std::string extension = value->str();
    while (!extension.empty() && (extension[0] == '.' || extension[0] == '*')) {
        extension.erase(extension.begin());
    }
    return extension;
}

void AppendUniqueExtension(std::vector<std::string>& values, String* extension) {
    std::string normalized = NormalizeExtension(extension);
    if (normalized.empty()) return;
    if (std::find(values.begin(), values.end(), normalized) == values.end()) values.push_back(normalized);
}

std::vector<std::string> ReadFilterExtensions(void* rawFilters) {
    std::vector<std::string> values;
    if (!rawFilters) return values;
    auto* filters = reinterpret_cast<Array<ExtensionFilterValue>*>(rawFilters);
    const std::size_t filterCount = static_cast<std::size_t>(filters->capacity);
    if (filterCount == 0 || filterCount > kMaxExtensionFilters) return values;
    for (std::size_t i = 0; i < filterCount; ++i) {
        Array<String*>* extensions = filters->m_Items[i].Extensions;
        if (!extensions) continue;
        const std::size_t count = static_cast<std::size_t>(extensions->capacity);
        if (count > kMaxExtensionsPerFilter) continue;
        for (std::size_t j = 0; j < count; ++j) AppendUniqueExtension(values, extensions->m_Items[j]);
    }
    return values;
}

std::string JoinExtensions(const std::vector<std::string>& values) {
    std::string joined;
    for (const std::string& value : values) {
        if (value.empty()) continue;
        if (!joined.empty()) joined += ',';
        joined += value;
    }
    return joined;
}

std::string PickerExtensions(String* extensions) {
    if (!extensions) return "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg";
    std::string value = extensions->str();
    if (value.empty()) return "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg";
    return value;
}

std::string PickerExtensions(void* filters) {
    std::string value = JoinExtensions(ReadFilterExtensions(filters));
    return value.empty() ? "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg" : value;
}

std::string FirstFilterExtension(void* filters) {
    std::vector<std::string> values = ReadFilterExtensions(filters);
    return values.empty() ? "" : values.front();
}

std::string SuggestedName(String* value, const std::string& extension) {
    std::string name = value ? value->str() : "";
    if (name.empty()) name = "level";

    std::string ext = extension;
    while (!ext.empty() && (ext[0] == '.' || ext[0] == '*')) ext.erase(ext.begin());
    if (ext.empty()) {
        const std::size_t slash = name.find_last_of("/\\");
        const std::size_t dot = name.find_last_of('.');
        if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) return name;
        ext = "adofai";
    }
    const std::string suffix = "." + ext;
    if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        name += suffix;
    }
    return name;
}

std::string RunPicker(PickerMode mode,
                      String* suggestedName = nullptr,
                      const std::string& extensions = "",
                      bool multiselect = false) {
    std::lock_guard<std::mutex> serialized(g_pickerCallMutex);
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!g_vm) return "";
    jint state = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (state == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return "";
        attached = true;
    } else if (state != JNI_OK || !env) {
        return "";
    }
    if (!InitSelector(env)) {
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    if (mode == PickerMode::Open) {
        const std::string filterValue = extensions.empty()
                ? "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg"
                : extensions;
        jstring filter = env->NewStringUTF(filterValue.c_str());
        env->CallStaticVoidMethod(g_selectorClass, g_selectFile, filter,
                                  multiselect ? JNI_TRUE : JNI_FALSE);
        env->DeleteLocalRef(filter);
    } else if (mode == PickerMode::Save) {
        std::string name = SuggestedName(suggestedName, extensions);
        jstring jname = env->NewStringUTF(name.c_str());
        env->CallStaticVoidMethod(g_selectorClass, g_saveAs, jname);
        env->DeleteLocalRef(jname);
    } else {
        env->CallStaticVoidMethod(g_selectorClass, g_selectFolder);
    }
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    bool done = false;
    for (int i = 0; i < 18000; ++i) {
        jboolean value = env->GetStaticBooleanField(g_selectorClass, g_isDone);
        if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
        if (value == JNI_TRUE) { done = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::string result;
    if (done) {
        jmethodID getter = mode == PickerMode::Folder ? g_getFolderPath : g_getFilePath;
        jstring path = reinterpret_cast<jstring>(env->CallStaticObjectMethod(g_selectorClass, getter));
        if (!env->ExceptionCheck() && path) {
            const char* chars = env->GetStringUTFChars(path, nullptr);
            if (chars) {
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
        std::size_t end = encoded.find(kPickerPathSeparator, start);
        std::string value = encoded.substr(start,
                end == std::string::npos ? std::string::npos : end - start);
        if (!value.empty()) paths.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return paths;
}

Array<String*>* ToArray(const std::string& encodedPaths) {
    std::vector<std::string> paths = SplitPickerPaths(encodedPaths);
    auto array = g_stringClass.NewArray<String*>(paths.size());
    if (!array) return nullptr;
    for (std::size_t i = 0; i < paths.size(); ++i) array->m_Items[i] = CreateMonoString(paths[i]);
    return array;
}

using StringArrayAction = Action<Array<String*>*>;
using StringAction = Action<String*>;

Array<String*>* HookOpenString(String*, String*, String* extension, bool multiselect) {
    return ToArray(RunPicker(PickerMode::Open, nullptr, PickerExtensions(extension), multiselect));
}
Array<String*>* HookOpenFilters(String*, String*, void* filters, bool multiselect) {
    return ToArray(RunPicker(PickerMode::Open, nullptr, PickerExtensions(filters), multiselect));
}
String* HookSaveString(String*, String*, String* defaultName, String* extension) {
    return CreateMonoString(RunPicker(PickerMode::Save, defaultName, NormalizeExtension(extension)));
}
String* HookSaveFilters(String*, String*, String* defaultName, void* filters) {
    return CreateMonoString(RunPicker(PickerMode::Save, defaultName, FirstFilterExtension(filters)));
}
Array<String*>* HookFolder(String*, String*, bool) {
    return ToArray(RunPicker(PickerMode::Folder));
}
void HookOpenAsyncString(String*, String*, String* extension, bool multiselect, StringArrayAction* callback) {
    Array<String*>* value = ToArray(RunPicker(PickerMode::Open, nullptr, PickerExtensions(extension), multiselect));
    if (callback) callback->Invoke(value);
}
void HookOpenAsyncFilters(String*, String*, void* filters, bool multiselect, StringArrayAction* callback) {
    Array<String*>* value = ToArray(RunPicker(PickerMode::Open, nullptr, PickerExtensions(filters), multiselect));
    if (callback) callback->Invoke(value);
}
void HookSaveAsyncString(String*, String*, String* defaultName, String* extension, StringAction* callback) {
    String* value = CreateMonoString(RunPicker(PickerMode::Save, defaultName, NormalizeExtension(extension)));
    if (callback) callback->Invoke(value);
}
void HookSaveAsyncFilters(String*, String*, String* defaultName, void* filters, StringAction* callback) {
    String* value = CreateMonoString(RunPicker(PickerMode::Save, defaultName, FirstFilterExtension(filters)));
    if (callback) callback->Invoke(value);
}
void HookFolderAsync(String*, String*, bool, StringArrayAction* callback) {
    Array<String*>* value = ToArray(RunPicker(PickerMode::Folder));
    if (callback) callback->Invoke(value);
}

void HookCanvasSetScaleFactor(IL2CPP::Il2CppObject* self, float factor) {
    if (g_oldCanvasSetScaleFactor) {
        if (IsEditorScene()) factor *= g_uiScale.load(std::memory_order_relaxed);
        g_oldCanvasSetScaleFactor(self, factor);
    }
}

bool IsDragAxis(String* axis) {
    return MonoStringEqualsAscii(axis, "Mouse X", 7) ||
           MonoStringEqualsAscii(axis, "Mouse Y", 7) ||
           MonoStringEqualsAscii(axis, "Mouse ScrollWheel", 17);
}
float HookGetAxis(String* axis) {
    float value = g_oldGetAxis ? g_oldGetAxis(axis) : 0.f;
    if (IsDragAxis(axis) && IsEditorScene()) value *= g_dragScale.load(std::memory_order_relaxed);
    return value;
}
float HookGetAxisRaw(String* axis) {
    float value = g_oldGetAxisRaw ? g_oldGetAxisRaw(axis) : 0.f;
    if (IsDragAxis(axis) && IsEditorScene()) value *= g_dragScale.load(std::memory_order_relaxed);
    return value;
}

bool RaycastUiWithContext(IL2CPP::Il2CppObject* eventSystem,
                          IL2CPP::Il2CppObject* eventData,
                          IL2CPP::Il2CppObject* results,
                          Vector2 point,
                          bool clearResults) {
    if (clearResults) g_listClear[results].Call();
    g_pointerPosition[eventData].Set(point);
    g_raycastAll[eventSystem].Call(eventData, results);
    return g_listCount[results].Get() > 0;
}

bool ExpandedUiHit(Vector2 point, float radius) {
    if (!g_eventSystemCurrent.IsValid() || !g_pointerEventDataClass || !g_listRaycastResultClass ||
        !g_raycastAll.IsValid() || !g_listClear.IsValid() ||
        !g_pointerPosition.IsValid() || !g_listCount.IsValid()) return false;
    IL2CPP::Il2CppObject* eventSystem = g_eventSystemCurrent.Get();
    if (!eventSystem) return false;
    IL2CPP::Il2CppObject* eventData = g_pointerEventDataClass.CreateNewObjectParameters(eventSystem);
    if (!eventData) return false;
    IL2CPP::Il2CppObject* results = g_listRaycastResultClass.CreateNewObjectParameters();
    if (!results) return false;

    if (RaycastUiWithContext(eventSystem, eventData, results, point, false)) return true;
    if (radius < 1.0f) return false;
    return RaycastUiWithContext(eventSystem, eventData, results, Vector2(point.x + radius, point.y), true) ||
           RaycastUiWithContext(eventSystem, eventData, results, Vector2(point.x - radius, point.y), true) ||
           RaycastUiWithContext(eventSystem, eventData, results, Vector2(point.x, point.y + radius), true) ||
           RaycastUiWithContext(eventSystem, eventData, results, Vector2(point.x, point.y - radius), true);
}

bool HookInsideUI(IL2CPP::Il2CppObject* self, Vector2 point) {
    if (g_oldInsideUI && g_oldInsideUI(self, point)) return true;
    if (!g_touchAssist.load(std::memory_order_relaxed) || !IsEditorScene()) return false;
    float scale = std::max(1.0f, g_touchScale.load(std::memory_order_relaxed));
    float radius = (scale - 1.0f) * 40.0f;
    return ExpandedUiHit(point, radius);
}

void HookSetTargetFrameRate(int fps) {
    g_lastRequestedFps.store(fps, std::memory_order_relaxed);
    if (!g_oldSetTargetFrameRate) return;
    if (!g_framePolicyConfigured.load(std::memory_order_acquire)) {
        g_oldSetTargetFrameRate(fps);
        return;
    }
    int target = g_targetFps.load(std::memory_order_relaxed);
    if (g_unlockFps.load(std::memory_order_relaxed) && target > 0) {
        g_oldSetTargetFrameRate(target);
    } else {
        g_oldSetTargetFrameRate(fps);
    }
}

void HookSetVSyncCount(int count) {
    g_lastRequestedVSync.store(count, std::memory_order_relaxed);
    if (!g_oldSetVSyncCount) return;
    if (!g_framePolicyConfigured.load(std::memory_order_acquire)) {
        g_oldSetVSyncCount(count);
        return;
    }
    if (g_lowLatency.load(std::memory_order_relaxed)) g_oldSetVSyncCount(0);
    else g_oldSetVSyncCount(count);
}

void ApplyFramePolicy() {
    if (!g_framePolicyConfigured.load(std::memory_order_acquire)) return;
    if (g_oldSetTargetFrameRate) {
        int requested = g_lastRequestedFps.load(std::memory_order_relaxed);
        int target = g_targetFps.load(std::memory_order_relaxed);
        if (g_unlockFps.load(std::memory_order_relaxed) && target > 0) {
            g_oldSetTargetFrameRate(target);
        } else {
            g_oldSetTargetFrameRate(requested);
        }
    }
    if (g_oldSetVSyncCount) {
        int requested = g_lastRequestedVSync.load(std::memory_order_relaxed);
        g_oldSetVSyncCount(g_lowLatency.load(std::memory_order_relaxed) ? 0 : requested);
    }
}

template <typename Fn>
void InstallNamedHook(
        Class& klass,
        const char* name,
        const std::initializer_list<std::string_view>& parameterNames,
        Fn replacement,
        const char* label) {
    auto method = klass.GetMethod(name, parameterNames);
    if (!method.IsValid()) {
        LOGW("V240: method missing: %s", label);
        return;
    }
    BasicHook(method, replacement, (void*)nullptr);
    LOGD("V240: hooked %s", label);
}

void InstallSfbHooks() {
    Class browser("SFB", "StandaloneFileBrowser");
    if (!browser) {
        LOGE("V240: SFB.StandaloneFileBrowser not found");
        return;
    }
    InstallNamedHook(browser, "OpenFilePanel", {"title","directory","extension","multiselect"}, HookOpenString, "OpenFilePanel(string)");
    InstallNamedHook(browser, "OpenFilePanel", {"title","directory","extensions","multiselect"}, HookOpenFilters, "OpenFilePanel(filters)");
    InstallNamedHook(browser, "SaveFilePanel", {"title","directory","defaultName","extension"}, HookSaveString, "SaveFilePanel(string)");
    InstallNamedHook(browser, "SaveFilePanel", {"title","directory","defaultName","extensions"}, HookSaveFilters, "SaveFilePanel(filters)");
    InstallNamedHook(browser, "OpenFolderPanel", {"title","directory","multiselect"}, HookFolder, "OpenFolderPanel");
    InstallNamedHook(browser, "OpenFilePanelAsync", {"title","directory","extension","multiselect","cb"}, HookOpenAsyncString, "OpenFilePanelAsync(string)");
    InstallNamedHook(browser, "OpenFilePanelAsync", {"title","directory","extensions","multiselect","cb"}, HookOpenAsyncFilters, "OpenFilePanelAsync(filters)");
    InstallNamedHook(browser, "SaveFilePanelAsync", {"title","directory","defaultName","extension","cb"}, HookSaveAsyncString, "SaveFilePanelAsync(string)");
    InstallNamedHook(browser, "SaveFilePanelAsync", {"title","directory","defaultName","extensions","cb"}, HookSaveAsyncFilters, "SaveFilePanelAsync(filters)");
    InstallNamedHook(browser, "OpenFolderPanelAsync", {"title","directory","multiselect","cb"}, HookFolderAsync, "OpenFolderPanelAsync");
}

void InstallMobileHooks() {
    g_getSceneName = Class("", "ADOBase").GetMethod("get_sceneName");
    g_stringClass = Defaults::Get<String*>();

    Class canvasScaler("UnityEngine.UI", "CanvasScaler");
    auto setScale = canvasScaler.GetMethod("SetScaleFactor", 1);
    if (setScale.IsValid()) BasicHook(setScale, HookCanvasSetScaleFactor, g_oldCanvasSetScaleFactor);
    else LOGW("V240: CanvasScaler.SetScaleFactor missing");

    Class input("UnityEngine", "Input");
    auto getAxis = input.GetMethod("GetAxis", 1);
    if (getAxis.IsValid()) BasicHook(getAxis, HookGetAxis, g_oldGetAxis);
    auto getAxisRaw = input.GetMethod("GetAxisRaw", 1);
    if (getAxisRaw.IsValid()) BasicHook(getAxisRaw, HookGetAxisRaw, g_oldGetAxisRaw);

    Class controller("", "scrController");
    auto inside = controller.GetMethod("IsScreenPointInsideUIElements", 1);
    if (inside.IsValid()) BasicHook(inside, HookInsideUI, g_oldInsideUI);
    else LOGW("V240: scrController.IsScreenPointInsideUIElements missing");

    Class eventSystem("UnityEngine.EventSystems", "EventSystem");
    g_eventSystemCurrent = eventSystem.GetProperty("current");
    g_pointerEventDataClass = Class("UnityEngine.EventSystems", "PointerEventData");
    g_raycastResultClass = Class("UnityEngine.EventSystems", "RaycastResult");
    Class list("System.Collections.Generic", "List`1");
    if (g_raycastResultClass) g_listRaycastResultClass = list.GetGeneric({g_raycastResultClass.GetCompileTimeClass()});
    g_raycastAll = eventSystem.GetMethod("RaycastAll");
    g_pointerPosition = g_pointerEventDataClass.GetProperty("position");
    if (g_listRaycastResultClass) {
        g_listCount = g_listRaycastResultClass.GetProperty("Count");
        g_listClear = g_listRaycastResultClass.GetMethod("Clear", 0);
    }

    Class application("UnityEngine", "Application");
    auto setTargetFrameRate = application.GetMethod("set_targetFrameRate", 1);
    if (setTargetFrameRate.IsValid()) {
        BasicHook(setTargetFrameRate, HookSetTargetFrameRate, g_oldSetTargetFrameRate);
        LOGD("V240: hooked Application.targetFrameRate");
    } else {
        LOGW("V240: Application.set_targetFrameRate missing");
    }

    Class qualitySettings("UnityEngine", "QualitySettings");
    auto setVSyncCount = qualitySettings.GetMethod("set_vSyncCount", 1);
    if (setVSyncCount.IsValid()) {
        BasicHook(setVSyncCount, HookSetVSyncCount, g_oldSetVSyncCount);
        LOGD("V240: hooked QualitySettings.vSyncCount");
    } else {
        LOGW("V240: QualitySettings.set_vSyncCount missing");
    }

    ApplyFramePolicy();
}

void InstallAllHooks() {
    JNIEnv* env = nullptr;
    if (g_vm && g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK && env) InitSelector(env);
    InstallSfbHooks();
    InstallMobileHooks();
    LOGD("V240: fixed runtime hooks installed");
}
} // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeApply(
        JNIEnv*, jclass,
        jfloat uiScale,
        jfloat touchScale,
        jfloat dragScale,
        jboolean touchAssist,
        jint targetFps,
        jboolean unlockFps,
        jboolean lowLatency) {
    g_uiScale.store(std::fmax(0.70f, std::fmin(1.60f, uiScale)), std::memory_order_relaxed);
    g_touchScale.store(std::fmax(1.00f, std::fmin(2.00f, touchScale)), std::memory_order_relaxed);
    g_dragScale.store(std::fmax(0.50f, std::fmin(2.00f, dragScale)), std::memory_order_relaxed);
    g_touchAssist.store(touchAssist == JNI_TRUE, std::memory_order_relaxed);
    int fps = targetFps <= 0 ? 0 : std::max(30, std::min(240, static_cast<int>(targetFps)));
    g_targetFps.store(fps, std::memory_order_relaxed);
    g_unlockFps.store(unlockFps == JNI_TRUE, std::memory_order_relaxed);
    g_lowLatency.store(lowLatency == JNI_TRUE, std::memory_order_relaxed);
    g_framePolicyConfigured.store(true, std::memory_order_release);
    ApplyFramePolicy();
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (!vm || vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) return JNI_ERR;
    g_vm = vm;
    StartLogging();
    InitSelector(env);
    Loading::TryLoadByJNI(env);
    Loading::AddOnLoadedEvent([]() { InstallAllHooks(); });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    StopLogging();
}
