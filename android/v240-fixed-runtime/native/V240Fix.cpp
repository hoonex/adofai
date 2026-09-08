#include <jni.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>

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

void (*g_oldCanvasSetScaleFactor)(IL2CPP::Il2CppObject*, float) = nullptr;
float (*g_oldGetAxis)(String*) = nullptr;
float (*g_oldGetAxisRaw)(String*) = nullptr;
bool (*g_oldInsideUI)(IL2CPP::Il2CppObject*, Vector2) = nullptr;

Property<IL2CPP::Il2CppObject*> g_eventSystemCurrent;
Class g_pointerEventDataClass;
Class g_raycastResultClass;
Class g_listRaycastResultClass;
Method<void> g_raycastAll;
Property<Vector2> g_pointerPosition;
Property<int> g_listCount;

enum class PickerMode { Open, Save, Folder };

bool IsEditorScene() {
    if (!g_getSceneName.IsValid()) return false;
    String* name = g_getSceneName.Call();
    if (!name) return false;
    const std::string value = name->str();
    return value == "scnEditor" || value.rfind("scnEditor", 0) == 0;
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
    g_selectFile = env->GetStaticMethodID(local, "selectFile", "(Ljava/lang/String;)V");
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

std::string SuggestedName(String* value) {
    if (!value) return "level.adofai";
    std::string name = value->str();
    if (name.empty()) name = "level.adofai";
    if (name.size() < 7 || name.substr(name.size() - 7) != ".adofai") name += ".adofai";
    return name;
}

std::string RunPicker(PickerMode mode, String* suggestedName = nullptr) {
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
        jstring filter = env->NewStringUTF("adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg");
        env->CallStaticVoidMethod(g_selectorClass, g_selectFile, filter);
        env->DeleteLocalRef(filter);
    } else if (mode == PickerMode::Save) {
        std::string name = SuggestedName(suggestedName);
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

    // The Android picker owns the screen while this Unity call is suspended. Polling here
    // keeps the managed callback on the original Unity thread when the picker returns.
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

Array<String*>* ToArray(const std::string& path) {
    auto array = g_stringClass.NewArray<String*>(path.empty() ? 0 : 1);
    if (!array) return nullptr;
    if (!path.empty()) array->m_Items[0] = CreateMonoString(path);
    return array;
}

using StringArrayAction = Action<Array<String*>*>;
using StringAction = Action<String*>;

Array<String*>* HookOpenString(String*, String*, String*, bool) {
    return ToArray(RunPicker(PickerMode::Open));
}
Array<String*>* HookOpenFilters(String*, String*, void*, bool) {
    return ToArray(RunPicker(PickerMode::Open));
}
String* HookSaveString(String*, String*, String* defaultName, String*) {
    return CreateMonoString(RunPicker(PickerMode::Save, defaultName));
}
String* HookSaveFilters(String*, String*, String* defaultName, void*) {
    return CreateMonoString(RunPicker(PickerMode::Save, defaultName));
}
Array<String*>* HookFolder(String*, String*, bool) {
    return ToArray(RunPicker(PickerMode::Folder));
}
void HookOpenAsyncString(String*, String*, String*, bool, StringArrayAction* callback) {
    Array<String*>* value = ToArray(RunPicker(PickerMode::Open));
    if (callback) callback->Invoke(value);
}
void HookOpenAsyncFilters(String*, String*, void*, bool, StringArrayAction* callback) {
    Array<String*>* value = ToArray(RunPicker(PickerMode::Open));
    if (callback) callback->Invoke(value);
}
void HookSaveAsyncString(String*, String*, String* defaultName, String*, StringAction* callback) {
    String* value = CreateMonoString(RunPicker(PickerMode::Save, defaultName));
    if (callback) callback->Invoke(value);
}
void HookSaveAsyncFilters(String*, String*, String* defaultName, void*, StringAction* callback) {
    String* value = CreateMonoString(RunPicker(PickerMode::Save, defaultName));
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
    if (!axis) return false;
    const std::string value = axis->str();
    return value == "Mouse X" || value == "Mouse Y" || value == "Mouse ScrollWheel";
}
float HookGetAxis(String* axis) {
    float value = g_oldGetAxis ? g_oldGetAxis(axis) : 0.f;
    if (IsEditorScene() && IsDragAxis(axis)) value *= g_dragScale.load(std::memory_order_relaxed);
    return value;
}
float HookGetAxisRaw(String* axis) {
    float value = g_oldGetAxisRaw ? g_oldGetAxisRaw(axis) : 0.f;
    if (IsEditorScene() && IsDragAxis(axis)) value *= g_dragScale.load(std::memory_order_relaxed);
    return value;
}

bool RaycastUi(Vector2 point) {
    if (!g_eventSystemCurrent.IsValid() || !g_pointerEventDataClass || !g_listRaycastResultClass ||
        !g_raycastAll.IsValid() || !g_pointerPosition.IsValid() || !g_listCount.IsValid()) return false;
    IL2CPP::Il2CppObject* eventSystem = g_eventSystemCurrent.Get();
    if (!eventSystem) return false;
    IL2CPP::Il2CppObject* eventData = g_pointerEventDataClass.CreateNewObjectParameters(eventSystem);
    if (!eventData) return false;
    g_pointerPosition[eventData].Set(point);
    IL2CPP::Il2CppObject* results = g_listRaycastResultClass.CreateNewObjectParameters();
    if (!results) return false;
    g_raycastAll[eventSystem].Call(eventData, results);
    return g_listCount[results].Get() > 0;
}

bool HookInsideUI(IL2CPP::Il2CppObject* self, Vector2 point) {
    if (g_oldInsideUI && g_oldInsideUI(self, point)) return true;
    if (!IsEditorScene() || !g_touchAssist.load(std::memory_order_relaxed)) return false;
    if (RaycastUi(point)) return true;
    float scale = std::max(1.0f, g_touchScale.load(std::memory_order_relaxed));
    float radius = (scale - 1.0f) * 40.0f;
    if (radius < 1.0f) return false;
    return RaycastUi(Vector2(point.x + radius, point.y)) ||
           RaycastUi(Vector2(point.x - radius, point.y)) ||
           RaycastUi(Vector2(point.x, point.y + radius)) ||
           RaycastUi(Vector2(point.x, point.y - radius));
}

template <typename Fn>
void InstallNamedHook(Class& klass, const char* name, std::initializer_list<const char*> parameterNames,
                      Fn replacement, const char* label) {
    std::vector<std::string_view> names;
    names.reserve(parameterNames.size());
    for (const char* value : parameterNames) names.emplace_back(value);
    auto method = klass.GetMethod(name, names);
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
    if (g_listRaycastResultClass) g_listCount = g_listRaycastResultClass.GetProperty("Count");
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
        JNIEnv*, jclass, jfloat uiScale, jfloat touchScale, jfloat dragScale, jboolean touchAssist) {
    g_uiScale.store(std::fmax(0.70f, std::fmin(1.60f, uiScale)), std::memory_order_relaxed);
    g_touchScale.store(std::fmax(1.00f, std::fmin(2.00f, touchScale)), std::memory_order_relaxed);
    g_dragScale.store(std::fmax(0.50f, std::fmin(2.00f, dragScale)), std::memory_order_relaxed);
    g_touchAssist.store(touchAssist == JNI_TRUE, std::memory_order_relaxed);
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
