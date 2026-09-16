#include <jni.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "universe.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

namespace {
JavaVM* g_vm = nullptr;
jclass g_selectorClass = nullptr;
jmethodID g_selectFile = nullptr;
jmethodID g_getFilePath = nullptr;
jfieldID g_isDone = nullptr;
Class g_stringClass;
std::mutex g_selectorMutex;
std::mutex g_pickerMutex;
std::mutex g_reportMutex;

std::atomic<bool> g_bnmLoadRequested{false};
std::atomic<bool> g_bnmLoadedCallback{false};
std::atomic<bool> g_probeComplete{false};
std::atomic<bool> g_hookInstalled{false};
std::atomic<bool> g_markerReady{false};
std::atomic<bool> g_installAttempted{false};
std::atomic<bool> g_safBridgeReady{false};
std::atomic<int> g_recoveryState{0};
std::atomic<int> g_calls{0};
std::atomic<int> g_returns{0};
std::atomic<int> g_callsInFlight{0};
std::atomic<int> g_markerWriteFailures{0};
std::atomic<int> g_safPickerCalls{0};
std::atomic<int> g_safPickerReturns{0};
std::atomic<int> g_safLastState{0};
std::string g_installMarker;
std::string g_callMarker;
std::string g_report =
        "nativeProbe=cache-post-bnm-sfb-saf-v1\n"
        "nativeStage=post-bnm-sfb-saf-open\n"
        "abiProbeRevision=6\n"
        "bnmLoadRequested=0\n"
        "bnmLoadedCallback=0\n"
        "probeComplete=0\n"
        "gameHooksInstalled=0\n"
        "sfbOpenFiltersHookInstalled=0\n"
        "sfbFilterMemoryRead=0\n"
        "sfbHookPolicy=bootstrap1-self-fused-saf-broad-open\n"
        "sfbCanarySelfFuse=1\n"
        "sfbCanaryMarkerReady=0\n"
        "sfbCanaryRecoveryState=0\n"
        "sfbCanaryAbiGuard=0\n"
        "sfbSafBridgeReady=0\n"
        "sfbOriginalCallUsed=0\n";

struct ExtensionFilterValue {
    String* Name;
    Array<String*>* Extensions;
};
static_assert(sizeof(ExtensionFilterValue) == sizeof(void*) * 2,
              "ExtensionFilter payload must be two managed references");

using OpenFiltersFn = Array<String*>* (*)(
        String*, String*, Array<ExtensionFilterValue>*, bool, IL2CPP::MethodInfo*);
OpenFiltersFn g_oldOpenFilters = nullptr;

bool SameClass(const Class& a, const Class& b) {
    return a && b && a.GetClass() == b.GetClass();
}
int TypeCode(const IL2CPP::Il2CppType* type) {
    return type == nullptr ? -1 : static_cast<int>(type->type);
}
int TypeByRef(const IL2CPP::Il2CppType* type) {
    return type == nullptr ? -1 : (type->byref ? 1 : 0);
}
int TypeValueType(const IL2CPP::Il2CppType* type) {
    return type == nullptr ? -1 : (type->valuetype ? 1 : 0);
}

std::string RuntimeDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&RuntimeDir), &info) == 0 || info.dli_fname == nullptr) return "";
    std::string path(info.dli_fname);
    const std::size_t slash = path.find_last_of('/');
    return slash == std::string::npos || slash == 0 ? "" : path.substr(0, slash);
}
bool MarkerExists(const std::string& path) {
    return !path.empty() && access(path.c_str(), F_OK) == 0;
}
bool WriteMarker(const std::string& path) {
    if (path.empty()) return false;
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    static constexpr char pending[] = "pending\n";
    const ssize_t written = write(fd, pending, sizeof(pending) - 1);
    const bool ok = written == static_cast<ssize_t>(sizeof(pending) - 1) && fsync(fd) == 0;
    close(fd);
    if (!ok) unlink(path.c_str());
    return ok;
}
void ClearMarker(const std::string& path) {
    if (!path.empty()) unlink(path.c_str());
}
bool PrepareSelfFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_recoveryState.store(3); return false; }
    g_installMarker = dir + "/sfb-canary-r6-install.pending";
    g_callMarker = dir + "/sfb-canary-r6-call.pending";
    g_markerReady.store(true);
    if (MarkerExists(g_installMarker)) { g_recoveryState.store(1); return false; }
    if (MarkerExists(g_callMarker)) { g_recoveryState.store(2); return false; }
    const std::string probe = dir + "/sfb-canary-r6-marker-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) { g_markerReady.store(false); g_recoveryState.store(3); return false; }
    ClearMarker(probe);
    return true;
}

bool GetEnv(JNIEnv** env, bool* attached) {
    if (g_vm == nullptr || env == nullptr || attached == nullptr) return false;
    *env = nullptr;
    *attached = false;
    const jint state = g_vm->GetEnv(reinterpret_cast<void**>(env), JNI_VERSION_1_6);
    if (state == JNI_OK && *env != nullptr) return true;
    if (state != JNI_EDETACHED) return false;
    if (g_vm->AttachCurrentThread(env, nullptr) != JNI_OK || *env == nullptr) return false;
    *attached = true;
    return true;
}

jclass LoadAppClass(JNIEnv* env) {
    jclass direct = env->FindClass("com/unity3d/player/FileSelector");
    if (direct != nullptr && !env->ExceptionCheck()) return direct;
    if (env->ExceptionCheck()) env->ExceptionClear();

    jclass threadClass = env->FindClass("android/app/ActivityThread");
    if (threadClass == nullptr) { if (env->ExceptionCheck()) env->ExceptionClear(); return nullptr; }
    jmethodID currentApplication = env->GetStaticMethodID(
            threadClass, "currentApplication", "()Landroid/app/Application;");
    jobject app = currentApplication == nullptr ? nullptr : env->CallStaticObjectMethod(threadClass, currentApplication);
    if (app == nullptr || env->ExceptionCheck()) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(threadClass);
        return nullptr;
    }
    jclass appClass = env->GetObjectClass(app);
    jmethodID getClassLoader = appClass == nullptr ? nullptr
            : env->GetMethodID(appClass, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject loader = getClassLoader == nullptr ? nullptr : env->CallObjectMethod(app, getClassLoader);
    jclass loaderClass = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClass = loaderClass == nullptr ? nullptr
            : env->GetMethodID(loaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    jclass result = nullptr;
    if (loader != nullptr && loadClass != nullptr && !env->ExceptionCheck()) {
        jstring name = env->NewStringUTF("com.unity3d.player.FileSelector");
        jobject clazz = name == nullptr ? nullptr : env->CallObjectMethod(loader, loadClass, name);
        if (name) env->DeleteLocalRef(name);
        if (clazz != nullptr && !env->ExceptionCheck()) result = reinterpret_cast<jclass>(clazz);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (loaderClass) env->DeleteLocalRef(loaderClass);
    if (loader) env->DeleteLocalRef(loader);
    if (appClass) env->DeleteLocalRef(appClass);
    env->DeleteLocalRef(app);
    env->DeleteLocalRef(threadClass);
    return result;
}

bool InitSelector(JNIEnv* env) {
    std::lock_guard<std::mutex> lock(g_selectorMutex);
    if (g_selectorClass && g_selectFile && g_getFilePath && g_isDone) return true;
    jclass local = LoadAppClass(env);
    if (local == nullptr) return false;
    jmethodID selectFile = env->GetStaticMethodID(local, "selectFile", "(Ljava/lang/String;Z)V");
    jmethodID getFilePath = env->GetStaticMethodID(local, "getFilePath", "()Ljava/lang/String;");
    jfieldID isDone = env->GetStaticFieldID(local, "isDone", "Z");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (selectFile == nullptr || getFilePath == nullptr || isDone == nullptr) {
        env->DeleteLocalRef(local);
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

bool ProbeSafBridge() {
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!GetEnv(&env, &attached)) return false;
    const bool ready = InitSelector(env);
    if (attached) g_vm->DetachCurrentThread();
    g_safBridgeReady.store(ready);
    return ready;
}

std::string RunSafPicker(bool multiselect) {
    std::lock_guard<std::mutex> lock(g_pickerMutex);
    g_safPickerCalls.fetch_add(1);
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!GetEnv(&env, &attached) || !InitSelector(env)) {
        g_safLastState.store(1);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    jstring filter = env->NewStringUTF("adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg");
    if (filter == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        g_safLastState.store(2);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    env->CallStaticVoidMethod(g_selectorClass, g_selectFile, filter,
                              multiselect ? JNI_TRUE : JNI_FALSE);
    env->DeleteLocalRef(filter);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        g_safLastState.store(2);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    bool done = false;
    for (int i = 0; i < 18000; ++i) {
        const jboolean value = env->GetStaticBooleanField(g_selectorClass, g_isDone);
        if (env->ExceptionCheck()) { env->ExceptionClear(); g_safLastState.store(2); break; }
        if (value == JNI_TRUE) { done = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::string result;
    if (done) {
        jstring path = reinterpret_cast<jstring>(env->CallStaticObjectMethod(g_selectorClass, g_getFilePath));
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_safLastState.store(2);
        } else if (path != nullptr) {
            const char* chars = env->GetStringUTFChars(path, nullptr);
            if (chars != nullptr) {
                result.assign(chars);
                env->ReleaseStringUTFChars(path, chars);
            }
            env->DeleteLocalRef(path);
        }
    }
    if (attached) g_vm->DetachCurrentThread();
    g_safPickerReturns.fetch_add(1);
    if (!result.empty()) g_safLastState.store(4);
    else if (g_safLastState.load() != 2) g_safLastState.store(3);
    return result;
}

Array<String*>* ToManagedStringArray(const std::string& encoded) {
    std::vector<std::string> paths;
    std::size_t start = 0;
    while (start < encoded.size()) {
        const std::size_t end = encoded.find('\x1f', start);
        const std::string value = encoded.substr(start,
                end == std::string::npos ? std::string::npos : end - start);
        if (!value.empty()) paths.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    auto array = g_stringClass.NewArray<String*>(paths.size());
    if (array == nullptr) return nullptr;
    for (std::size_t i = 0; i < paths.size(); ++i) array->m_Items[i] = CreateMonoString(paths[i]);
    return array;
}

Array<String*>* HookOpenFilePanelFilters(
        String* title, String* directory, Array<ExtensionFilterValue>* filters,
        bool multiselect, IL2CPP::MethodInfo* methodInfo) {
    (void)title;
    (void)directory;
    (void)filters;
    (void)methodInfo;
    if (g_oldOpenFilters == nullptr) return nullptr;
    g_calls.fetch_add(1);
    const int previous = g_callsInFlight.fetch_add(1);
    if (previous == 0 && !WriteMarker(g_callMarker)) {
        g_markerWriteFailures.fetch_add(1);
        g_callsInFlight.fetch_sub(1);
        return nullptr;
    }
    // r6: no original SFB call and no ExtensionFilter[] read; only the embedded Java SAF bridge.
    Array<String*>* result = ToManagedStringArray(RunSafPicker(multiselect));
    g_returns.fetch_add(1);
    if (g_callsInFlight.fetch_sub(1) == 1) ClearMarker(g_callMarker);
    return result;
}

void RunAbiProbeAndMaybeInstallCanary() {
    Class browser("SFB", "StandaloneFileBrowser");
    Class extensionFilter("SFB", "ExtensionFilter");
    MethodBase openFilters = browser ? browser.GetMethod(
            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"}) : MethodBase{};
    IL2CPP::MethodInfo* info = openFilters.IsValid() ? openFilters.GetInfo() : nullptr;
    Class stringClass = Defaults::Get<String*>();
    Class boolClass = Defaults::Get<bool>();
    Class stringArrayClass = stringClass ? stringClass.GetArray() : Class{};
    Class filterArrayClass = extensionFilter ? extensionFilter.GetArray() : Class{};
    const bool params4 = info != nullptr && info->parameters_count == 4;
    const IL2CPP::Il2CppType* p0 = params4 ? info->parameters[0] : nullptr;
    const IL2CPP::Il2CppType* p1 = params4 ? info->parameters[1] : nullptr;
    const IL2CPP::Il2CppType* p2 = params4 ? info->parameters[2] : nullptr;
    const IL2CPP::Il2CppType* p3 = params4 ? info->parameters[3] : nullptr;
    Class returnClass = info && info->return_type ? Class(info->return_type) : Class{};
    FieldBase nameField = extensionFilter ? extensionFilter.GetField("Name") : FieldBase{};
    FieldBase extensionsField = extensionFilter ? extensionFilter.GetField("Extensions") : FieldBase{};
    IL2CPP::Il2CppType* filterType = extensionFilter ? extensionFilter.GetIl2CppType() : nullptr;
    IL2CPP::Il2CppClass* filterClass = extensionFilter ? extensionFilter.GetClass() : nullptr;
    const uint32_t expectedBoxedSize =
            static_cast<uint32_t>(sizeof(IL2CPP::Il2CppObject) + sizeof(ExtensionFilterValue));

    const bool abiGuard =
            openFilters.IsValid() && info && info->methodPointer && openFilters._isStatic && params4 &&
            SameClass(returnClass, stringArrayClass) && SameClass(Class(p0), stringClass) &&
            SameClass(Class(p1), stringClass) && SameClass(Class(p2), filterArrayClass) &&
            SameClass(Class(p3), boolClass) && TypeByRef(p2) == 0 && TypeValueType(p2) == 0 &&
            filterType && filterType->valuetype && filterClass &&
            filterClass->instance_size == expectedBoxedSize && filterClass->actualSize == expectedBoxedSize &&
            nameField.IsValid() && extensionsField.IsValid() && nameField.GetOffset() == 0 &&
            extensionsField.GetOffset() == static_cast<int32_t>(sizeof(void*)) &&
            SameClass(nameField.GetType(), stringClass) &&
            SameClass(extensionsField.GetType(), stringArrayClass);

    g_stringClass = stringClass;
    const bool safReady = ProbeSafBridge();
    const bool fuseReady = PrepareSelfFuse();
    if (abiGuard && safReady && fuseReady && WriteMarker(g_installMarker)) {
        g_installAttempted.store(true);
        BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters);
        const bool installed = g_oldOpenFilters != nullptr;
        g_hookInstalled.store(installed);
        if (installed) ClearMarker(g_installMarker);
        else g_recoveryState.store(4);
    } else if (abiGuard && safReady && fuseReady) {
        g_markerReady.store(false);
        g_recoveryState.store(3);
    }

    std::ostringstream out;
    out << "nativeProbe=cache-post-bnm-sfb-saf-v1\n"
        << "nativeStage=post-bnm-sfb-saf-open\n"
        << "abiProbeRevision=6\n"
        << "bnmLoadRequested=1\n"
        << "bnmLoadedCallback=1\n"
        << "probeComplete=1\n"
        << "gameHooksInstalled=" << (g_hookInstalled.load() ? 1 : 0) << '\n'
        << "sfbOpenFiltersHookInstalled=" << (g_hookInstalled.load() ? 1 : 0) << '\n'
        << "sfbFilterMemoryRead=0\n"
        << "sfbHookPolicy=bootstrap1-self-fused-saf-broad-open\n"
        << "sfbCanarySelfFuse=1\n"
        << "sfbCanaryMarkerReady=" << (g_markerReady.load() ? 1 : 0) << '\n'
        << "sfbCanaryRecoveryState=" << g_recoveryState.load() << '\n'
        << "sfbCanaryInstallAttempted=" << (g_installAttempted.load() ? 1 : 0) << '\n'
        << "sfbCanaryAbiGuard=" << (abiGuard ? 1 : 0) << '\n'
        << "sfbCanaryOriginalCaptured=" << (g_oldOpenFilters ? 1 : 0) << '\n'
        << "sfbCanaryHiddenMethodInfo=1\n"
        << "sfbSafBridgeReady=" << (safReady ? 1 : 0) << '\n'
        << "sfbOriginalCallUsed=0\n"
        << "abi.SFB.class=" << (browser ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.filtersExact=" << (openFilters.IsValid() ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.static=" << ((info && openFilters._isStatic) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.methodPointer=" << ((info && info->methodPointer) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.parameterCount4=" << (params4 ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.return.StringArray=" << (SameClass(returnClass, stringArrayClass) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.param0.String=" << ((p0 && SameClass(Class(p0), stringClass)) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.param1.String=" << ((p1 && SameClass(Class(p1), stringClass)) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.param2.ExtensionFilterArray=" << ((p2 && SameClass(Class(p2), filterArrayClass)) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.param3.Boolean=" << ((p3 && SameClass(Class(p3), boolClass)) ? 1 : 0) << '\n'
        << "abi.SFB.OpenFilePanel.param0.typeCode=" << TypeCode(p0) << '\n'
        << "abi.SFB.OpenFilePanel.param1.typeCode=" << TypeCode(p1) << '\n'
        << "abi.SFB.OpenFilePanel.param2.typeCode=" << TypeCode(p2) << '\n'
        << "abi.SFB.OpenFilePanel.param3.typeCode=" << TypeCode(p3) << '\n'
        << "abi.SFB.OpenFilePanel.param2.byref=" << TypeByRef(p2) << '\n'
        << "abi.SFB.OpenFilePanel.param2.valuetype=" << TypeValueType(p2) << '\n'
        << "abi.SFB.ExtensionFilter.valueType=" << ((filterType && filterType->valuetype) ? 1 : 0) << '\n'
        << "abi.SFB.ExtensionFilter.instanceSize=" << (filterClass ? filterClass->instance_size : 0) << '\n'
        << "abi.SFB.ExtensionFilter.actualSize=" << (filterClass ? filterClass->actualSize : 0) << '\n'
        << "abi.SFB.ExtensionFilter.expectedBoxedSize=" << expectedBoxedSize << '\n'
        << "abi.SFB.ExtensionFilter.payloadSize=" << sizeof(ExtensionFilterValue) << '\n'
        << "abi.SFB.ExtensionFilter.Name.offset=" << (nameField.IsValid() ? nameField.GetOffset() : -1) << '\n'
        << "abi.SFB.ExtensionFilter.Extensions.offset=" << (extensionsField.IsValid() ? extensionsField.GetOffset() : -1) << '\n';
    {
        std::lock_guard<std::mutex> lock(g_reportMutex);
        g_report = out.str();
    }
    g_probeComplete.store(true);
}

std::string CurrentReport() {
    std::ostringstream out;
    if (!g_bnmLoadRequested.load()) out << g_report;
    else if (!g_bnmLoadedCallback.load()) {
        out << "nativeProbe=cache-post-bnm-sfb-saf-v1\n"
            << "nativeStage=post-bnm-sfb-saf-open\n"
            << "abiProbeRevision=6\n"
            << "bnmLoadRequested=1\n"
            << "bnmLoadedCallback=0\n"
            << "probeComplete=0\n"
            << "gameHooksInstalled=0\n"
            << "sfbOpenFiltersHookInstalled=0\n"
            << "sfbFilterMemoryRead=0\n"
            << "sfbHookPolicy=bootstrap1-self-fused-saf-broad-open\n"
            << "sfbSafBridgeReady=0\n"
            << "sfbOriginalCallUsed=0\n";
    } else {
        std::lock_guard<std::mutex> lock(g_reportMutex);
        out << g_report;
    }
    out << "sfbOpenFiltersCanaryCalls=" << g_calls.load() << '\n'
        << "sfbOpenFiltersCanaryReturns=" << g_returns.load() << '\n'
        << "sfbCanaryMarkerWriteFailures=" << g_markerWriteFailures.load() << '\n'
        << "sfbSafPickerCalls=" << g_safPickerCalls.load() << '\n'
        << "sfbSafPickerReturns=" << g_safPickerReturns.load() << '\n'
        << "sfbSafLastState=" << g_safLastState.load() << '\n';
    return out.str();
}
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm == nullptr || vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || env == nullptr)
        return JNI_ERR;
    g_vm = vm;
    g_bnmLoadRequested.store(true);
    Loading::TryLoadByJNI(env);
    Loading::AddOnLoadedEvent([]() {
        g_bnmLoadedCallback.store(true);
        RunAbiProbeAndMaybeInstallCanary();
    });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(JNIEnv* env, jclass) {
    if (env == nullptr) return nullptr;
    const std::string report = CurrentReport();
    return env->NewStringUTF(report.c_str());
}
