#include <jni.h>
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "universe.h"

using namespace BNM;
using namespace BNM::Structures::Mono;
using namespace BNM::Structures::Unity;

namespace {
JavaVM* g_vm = nullptr;
Class g_stringClass;

// Dynamic DEX picker bridge. Registered by RuntimeEntry after DexClassLoader activation.
jclass g_dynamicBridgeClass = nullptr;
jmethodID g_dynamicBegin = nullptr;
jmethodID g_dynamicAwait = nullptr;
jmethodID g_dynamicDiagnostics = nullptr;
std::mutex g_dynamicBridgeMutex;

std::mutex g_installMutex;
std::mutex g_pickerMutex;
std::mutex g_reportMutex;
std::mutex g_textMutex;
std::mutex g_uiFirstCallMutex;

std::atomic<bool> g_bnmLoadRequested{false};
std::atomic<bool> g_bnmLoadedCallback{false};
std::atomic<bool> g_probeComplete{false};
std::atomic<bool> g_dynamicBridgeReady{false};
std::atomic<bool> g_sfbHookInstalled{false};
std::atomic<bool> g_uiHookInstalled{false};
std::atomic<bool> g_sfbInstallAttempted{false};
std::atomic<bool> g_uiInstallAttempted{false};
std::atomic<bool> g_sfbMarkerReady{false};
std::atomic<bool> g_uiMarkerReady{false};
std::atomic<bool> g_uiFirstCallProven{false};
std::atomic<int> g_sfbRecoveryState{0};
std::atomic<int> g_uiRecoveryState{0};
std::atomic<int> g_markerWriteFailures{0};
std::atomic<int> g_sfbCalls{0};
std::atomic<int> g_sfbReturns{0};
std::atomic<int> g_sfbCallsInFlight{0};
std::atomic<int> g_safPickerCalls{0};
std::atomic<int> g_safPickerReturns{0};
std::atomic<int> g_safLastState{0};
std::atomic<int> g_filterReadAttempts{0};
std::atomic<int> g_filterReadSuccess{0};
std::atomic<int> g_filterFallbacks{0};
std::atomic<int> g_filterCount{0};
std::atomic<int> g_extensionCount{0};
std::atomic<int> g_uiHitCalls{0};
std::atomic<int> g_uiHitTrue{0};
std::atomic<int> g_uiHitFalse{0};
std::atomic<int> g_uiLastX100{0};
std::atomic<int> g_uiLastY100{0};
std::atomic<int> g_uiGuardValue{0};
std::atomic<int> g_uiOriginalCalls{0};

std::string g_lastExtensions = "<none>";
std::string g_lastMime = "<none>";
std::string g_dynamicDiagnostics = "directBridge=not-run";
std::string g_report;
std::string g_sfbInstallMarker;
std::string g_sfbCallMarker;
std::string g_uiInstallMarker;
std::string g_uiCallMarker;

constexpr std::size_t kMaxExtensionFilters = 32;
constexpr std::size_t kMaxExtensionsPerFilter = 64;
constexpr std::size_t kMaxUniqueExtensions = 128;
constexpr int kMaxExtensionChars = 32;
constexpr std::size_t kMaxJoinedExtensions = 512;
constexpr const char* kBroadExtensions = "adofai,zip,json,ogg,mp3,wav,png,jpg,jpeg";

struct ExtensionFilterValue {
    String* Name;
    Array<String*>* Extensions;
};
static_assert(sizeof(ExtensionFilterValue) == sizeof(void*) * 2,
              "ExtensionFilter payload must be two managed references");

using OpenFiltersFn = Array<String*>* (*)(
        String*, String*, Array<ExtensionFilterValue>*, bool, IL2CPP::MethodInfo*);
OpenFiltersFn g_oldOpenFilters = nullptr;

using UiHitFn = bool (*)(IL2CPP::Il2CppObject*, Vector2, IL2CPP::MethodInfo*);
UiHitFn g_oldUiHit = nullptr;
Property<IL2CPP::Il2CppObject*> g_eventSystemCurrent;
Class g_pointerEventDataClass;
Class g_listRaycastResultClass;
Property<Vector2> g_pointerPosition;
Property<int> g_listCount;
Method<void> g_raycastAll;

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

bool PrepareSfbFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_sfbRecoveryState.store(3); return false; }
    g_sfbInstallMarker = dir + "/sfb-r9-install.pending";
    g_sfbCallMarker = dir + "/sfb-r9-call.pending";
    g_sfbMarkerReady.store(true);
    if (MarkerExists(g_sfbInstallMarker)) { g_sfbRecoveryState.store(1); return false; }
    if (MarkerExists(g_sfbCallMarker)) { g_sfbRecoveryState.store(2); return false; }
    const std::string probe = dir + "/sfb-r9-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) { g_sfbMarkerReady.store(false); g_sfbRecoveryState.store(3); return false; }
    ClearMarker(probe);
    return true;
}

bool PrepareUiFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_uiRecoveryState.store(3); return false; }
    g_uiInstallMarker = dir + "/uihit-r9-install.pending";
    g_uiCallMarker = dir + "/uihit-r9-call.pending";
    g_uiMarkerReady.store(true);
    if (MarkerExists(g_uiInstallMarker)) { g_uiRecoveryState.store(1); return false; }
    if (MarkerExists(g_uiCallMarker)) { g_uiRecoveryState.store(2); return false; }
    const std::string probe = dir + "/uihit-r9-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) { g_uiMarkerReady.store(false); g_uiRecoveryState.store(3); return false; }
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

bool DynamicBridgeReady() {
    std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
    return g_dynamicBridgeClass != nullptr && g_dynamicBegin != nullptr &&
            g_dynamicAwait != nullptr && g_dynamicDiagnostics != nullptr;
}

void RefreshDynamicDiagnostics(JNIEnv* env) {
    if (env == nullptr) return;
    jclass bridgeClass = nullptr;
    jmethodID diagnostics = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
        bridgeClass = g_dynamicBridgeClass;
        diagnostics = g_dynamicDiagnostics;
    }
    if (!bridgeClass || !diagnostics) return;
    jstring value = reinterpret_cast<jstring>(env->CallStaticObjectMethod(
            bridgeClass, diagnostics));
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return;
    }
    if (value) {
        const char* chars = env->GetStringUTFChars(value, nullptr);
        if (chars) {
            std::lock_guard<std::mutex> textLock(g_textMutex);
            g_dynamicDiagnostics.assign(chars);
            env->ReleaseStringUTFChars(value, chars);
        }
        env->DeleteLocalRef(value);
    }
}

bool ContainsExtension(const std::vector<std::string>& values, const std::string& candidate) {
    for (const std::string& value : values) if (value == candidate) return true;
    return false;
}

bool NormalizeExtension(String* value, std::string* out) {
    if (value == nullptr || out == nullptr) return false;
    const int length = value->length;
    if (length <= 0 || length > kMaxExtensionChars + 2) return false;
    std::string ascii;
    ascii.reserve(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) {
        const auto code = value->chars[i];
        if (code > 0x7f) return false;
        char c = static_cast<char>(code);
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        ascii.push_back(c);
    }
    std::size_t start = 0;
    while (start < ascii.size() && (ascii[start] == '.' || ascii[start] == '*')) ++start;
    if (start >= ascii.size()) return false;
    std::string normalized = ascii.substr(start);
    if (normalized.empty() || normalized.size() > static_cast<std::size_t>(kMaxExtensionChars)) return false;
    for (char c : normalized) {
        const bool valid = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                c == '_' || c == '-' || c == '+';
        if (!valid) return false;
    }
    *out = normalized;
    return true;
}

bool ReadFilterExtensions(Array<ExtensionFilterValue>* filters,
                          std::vector<std::string>* values) {
    g_filterReadAttempts.fetch_add(1);
    if (values == nullptr) return false;
    values->clear();
    g_filterCount.store(0);
    g_extensionCount.store(0);
    if (filters == nullptr) return false;
    const std::size_t filterCount = static_cast<std::size_t>(filters->capacity);
    if (filterCount > kMaxExtensionFilters) return false;
    g_filterCount.store(static_cast<int>(filterCount));
    for (std::size_t i = 0; i < filterCount; ++i) {
        Array<String*>* extensions = filters->m_Items[i].Extensions;
        if (extensions == nullptr) continue;
        const std::size_t extensionCount = static_cast<std::size_t>(extensions->capacity);
        if (extensionCount > kMaxExtensionsPerFilter) return false;
        for (std::size_t j = 0; j < extensionCount; ++j) {
            String* extension = extensions->m_Items[j];
            if (extension == nullptr) continue;
            std::string normalized;
            if (!NormalizeExtension(extension, &normalized)) return false;
            if (!ContainsExtension(*values, normalized)) {
                if (values->size() >= kMaxUniqueExtensions) return false;
                values->push_back(normalized);
            }
        }
    }
    g_extensionCount.store(static_cast<int>(values->size()));
    g_filterReadSuccess.fetch_add(1);
    return true;
}

bool JoinExtensions(const std::vector<std::string>& values, std::string* joined) {
    if (joined == nullptr) return false;
    joined->clear();
    for (const std::string& value : values) {
        if (value.empty()) continue;
        const std::size_t extra = value.size() + (joined->empty() ? 0U : 1U);
        if (joined->size() + extra > kMaxJoinedExtensions) return false;
        if (!joined->empty()) joined->push_back(',');
        joined->append(value);
    }
    return true;
}

void SetLastExtensions(const std::string& value) {
    std::lock_guard<std::mutex> lock(g_textMutex);
    g_lastExtensions = value;
}
void SetLastMime(const std::string& value) {
    std::lock_guard<std::mutex> lock(g_textMutex);
    g_lastMime = value;
}

std::string ResolvePickerExtensions(Array<ExtensionFilterValue>* filters) {
    std::vector<std::string> values;
    std::string joined;
    if (ReadFilterExtensions(filters, &values) && !values.empty() &&
            JoinExtensions(values, &joined) && !joined.empty()) {
        SetLastExtensions(joined);
        return joined;
    }
    g_filterFallbacks.fetch_add(1);
    SetLastExtensions("<broad>");
    return kBroadExtensions;
}

std::string MimeForExtension(const std::string& extension) {
    if (extension == "png") return "image/png";
    if (extension == "jpg" || extension == "jpeg") return "image/jpeg";
    if (extension == "ogg") return "audio/ogg";
    if (extension == "mp3") return "audio/mpeg";
    if (extension == "wav") return "audio/wav";
    if (extension == "zip" || extension == "adozip") return "application/zip";
    if (extension == "json") return "application/json";
    return "";
}

std::string ResolvePickerMime(const std::string& extensions) {
    std::string common;
    std::size_t start = 0;
    while (start <= extensions.size()) {
        const std::size_t end = extensions.find(',', start);
        const std::string extension = extensions.substr(start,
                end == std::string::npos ? std::string::npos : end - start);
        if (!extension.empty()) {
            const std::string mime = MimeForExtension(extension);
            if (mime.empty()) return "*/*";
            if (common.empty()) common = mime;
            else if (common != mime) return "*/*";
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return common.empty() ? "*/*" : common;
}

std::string RunDynamicPicker(bool multiselect, const std::string& extensions) {
    std::lock_guard<std::mutex> serialized(g_pickerMutex);
    g_safPickerCalls.fetch_add(1);
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!GetEnv(&env, &attached) || !DynamicBridgeReady()) {
        g_safLastState.store(1);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    const std::string mime = ResolvePickerMime(extensions);
    SetLastMime(mime);
    jstring jmime = env->NewStringUTF(mime.c_str());
    jstring jextensions = env->NewStringUTF(extensions.c_str());
    if (jmime == nullptr || jextensions == nullptr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (jmime) env->DeleteLocalRef(jmime);
        if (jextensions) env->DeleteLocalRef(jextensions);
        g_safLastState.store(2);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    jclass bridgeClass = nullptr;
    jmethodID beginMethod = nullptr;
    jmethodID awaitMethod = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
        bridgeClass = g_dynamicBridgeClass;
        beginMethod = g_dynamicBegin;
        awaitMethod = g_dynamicAwait;
    }
    if (!bridgeClass || !beginMethod || !awaitMethod) {
        env->DeleteLocalRef(jmime);
        env->DeleteLocalRef(jextensions);
        g_safLastState.store(1);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    const jint requestId = env->CallStaticIntMethod(bridgeClass, beginMethod,
                                                     jmime, jextensions,
                                                     multiselect ? JNI_TRUE : JNI_FALSE);
    env->DeleteLocalRef(jmime);
    env->DeleteLocalRef(jextensions);
    if (env->ExceptionCheck() || requestId <= 0) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        g_safLastState.store(2);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    jstring state = reinterpret_cast<jstring>(env->CallStaticObjectMethod(
            bridgeClass, awaitMethod, requestId, static_cast<jlong>(600000)));
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        g_safLastState.store(2);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    std::string encoded;
    if (state != nullptr) {
        const char* chars = env->GetStringUTFChars(state, nullptr);
        if (chars != nullptr) {
            const std::string value(chars);
            env->ReleaseStringUTFChars(state, chars);
            if (value.rfind("O:", 0) == 0) {
                encoded = value.substr(2);
                g_safLastState.store(encoded.empty() ? 3 : 4);
            } else if (value.rfind("C:", 0) == 0) {
                g_safLastState.store(3);
            } else {
                g_safLastState.store(2);
            }
        }
        env->DeleteLocalRef(state);
    } else {
        g_safLastState.store(2);
    }
    RefreshDynamicDiagnostics(env);
    if (attached) g_vm->DetachCurrentThread();
    g_safPickerReturns.fetch_add(1);
    return encoded;
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
    (void)methodInfo;
    if (g_oldOpenFilters == nullptr) return nullptr;
    g_sfbCalls.fetch_add(1);
    const int previous = g_sfbCallsInFlight.fetch_add(1);
    if (previous == 0 && !WriteMarker(g_sfbCallMarker)) {
        g_markerWriteFailures.fetch_add(1);
        g_sfbCallsInFlight.fetch_sub(1);
        return nullptr;
    }
    const std::string extensions = ResolvePickerExtensions(filters);
    Array<String*>* result = ToManagedStringArray(RunDynamicPicker(multiselect, extensions));
    g_sfbReturns.fetch_add(1);
    if (g_sfbCallsInFlight.fetch_sub(1) == 1) ClearMarker(g_sfbCallMarker);
    return result;
}

bool CustomUiHit(Vector2 position) {
    IL2CPP::Il2CppObject* eventSystem = g_eventSystemCurrent.IsValid()
            ? g_eventSystemCurrent.Get() : nullptr;
    if (!eventSystem || !g_pointerEventDataClass || !g_listRaycastResultClass ||
        !g_pointerPosition.IsValid() || !g_listCount.IsValid() || !g_raycastAll.IsValid()) {
        return false;
    }
    IL2CPP::Il2CppObject* eventData = g_pointerEventDataClass.CreateNewObjectParameters(eventSystem);
    if (!eventData) return false;
    g_pointerPosition[eventData].Set(position);
    IL2CPP::Il2CppObject* results = g_listRaycastResultClass.CreateNewObjectParameters();
    if (!results) return false;
    g_raycastAll[eventSystem].Call(eventData, results);
    return g_listCount[results].Get() > 0;
}

bool HookUiHit(IL2CPP::Il2CppObject* self, Vector2 position, IL2CPP::MethodInfo* methodInfo) {
    (void)self;
    (void)methodInfo;
    g_uiHitCalls.fetch_add(1);
    g_uiLastX100.store(static_cast<int>(position.x * 100.0f));
    g_uiLastY100.store(static_cast<int>(position.y * 100.0f));

    bool firstCanary = false;
    if (!g_uiFirstCallProven.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_uiFirstCallMutex);
        if (!g_uiFirstCallProven.load(std::memory_order_relaxed)) {
            if (!WriteMarker(g_uiCallMarker)) {
                g_markerWriteFailures.fetch_add(1);
                if (g_oldUiHit) {
                    g_uiOriginalCalls.fetch_add(1);
                    return g_oldUiHit(self, position, methodInfo);
                }
                return false;
            }
            firstCanary = true;
        }
    }

    const bool hit = CustomUiHit(position);
    if (hit) g_uiHitTrue.fetch_add(1);
    else g_uiHitFalse.fetch_add(1);

    if (firstCanary) {
        ClearMarker(g_uiCallMarker);
        g_uiFirstCallProven.store(true, std::memory_order_release);
    }
    return hit;
}

bool ResolveUiSurface(MethodBase* uiMethod) {
    Class controller("", "scrController");
    Class vector2("UnityEngine", "Vector2");
    Class boolClass = Defaults::Get<bool>();
    if (!controller || !vector2 || !boolClass) return false;

    MethodBase method = controller.GetMethod("IsScreenPointInsideUIElements", 1);
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    const bool abi = method.IsValid() && info && info->methodPointer && !method._isStatic &&
            info->parameters_count == 1 && info->parameters &&
            SameClass(Class(info->parameters[0]), vector2) && TypeByRef(info->parameters[0]) == 0 &&
            info->return_type && SameClass(Class(info->return_type), boolClass);
    g_uiGuardValue.store(abi ? 1 : 0);
    if (!abi) return false;

    Class eventSystem("UnityEngine.EventSystems", "EventSystem");
    Class pointerEventData("UnityEngine.EventSystems", "PointerEventData");
    Class raycastResult("UnityEngine.EventSystems", "RaycastResult");
    Class list("System.Collections.Generic", "List`1");
    if (!eventSystem || !pointerEventData || !raycastResult || !list) return false;
    Class listRaycast = list.GetGeneric({raycastResult.GetCompileTimeClass()});
    if (!listRaycast) return false;

    g_eventSystemCurrent = eventSystem.GetProperty("current");
    g_pointerEventDataClass = pointerEventData;
    g_listRaycastResultClass = listRaycast;
    g_pointerPosition = pointerEventData.GetProperty("position");
    g_listCount = listRaycast.GetProperty("Count");
    g_raycastAll = eventSystem.GetMethod("RaycastAll");
    const bool surface = g_eventSystemCurrent.IsValid() && g_pointerPosition.IsValid() &&
            g_listCount.IsValid() && g_raycastAll.IsValid();
    if (surface && uiMethod) *uiMethod = method;
    return surface;
}

void MaybeInstallUiHook() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_uiHookInstalled.load(std::memory_order_acquire) ||
        g_uiRecoveryState.load() != 0) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_uiHookInstalled.load() || g_uiInstallAttempted.load()) return;

    MethodBase uiMethod;
    if (!ResolveUiSurface(&uiMethod) || !PrepareUiFuse()) return;
    if (!WriteMarker(g_uiInstallMarker)) {
        g_uiMarkerReady.store(false);
        g_uiRecoveryState.store(3);
        return;
    }
    g_uiInstallAttempted.store(true);
    BasicHook(uiMethod, HookUiHit, g_oldUiHit);
    const bool installed = g_oldUiHit != nullptr;
    g_uiHookInstalled.store(installed);
    if (installed) ClearMarker(g_uiInstallMarker);
    else g_uiRecoveryState.store(4);
}

void MaybeInstallSfbHook() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        !g_dynamicBridgeReady.load(std::memory_order_acquire) ||
        g_sfbHookInstalled.load(std::memory_order_acquire) ||
        g_sfbRecoveryState.load() != 0) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_sfbHookInstalled.load() || g_sfbInstallAttempted.load()) return;

    Class browser("SFB", "StandaloneFileBrowser");
    Class extensionFilter("SFB", "ExtensionFilter");
    MethodBase openFilters = browser ? browser.GetMethod(
            "OpenFilePanel", {"title", "directory", "extensions", "multiselect"}) : MethodBase{};
    IL2CPP::MethodInfo* info = openFilters.IsValid() ? openFilters.GetInfo() : nullptr;
    Class stringClass = Defaults::Get<String*>();
    Class boolClass = Defaults::Get<bool>();
    Class stringArrayClass = stringClass ? stringClass.GetArray() : Class{};
    Class filterArrayClass = extensionFilter ? extensionFilter.GetArray() : Class{};
    const bool params4 = info != nullptr && info->parameters_count == 4 && info->parameters;
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

    const bool abiGuard = openFilters.IsValid() && info && info->methodPointer &&
            openFilters._isStatic && params4 && SameClass(returnClass, stringArrayClass) &&
            SameClass(Class(p0), stringClass) && SameClass(Class(p1), stringClass) &&
            SameClass(Class(p2), filterArrayClass) && SameClass(Class(p3), boolClass) &&
            TypeByRef(p2) == 0 && TypeValueType(p2) == 0 && filterType && filterType->valuetype &&
            filterClass && filterClass->instance_size == expectedBoxedSize &&
            filterClass->actualSize == expectedBoxedSize && nameField.IsValid() &&
            extensionsField.IsValid() && nameField.GetOffset() == 0 &&
            extensionsField.GetOffset() == static_cast<int32_t>(sizeof(void*)) &&
            SameClass(nameField.GetType(), stringClass) &&
            SameClass(extensionsField.GetType(), stringArrayClass);

    g_stringClass = stringClass;
    if (!abiGuard || !PrepareSfbFuse()) return;
    if (!WriteMarker(g_sfbInstallMarker)) {
        g_sfbMarkerReady.store(false);
        g_sfbRecoveryState.store(3);
        return;
    }
    g_sfbInstallAttempted.store(true);
    BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters);
    const bool installed = g_oldOpenFilters != nullptr;
    g_sfbHookInstalled.store(installed);
    if (installed) ClearMarker(g_sfbInstallMarker);
    else g_sfbRecoveryState.store(4);
}

void BuildStaticReport() {
    std::ostringstream out;
    out << "nativeProbe=cache-post-bnm-sfb-dynamic-import-uihit-v1\n"
        << "nativeStage=post-bnm-dynamic-document-and-uihit\n"
        << "abiProbeRevision=9\n"
        << "bnmLoadRequested=" << (g_bnmLoadRequested.load() ? 1 : 0) << '\n'
        << "bnmLoadedCallback=" << (g_bnmLoadedCallback.load() ? 1 : 0) << '\n'
        << "probeComplete=1\n"
        << "gameHooksInstalled=" << ((g_sfbHookInstalled.load() ? 1 : 0) +
                                        (g_uiHookInstalled.load() ? 1 : 0)) << '\n'
        << "sfbOpenFiltersHookInstalled=" << (g_sfbHookInstalled.load() ? 1 : 0) << '\n'
        << "sfbFilterMemoryRead=1\n"
        << "sfbFilterReadBounded=1\n"
        << "sfbFilterReadMaxFilters=" << kMaxExtensionFilters << '\n'
        << "sfbFilterReadMaxExtensionsPerFilter=" << kMaxExtensionsPerFilter << '\n'
        << "sfbHookPolicy=dynamic-document-preprocess-before-bind\n"
        << "sfbPickerBackend=dynamic-document\n"
        << "sfbFileSelectorBypassed=1\n"
        << "sfbEmbeddedBridgeBypassed=1\n"
        << "sfbDynamicBridgeReady=" << (g_dynamicBridgeReady.load() ? 1 : 0) << '\n'
        << "sfbOriginalCallUsed=0\n"
        << "sfbCanarySelfFuse=1\n"
        << "sfbCanaryMarkerReady=" << (g_sfbMarkerReady.load() ? 1 : 0) << '\n'
        << "sfbCanaryRecoveryState=" << g_sfbRecoveryState.load() << '\n'
        << "sfbCanaryInstallAttempted=" << (g_sfbInstallAttempted.load() ? 1 : 0) << '\n'
        << "sfbCanaryOriginalCaptured=" << (g_oldOpenFilters ? 1 : 0) << '\n'
        << "sfbCanaryHiddenMethodInfo=1\n"
        << "uiHitHookInstalled=" << (g_uiHookInstalled.load() ? 1 : 0) << '\n'
        << "uiHitAbiGuard=" << g_uiGuardValue.load() << '\n'
        << "uiHitPolicy=pinned-upstream-eventsystem-raycast\n"
        << "uiHitSourceCommit=74bcc7a0d8c8be1267504e21e28a35e199b5d4eb\n"
        << "uiHitOriginalCalled=" << (g_uiOriginalCalls.load() > 0 ? 1 : 0) << '\n'
        << "uiHitOriginalCalls=" << g_uiOriginalCalls.load() << '\n'
        << "uiHitSelfFuse=1\n"
        << "uiHitMarkerReady=" << (g_uiMarkerReady.load() ? 1 : 0) << '\n'
        << "uiHitRecoveryState=" << g_uiRecoveryState.load() << '\n'
        << "uiHitInstallAttempted=" << (g_uiInstallAttempted.load() ? 1 : 0) << '\n'
        << "uiHitOriginalCaptured=" << (g_oldUiHit ? 1 : 0) << '\n';
    std::lock_guard<std::mutex> lock(g_reportMutex);
    g_report = out.str();
    g_probeComplete.store(true);
}

void ReconcileInstallState() {
    MaybeInstallUiHook();
    MaybeInstallSfbHook();
    BuildStaticReport();
}

std::string CurrentReport(JNIEnv* env) {
    if (env) RefreshDynamicDiagnostics(env);
    BuildStaticReport();
    std::string base;
    {
        std::lock_guard<std::mutex> lock(g_reportMutex);
        base = g_report;
    }
    std::string lastExtensions;
    std::string lastMime;
    std::string dynamicDiag;
    {
        std::lock_guard<std::mutex> lock(g_textMutex);
        lastExtensions = g_lastExtensions;
        lastMime = g_lastMime;
        dynamicDiag = g_dynamicDiagnostics;
    }
    std::ostringstream out;
    out << base
        << "sfbOpenFiltersCanaryCalls=" << g_sfbCalls.load() << '\n'
        << "sfbOpenFiltersCanaryReturns=" << g_sfbReturns.load() << '\n'
        << "sfbCanaryMarkerWriteFailures=" << g_markerWriteFailures.load() << '\n'
        << "sfbSafPickerCalls=" << g_safPickerCalls.load() << '\n'
        << "sfbSafPickerReturns=" << g_safPickerReturns.load() << '\n'
        << "sfbSafLastState=" << g_safLastState.load() << '\n'
        << "sfbFilterReadAttempts=" << g_filterReadAttempts.load() << '\n'
        << "sfbFilterReadSuccess=" << g_filterReadSuccess.load() << '\n'
        << "sfbFilterFallbacks=" << g_filterFallbacks.load() << '\n'
        << "sfbFilterCount=" << g_filterCount.load() << '\n'
        << "sfbExtensionCount=" << g_extensionCount.load() << '\n'
        << "sfbLastExtensions=" << lastExtensions << '\n'
        << "sfbLastMime=" << lastMime << '\n'
        << dynamicDiag;
    if (!dynamicDiag.empty() && dynamicDiag.back() != '\n') out << '\n';
    out << "uiHitCalls=" << g_uiHitCalls.load() << '\n'
        << "uiHitTrue=" << g_uiHitTrue.load() << '\n'
        << "uiHitFalse=" << g_uiHitFalse.load() << '\n'
        << "uiHitFirstCallProven=" << (g_uiFirstCallProven.load() ? 1 : 0) << '\n'
        << "uiHitLastX100=" << g_uiLastX100.load() << '\n'
        << "uiHitLastY100=" << g_uiLastY100.load() << '\n';
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
        ReconcileInstallState();
    });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL
Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeRegisterDynamicBridge(
        JNIEnv* env, jclass, jclass bridgeClass) {
    if (env == nullptr || bridgeClass == nullptr) return;
    std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
    if (g_dynamicBridgeClass != nullptr) return;
    jmethodID begin = env->GetStaticMethodID(
            bridgeClass, "begin", "(Ljava/lang/String;Ljava/lang/String;Z)I");
    jmethodID await = env->GetStaticMethodID(
            bridgeClass, "await", "(IJ)Ljava/lang/String;");
    jmethodID diagnostics = env->GetStaticMethodID(
            bridgeClass, "diagnostics", "()Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (begin == nullptr || await == nullptr || diagnostics == nullptr) return;
    jclass global = reinterpret_cast<jclass>(env->NewGlobalRef(bridgeClass));
    if (global == nullptr) return;
    g_dynamicBridgeClass = global;
    g_dynamicBegin = begin;
    g_dynamicAwait = await;
    g_dynamicDiagnostics = diagnostics;
    g_dynamicBridgeReady.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeReconcileDynamicRuntime(
        JNIEnv*, jclass) {
    // RuntimeEntry calls this immediately after bridge registration so SFB installation
    // does not depend on the user opening the diagnostics panel first.
    ReconcileInstallState();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(JNIEnv* env, jclass) {
    if (env == nullptr) return nullptr;
    ReconcileInstallState();
    const std::string report = CurrentReport(env);
    return env->NewStringUTF(report.c_str());
}
