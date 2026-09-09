#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <atomic>
#include <string>

#define LOG_TAG "ADOFAI.V240Fix"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Il2CppDomain;
struct Il2CppAssembly;
struct Il2CppImage;
struct Il2CppClass;
struct Il2CppObject;
struct Il2CppString;
struct Il2CppArray;
struct Il2CppType;
struct MethodInfo;

struct Il2CppArrayLayout {
    void* klass;
    void* monitor;
    void* bounds;
    uintptr_t max_length;
};

struct Api {
    Il2CppDomain* (*domain_get)();
    const Il2CppAssembly** (*domain_get_assemblies)(const Il2CppDomain*, size_t*);
    const Il2CppImage* (*assembly_get_image)(const Il2CppAssembly*);
    Il2CppClass* (*class_from_name)(const Il2CppImage*, const char*, const char*);
    const MethodInfo* (*class_get_method_from_name)(Il2CppClass*, const char*, int);
    Il2CppClass* (*object_get_class)(Il2CppObject*);
    Il2CppObject* (*runtime_invoke)(const MethodInfo*, void*, void**, Il2CppObject**);
    Il2CppString* (*string_new)(const char*);
    int32_t (*string_length)(Il2CppString*);
    const uint16_t* (*string_chars)(Il2CppString*);
    const Il2CppImage* (*get_corlib)();
    Il2CppArray* (*array_new)(Il2CppClass*, size_t);
    void (*gc_wbarrier_set_field)(Il2CppObject*, void**, void*);
    uint32_t (*gchandle_new)(Il2CppObject*, bool);
    Il2CppObject* (*gchandle_get_target)(uint32_t);
    void (*gchandle_free)(uint32_t);
    void* (*thread_attach)(Il2CppDomain*);
    void (*thread_detach)(void*);
};

Api g_api{};
std::atomic<bool> g_api_ready(false);
JavaVM* g_vm = nullptr;
jclass g_bridge_class = nullptr;
jclass g_settings_class = nullptr;
jmethodID g_begin_open = nullptr;
jmethodID g_begin_save = nullptr;
jmethodID g_begin_folder = nullptr;
jmethodID g_poll = nullptr;
jmethodID g_touch_assist = nullptr;

static JNIEnv* env_for_thread(bool* attached) {
    if (attached) *attached = false;
    if (!g_vm) return nullptr;
    JNIEnv* env = nullptr;
    jint state = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (state == JNI_OK) return env;
    if (state != JNI_EDETACHED) return nullptr;
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return nullptr;
    if (attached) *attached = true;
    return env;
}

static void detach_jni(bool attached) {
    if (attached && g_vm) g_vm->DetachCurrentThread();
}

static bool clear_jni_exception(JNIEnv* env, const char* where) {
    if (!env || !env->ExceptionCheck()) return false;
    LOGE("JNI exception at %s", where);
    env->ExceptionClear();
    return true;
}

static std::string utf16_to_utf8(const uint16_t* data, int32_t length) {
    std::string out;
    if (!data || length <= 0) return out;
    out.reserve(static_cast<size_t>(length));
    for (int32_t i = 0; i < length; ++i) {
        uint32_t cp = data[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < length) {
            uint32_t lo = data[i + 1];
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                ++i;
            }
        }
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

static std::string managed_string(Il2CppString* value) {
    if (!value || !g_api.string_length || !g_api.string_chars) return std::string();
    return utf16_to_utf8(g_api.string_chars(value), g_api.string_length(value));
}

static bool resolve_api(void* handle) {
#define RESOLVE(field, symbol) \
    do { \
        g_api.field = reinterpret_cast<decltype(g_api.field)>(dlsym(handle, symbol)); \
        if (!g_api.field) { LOGE("missing %s", symbol); return false; } \
    } while (0)
    RESOLVE(domain_get, "il2cpp_domain_get");
    RESOLVE(domain_get_assemblies, "il2cpp_domain_get_assemblies");
    RESOLVE(assembly_get_image, "il2cpp_assembly_get_image");
    RESOLVE(class_from_name, "il2cpp_class_from_name");
    RESOLVE(class_get_method_from_name, "il2cpp_class_get_method_from_name");
    RESOLVE(object_get_class, "il2cpp_object_get_class");
    RESOLVE(runtime_invoke, "il2cpp_runtime_invoke");
    RESOLVE(string_new, "il2cpp_string_new");
    RESOLVE(string_length, "il2cpp_string_length");
    RESOLVE(string_chars, "il2cpp_string_chars");
    RESOLVE(get_corlib, "il2cpp_get_corlib");
    RESOLVE(array_new, "il2cpp_array_new");
    RESOLVE(gc_wbarrier_set_field, "il2cpp_gc_wbarrier_set_field");
    RESOLVE(gchandle_new, "il2cpp_gchandle_new");
    RESOLVE(gchandle_get_target, "il2cpp_gchandle_get_target");
    RESOLVE(gchandle_free, "il2cpp_gchandle_free");
    RESOLVE(thread_attach, "il2cpp_thread_attach");
    RESOLVE(thread_detach, "il2cpp_thread_detach");
#undef RESOLVE
    return true;
}

static Il2CppClass* find_class(const char* namespaze, const char* name) {
    Il2CppDomain* domain = g_api.domain_get ? g_api.domain_get() : nullptr;
    if (!domain) return nullptr;
    size_t count = 0;
    const Il2CppAssembly** assemblies = g_api.domain_get_assemblies(domain, &count);
    if (!assemblies) return nullptr;
    for (size_t i = 0; i < count; ++i) {
        const Il2CppImage* image = g_api.assembly_get_image(assemblies[i]);
        if (!image) continue;
        Il2CppClass* klass = g_api.class_from_name(image, namespaze, name);
        if (klass) return klass;
    }
    return nullptr;
}

static bool patch_aarch64(void* target, void* replacement) {
#if defined(__aarch64__)
    if (!target || !replacement) return false;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return false;
    uintptr_t start = reinterpret_cast<uintptr_t>(target);
    uintptr_t page = start & ~static_cast<uintptr_t>(page_size - 1);
    uintptr_t end = (start + 16 + page_size - 1) & ~static_cast<uintptr_t>(page_size - 1);
    size_t span = end - page;
    if (mprotect(reinterpret_cast<void*>(page), span, PROT_READ | PROT_WRITE) != 0) {
        LOGE("mprotect RW failed target=%p errno=%d", target, errno);
        return false;
    }
    uint32_t instructions[2];
    instructions[0] = 0x58000050u; // ldr x16, #8
    instructions[1] = 0xD61F0200u; // br x16
    memcpy(target, instructions, sizeof(instructions));
    uintptr_t absolute = reinterpret_cast<uintptr_t>(replacement);
    memcpy(reinterpret_cast<uint8_t*>(target) + 8, &absolute, sizeof(absolute));
    __builtin___clear_cache(reinterpret_cast<char*>(target), reinterpret_cast<char*>(target) + 16);
    if (mprotect(reinterpret_cast<void*>(page), span, PROT_READ | PROT_EXEC) != 0) {
        LOGE("mprotect RX restore failed target=%p errno=%d", target, errno);
        return false;
    }
    return true;
#else
    (void)target;
    (void)replacement;
    LOGE("unsupported ABI for inline hook");
    return false;
#endif
}

static void* method_pointer(const MethodInfo* method) {
    if (!method) return nullptr;
    return *reinterpret_cast<void* const*>(method);
}

static int java_begin_open(const char* mime) {
    bool attached = false;
    JNIEnv* env = env_for_thread(&attached);
    if (!env || !g_bridge_class || !g_begin_open) {
        detach_jni(attached);
        return -1;
    }
    jstring jmime = env->NewStringUTF(mime ? mime : "*/*");
    jint id = env->CallStaticIntMethod(g_bridge_class, g_begin_open, jmime);
    if (jmime) env->DeleteLocalRef(jmime);
    if (clear_jni_exception(env, "beginOpen")) id = -1;
    detach_jni(attached);
    return id;
}

static int java_begin_save(const char* name, const char* mime) {
    bool attached = false;
    JNIEnv* env = env_for_thread(&attached);
    if (!env || !g_bridge_class || !g_begin_save) {
        detach_jni(attached);
        return -1;
    }
    jstring jname = env->NewStringUTF(name && *name ? name : "level.adofai");
    jstring jmime = env->NewStringUTF(mime && *mime ? mime : "application/octet-stream");
    jint id = env->CallStaticIntMethod(g_bridge_class, g_begin_save, jname, jmime);
    if (jname) env->DeleteLocalRef(jname);
    if (jmime) env->DeleteLocalRef(jmime);
    if (clear_jni_exception(env, "beginSave")) id = -1;
    detach_jni(attached);
    return id;
}

static int java_begin_folder() {
    bool attached = false;
    JNIEnv* env = env_for_thread(&attached);
    if (!env || !g_bridge_class || !g_begin_folder) {
        detach_jni(attached);
        return -1;
    }
    jint id = env->CallStaticIntMethod(g_bridge_class, g_begin_folder);
    if (clear_jni_exception(env, "beginFolder")) id = -1;
    detach_jni(attached);
    return id;
}

static std::string java_poll(int id) {
    bool attached = false;
    JNIEnv* env = env_for_thread(&attached);
    if (!env || !g_bridge_class || !g_poll) {
        detach_jni(attached);
        return "E:JNI unavailable";
    }
    jstring result = static_cast<jstring>(env->CallStaticObjectMethod(g_bridge_class, g_poll, static_cast<jint>(id)));
    if (clear_jni_exception(env, "poll")) {
        detach_jni(attached);
        return "E:poll exception";
    }
    std::string value;
    if (result) {
        const char* chars = env->GetStringUTFChars(result, nullptr);
        if (chars) {
            value.assign(chars);
            env->ReleaseStringUTFChars(result, chars);
        }
        env->DeleteLocalRef(result);
    }
    detach_jni(attached);
    return value.empty() ? "E:empty poll result" : value;
}

static bool java_touch_assist() {
    bool attached = false;
    JNIEnv* env = env_for_thread(&attached);
    if (!env || !g_settings_class || !g_touch_assist) {
        detach_jni(attached);
        return true;
    }
    jboolean enabled = env->CallStaticBooleanMethod(g_settings_class, g_touch_assist);
    if (clear_jni_exception(env, "touchAssist")) enabled = JNI_TRUE;
    detach_jni(attached);
    return enabled == JNI_TRUE;
}

enum RequestKind {
    REQUEST_OPEN = 1,
    REQUEST_SAVE = 2,
    REQUEST_FOLDER = 3
};

struct Request {
    RequestKind kind;
    int id;
    uint32_t callback_handle;
};

static Il2CppArray* make_string_array(const std::string* value) {
    const Il2CppImage* corlib = g_api.get_corlib();
    if (!corlib) return nullptr;
    Il2CppClass* string_class = g_api.class_from_name(corlib, "System", "String");
    if (!string_class) return nullptr;
    const size_t count = value ? 1u : 0u;
    Il2CppArray* array = g_api.array_new(string_class, count);
    if (!array || !value) return array;
    Il2CppString* managed = g_api.string_new(value->c_str());
    uint8_t* raw = reinterpret_cast<uint8_t*>(array);
    void** slot = reinterpret_cast<void**>(raw + sizeof(Il2CppArrayLayout));
    g_api.gc_wbarrier_set_field(reinterpret_cast<Il2CppObject*>(array), slot, managed);
    return array;
}

static void invoke_callback(RequestKind kind, uint32_t handle, const std::string* value) {
    Il2CppObject* callback = g_api.gchandle_get_target(handle);
    if (!callback) return;
    Il2CppClass* klass = g_api.object_get_class(callback);
    if (!klass) return;
    const MethodInfo* invoke = g_api.class_get_method_from_name(klass, "Invoke", 1);
    if (!invoke) {
        LOGE("delegate Invoke method not found kind=%d", static_cast<int>(kind));
        return;
    }

    Il2CppObject* exception = nullptr;
    void* args[1] = {nullptr};
    if (kind == REQUEST_SAVE) {
        Il2CppString* text = g_api.string_new(value ? value->c_str() : "");
        args[0] = text;
    } else {
        Il2CppArray* array = make_string_array(value);
        args[0] = array;
    }
    g_api.runtime_invoke(invoke, callback, args, &exception);
    if (exception) LOGE("managed callback threw kind=%d", static_cast<int>(kind));
}

static void* request_worker(void* opaque) {
    Request* request = static_cast<Request*>(opaque);
    if (!request) return nullptr;
    void* il2cpp_thread = nullptr;
    Il2CppDomain* domain = g_api.domain_get();
    if (domain) il2cpp_thread = g_api.thread_attach(domain);

    std::string final_value;
    bool success = false;
    const int max_ticks = 18000; // 15 minutes at 50 ms.
    for (int tick = 0; tick < max_ticks; ++tick) {
        std::string state = java_poll(request->id);
        if (state == "P") {
            usleep(50 * 1000);
            continue;
        }
        if (state.rfind("O:", 0) == 0) {
            final_value = state.substr(2);
            success = true;
        } else if (state.rfind("E:", 0) == 0) {
            LOGE("picker failed kind=%d: %s", static_cast<int>(request->kind), state.c_str() + 2);
        }
        break;
    }

    if (success) invoke_callback(request->kind, request->callback_handle, &final_value);
    else invoke_callback(request->kind, request->callback_handle, nullptr);

    g_api.gchandle_free(request->callback_handle);
    if (il2cpp_thread) g_api.thread_detach(il2cpp_thread);
    delete request;
    return nullptr;
}

static void begin_request(RequestKind kind, int id, Il2CppObject* callback) {
    if (!callback) {
        LOGE("null callback kind=%d", static_cast<int>(kind));
        return;
    }
    uint32_t handle = g_api.gchandle_new(callback, false);
    if (!handle) {
        LOGE("gchandle allocation failed kind=%d", static_cast<int>(kind));
        return;
    }
    if (id <= 0) {
        invoke_callback(kind, handle, nullptr);
        g_api.gchandle_free(handle);
        return;
    }
    Request* request = new Request{kind, id, handle};
    pthread_t thread;
    int rc = pthread_create(&thread, nullptr, request_worker, request);
    if (rc != 0) {
        LOGE("request worker create failed rc=%d", rc);
        invoke_callback(kind, handle, nullptr);
        g_api.gchandle_free(handle);
        delete request;
        return;
    }
    pthread_detach(thread);
}

extern "C" void hook_OpenFilePanelAsync(Il2CppString*, Il2CppString*, Il2CppArray*, bool,
                                         Il2CppObject* callback, const MethodInfo*) {
    int id = java_begin_open("*/*");
    begin_request(REQUEST_OPEN, id, callback);
}

extern "C" void hook_OpenFolderPanelAsync(Il2CppString*, Il2CppString*, bool,
                                           Il2CppObject* callback, const MethodInfo*) {
    int id = java_begin_folder();
    begin_request(REQUEST_FOLDER, id, callback);
}

extern "C" void hook_SaveFilePanelAsync(Il2CppString*, Il2CppString*, Il2CppString* default_name,
                                         Il2CppArray*, Il2CppObject* callback, const MethodInfo*) {
    std::string name = managed_string(default_name);
    if (name.empty()) name = "level.adofai";
    const char* mime = name.size() >= 7 && name.substr(name.size() - 7) == ".adofai"
            ? "application/json" : "application/octet-stream";
    int id = java_begin_save(name.c_str(), mime);
    begin_request(REQUEST_SAVE, id, callback);
}

static bool install_one(Il2CppClass* sfb, const char* name, int argc, void* replacement) {
    const MethodInfo* method = g_api.class_get_method_from_name(sfb, name, argc);
    if (!method) {
        LOGE("SFB method missing %s/%d", name, argc);
        return false;
    }
    void* target = method_pointer(method);
    Dl_info info{};
    if (!target || dladdr(target, &info) == 0) {
        LOGE("SFB method pointer invalid %s", name);
        return false;
    }
    LOGI("hooking %s target=%p image=%s", name, target, info.dli_fname ? info.dli_fname : "?");
    return patch_aarch64(target, replacement);
}

static void apply_touch_mouse(bool enabled) {
    Il2CppClass* input = find_class("UnityEngine", "Input");
    if (!input) return;
    const MethodInfo* setter = g_api.class_get_method_from_name(input, "set_simulateMouseWithTouches", 1);
    if (!setter) return;
    uint8_t value = enabled ? 1 : 0;
    void* args[1] = {&value};
    Il2CppObject* exception = nullptr;
    g_api.runtime_invoke(setter, nullptr, args, &exception);
    if (exception) LOGW("Input.simulateMouseWithTouches setter threw");
}

static void* settings_worker(void*) {
    void* attached_thread = nullptr;
    Il2CppDomain* domain = g_api.domain_get();
    if (domain) attached_thread = g_api.thread_attach(domain);
    int last = -1;
    for (;;) {
        int current = java_touch_assist() ? 1 : 0;
        if (current != last) {
            apply_touch_mouse(current != 0);
            LOGI("touch assist=%d", current);
            last = current;
        }
        usleep(1000 * 1000);
    }
    if (attached_thread) g_api.thread_detach(attached_thread);
    return nullptr;
}

static void* hook_worker(void*) {
    void* il2cpp = nullptr;
    for (int i = 0; i < 300; ++i) {
        il2cpp = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
        if (il2cpp) break;
        usleep(100 * 1000);
    }
    if (!il2cpp) {
        LOGE("libil2cpp.so was not loaded");
        return nullptr;
    }
    if (!resolve_api(il2cpp)) return nullptr;

    Il2CppDomain* domain = nullptr;
    for (int i = 0; i < 300; ++i) {
        domain = g_api.domain_get();
        if (domain) break;
        usleep(100 * 1000);
    }
    if (!domain) {
        LOGE("IL2CPP domain unavailable");
        return nullptr;
    }
    void* attached_thread = g_api.thread_attach(domain);

    Il2CppClass* sfb = nullptr;
    for (int i = 0; i < 300; ++i) {
        sfb = find_class("SFB", "StandaloneFileBrowser");
        if (sfb) break;
        usleep(100 * 1000);
    }
    if (!sfb) {
        LOGE("SFB.StandaloneFileBrowser class unavailable");
        if (attached_thread) g_api.thread_detach(attached_thread);
        return nullptr;
    }

    bool ok = true;
    ok &= install_one(sfb, "OpenFilePanelAsync", 5, reinterpret_cast<void*>(&hook_OpenFilePanelAsync));
    ok &= install_one(sfb, "OpenFolderPanelAsync", 4, reinterpret_cast<void*>(&hook_OpenFolderPanelAsync));
    ok &= install_one(sfb, "SaveFilePanelAsync", 5, reinterpret_cast<void*>(&hook_SaveFilePanelAsync));
    g_api_ready.store(ok, std::memory_order_release);
    LOGI("SFB Android hook install %s", ok ? "complete" : "incomplete");

    apply_touch_mouse(java_touch_assist());
    pthread_t settings;
    if (pthread_create(&settings, nullptr, settings_worker, nullptr) == 0) pthread_detach(settings);

    if (attached_thread) g_api.thread_detach(attached_thread);
    return nullptr;
}

static bool cache_java(JNIEnv* env) {
    jclass bridge = env->FindClass("com/unity3d/player/V240AndroidBridge");
    if (!bridge || clear_jni_exception(env, "FindClass V240AndroidBridge")) return false;
    g_bridge_class = static_cast<jclass>(env->NewGlobalRef(bridge));
    env->DeleteLocalRef(bridge);

    jclass settings = env->FindClass("com/unity3d/player/V240SettingsOverlay");
    if (settings && !clear_jni_exception(env, "FindClass V240SettingsOverlay")) {
        g_settings_class = static_cast<jclass>(env->NewGlobalRef(settings));
        env->DeleteLocalRef(settings);
    } else {
        clear_jni_exception(env, "settings class fallback");
    }

    g_begin_open = env->GetStaticMethodID(g_bridge_class, "beginOpen", "(Ljava/lang/String;)I");
    g_begin_save = env->GetStaticMethodID(g_bridge_class, "beginSave", "(Ljava/lang/String;Ljava/lang/String;)I");
    g_begin_folder = env->GetStaticMethodID(g_bridge_class, "beginFolder", "()I");
    g_poll = env->GetStaticMethodID(g_bridge_class, "poll", "(I)Ljava/lang/String;");
    if (g_settings_class) {
        g_touch_assist = env->GetStaticMethodID(g_settings_class, "touchAssist", "()Z");
    }
    if (clear_jni_exception(env, "GetStaticMethodID")) return false;
    return g_begin_open && g_begin_save && g_begin_folder && g_poll;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    JNIEnv* env = nullptr;
    if (!vm || vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        return JNI_ERR;
    }
    if (!cache_java(env)) {
        LOGE("Java bridge cache failed");
        return JNI_ERR;
    }
    pthread_t thread;
    if (pthread_create(&thread, nullptr, hook_worker, nullptr) != 0) {
        LOGE("hook worker create failed");
        return JNI_ERR;
    }
    pthread_detach(thread);
    LOGI("v240fix JNI loaded");
    return JNI_VERSION_1_6;
}
