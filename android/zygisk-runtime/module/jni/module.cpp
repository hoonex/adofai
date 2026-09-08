#include <jni.h>
#include <android/dlext.h>
#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "zygisk.hpp"

namespace {
constexpr const char *kTag = "ADOFAI.ZygiskEditor";
constexpr const char *kTargetProcess = "com.fizzd.connectedworlds";
constexpr const char *kDexPath = "payload/editor.dex";
constexpr const char *kNativePath = "payload/libOctober.so";
constexpr int kApplicationWaitAttempts = 300;
constexpr useconds_t kApplicationWaitUs = 100000;

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, kTag, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kTag, __VA_ARGS__)

bool clearException(JNIEnv *env, const char *stage) {
    if (!env || !env->ExceptionCheck()) return false;
    env->ExceptionDescribe();
    env->ExceptionClear();
    LOGE("JNI exception during %s", stage);
    return true;
}

jobject waitForApplication(JNIEnv *env) {
    jclass activityThread = env->FindClass("android/app/ActivityThread");
    if (!activityThread || clearException(env, "ActivityThread lookup")) return nullptr;
    jmethodID currentApplication = env->GetStaticMethodID(
            activityThread, "currentApplication", "()Landroid/app/Application;");
    if (!currentApplication || clearException(env, "currentApplication lookup")) return nullptr;

    for (int attempt = 0; attempt < kApplicationWaitAttempts; ++attempt) {
        jobject application = env->CallStaticObjectMethod(activityThread, currentApplication);
        if (clearException(env, "currentApplication call")) return nullptr;
        if (application) return application;
        usleep(kApplicationWaitUs);
    }
    LOGE("Timed out waiting for Android Application");
    return nullptr;
}

jobject createDexLoader(JNIEnv *env, jobject application, int dexFd,
                        void **mappedDexOut, size_t *mappedDexSizeOut) {
    struct stat st{};
    if (fstat(dexFd, &st) != 0 || st.st_size <= 0) {
        LOGE("editor.dex fstat failed: %s", strerror(errno));
        return nullptr;
    }

    void *mapped = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, dexFd, 0);
    if (mapped == MAP_FAILED) {
        LOGE("editor.dex mmap failed: %s", strerror(errno));
        return nullptr;
    }

    jobject buffer = env->NewDirectByteBuffer(mapped, static_cast<jlong>(st.st_size));
    if (!buffer || clearException(env, "NewDirectByteBuffer")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }

    jclass appClass = env->GetObjectClass(application);
    jmethodID getClassLoader = env->GetMethodID(
            appClass, "getClassLoader", "()Ljava/lang/ClassLoader;");
    if (!getClassLoader || clearException(env, "Application.getClassLoader lookup")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }
    jobject parent = env->CallObjectMethod(application, getClassLoader);
    if (!parent || clearException(env, "Application.getClassLoader call")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }

    jclass loaderClass = env->FindClass("dalvik/system/InMemoryDexClassLoader");
    if (!loaderClass || clearException(env, "InMemoryDexClassLoader lookup")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(
            loaderClass, "<init>", "(Ljava/nio/ByteBuffer;Ljava/lang/ClassLoader;)V");
    if (!ctor || clearException(env, "InMemoryDexClassLoader ctor lookup")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }

    jobject loader = env->NewObject(loaderClass, ctor, buffer, parent);
    if (!loader || clearException(env, "InMemoryDexClassLoader construction")) {
        munmap(mapped, static_cast<size_t>(st.st_size));
        return nullptr;
    }

    *mappedDexOut = mapped;
    *mappedDexSizeOut = static_cast<size_t>(st.st_size);
    return loader;
}

jclass loadClass(JNIEnv *env, jobject loader, const char *name) {
    jclass classLoaderClass = env->FindClass("java/lang/ClassLoader");
    if (!classLoaderClass || clearException(env, "ClassLoader lookup")) return nullptr;
    jmethodID loadClassMethod = env->GetMethodID(
            classLoaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    if (!loadClassMethod || clearException(env, "ClassLoader.loadClass lookup")) return nullptr;

    jstring className = env->NewStringUTF(name);
    if (!className) return nullptr;
    jobject result = env->CallObjectMethod(loader, loadClassMethod, className);
    env->DeleteLocalRef(className);
    if (!result || clearException(env, name)) return nullptr;
    return reinterpret_cast<jclass>(result);
}

void *loadOctober(JavaVM *vm, JNIEnv *env, int nativeFd) {
    android_dlextinfo ext{};
    ext.flags = ANDROID_DLEXT_USE_LIBRARY_FD;
    ext.library_fd = nativeFd;
    void *handle = android_dlopen_ext("libOctober.so", RTLD_NOW | RTLD_LOCAL, &ext);
    if (!handle) {
        LOGE("android_dlopen_ext(libOctober.so) failed: %s", dlerror());
        return nullptr;
    }

    using OnLoad = jint (*)(JavaVM *, void *);
    auto onLoad = reinterpret_cast<OnLoad>(dlsym(handle, "JNI_OnLoad"));
    if (!onLoad) {
        LOGE("libOctober.so does not export JNI_OnLoad");
        return nullptr;
    }
    jint version = onLoad(vm, nullptr);
    if (version == JNI_ERR || clearException(env, "libOctober JNI_OnLoad")) {
        LOGE("libOctober JNI_OnLoad failed");
        return nullptr;
    }
    LOGI("libOctober loaded from preserved module file descriptor");
    return handle;
}

bool registerPreviewNative(JNIEnv *env, jclass shellClass, void *octoberHandle) {
    void *preview = dlsym(
            octoberHandle,
            "Java_com_unity3d_player_MobileEditorShell_nativeQueuePreview");
    if (!preview) {
        LOGE("nativeQueuePreview export not found in libOctober.so");
        return false;
    }
    JNINativeMethod method{
            const_cast<char *>("nativeQueuePreview"),
            const_cast<char *>("(Ljava/lang/String;)Z"),
            preview};
    if (env->RegisterNatives(shellClass, &method, 1) != JNI_OK ||
        clearException(env, "RegisterNatives(nativeQueuePreview)")) {
        LOGE("RegisterNatives(nativeQueuePreview) failed");
        return false;
    }
    return true;
}
}  // namespace

class ADOFAIEditorModule final : public zygisk::ModuleBase {
public:
    void onLoad(zygisk::Api *api, JNIEnv *env) override {
        api_ = api;
        env_ = env;
        if (env_) env_->GetJavaVM(&vm_);
    }

    void preAppSpecialize(zygisk::AppSpecializeArgs *args) override {
        target_ = false;
        if (!env_ || !args || !args->nice_name) {
            api_->setOption(zygisk::Option::DLCLOSE_MODULE_LIBRARY);
            return;
        }

        const char *process = env_->GetStringUTFChars(args->nice_name, nullptr);
        if (process) {
            target_ = strcmp(process, kTargetProcess) == 0;
            env_->ReleaseStringUTFChars(args->nice_name, process);
        }
        if (!target_) {
            api_->setOption(zygisk::Option::DLCLOSE_MODULE_LIBRARY);
            return;
        }

        int moduleDir = api_->getModuleDir();
        if (moduleDir < 0) {
            LOGE("Zygisk getModuleDir failed");
            target_ = false;
            return;
        }
        dexFd_ = openat(moduleDir, kDexPath, O_RDONLY | O_CLOEXEC);
        nativeFd_ = openat(moduleDir, kNativePath, O_RDONLY | O_CLOEXEC);
        close(moduleDir);

        if (dexFd_ < 0 || nativeFd_ < 0) {
            LOGE("Could not open runtime payload FDs: dex=%d native=%d errno=%s",
                 dexFd_, nativeFd_, strerror(errno));
            if (dexFd_ >= 0) close(dexFd_);
            if (nativeFd_ >= 0) close(nativeFd_);
            dexFd_ = nativeFd_ = -1;
            target_ = false;
            return;
        }
        if (!api_->exemptFd(dexFd_) || !api_->exemptFd(nativeFd_)) {
            LOGE("Zygisk could not preserve payload FDs across specialization");
            close(dexFd_);
            close(nativeFd_);
            dexFd_ = nativeFd_ = -1;
            target_ = false;
            return;
        }
        LOGI("Target ADOFAI process selected; official APK remains untouched");
    }

    void postAppSpecialize(const zygisk::AppSpecializeArgs *) override {
        if (!target_ || !vm_ || dexFd_ < 0 || nativeFd_ < 0) return;
        pthread_t thread{};
        int result = pthread_create(&thread, nullptr, &ADOFAIEditorModule::threadEntry, this);
        if (result != 0) {
            LOGE("pthread_create failed: %d", result);
            return;
        }
        pthread_detach(thread);
    }

private:
    static void *threadEntry(void *opaque) {
        static_cast<ADOFAIEditorModule *>(opaque)->inject();
        return nullptr;
    }

    void inject() {
        JNIEnv *env = nullptr;
        if (vm_->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
            LOGE("Could not attach editor injection thread to JVM");
            return;
        }

        jobject application = waitForApplication(env);
        if (!application) {
            vm_->DetachCurrentThread();
            return;
        }

        octoberHandle_ = loadOctober(vm_, env, nativeFd_);
        if (!octoberHandle_) {
            vm_->DetachCurrentThread();
            return;
        }

        jobject loader = createDexLoader(
                env, application, dexFd_, &dexMap_, &dexMapSize_);
        if (!loader) {
            vm_->DetachCurrentThread();
            return;
        }
        dexLoader_ = env->NewGlobalRef(loader);

        jclass shellClass = loadClass(env, loader, "com.unity3d.player.MobileEditorShell");
        if (!shellClass || !registerPreviewNative(env, shellClass, octoberHandle_)) {
            vm_->DetachCurrentThread();
            return;
        }

        jclass bootstrap = loadClass(
                env, loader, "com.unity3d.player.ZygiskEditorBootstrap");
        if (!bootstrap) {
            vm_->DetachCurrentThread();
            return;
        }
        jmethodID start = env->GetStaticMethodID(bootstrap, "start", "()V");
        if (!start || clearException(env, "ZygiskEditorBootstrap.start lookup")) {
            vm_->DetachCurrentThread();
            return;
        }
        env->CallStaticVoidMethod(bootstrap, start);
        if (!clearException(env, "ZygiskEditorBootstrap.start")) {
            LOGI("ADOFAI mobile editor runtime injection initialized");
        }
        vm_->DetachCurrentThread();
    }

    zygisk::Api *api_ = nullptr;
    JNIEnv *env_ = nullptr;
    JavaVM *vm_ = nullptr;
    bool target_ = false;
    int dexFd_ = -1;
    int nativeFd_ = -1;
    void *octoberHandle_ = nullptr;
    void *dexMap_ = nullptr;
    size_t dexMapSize_ = 0;
    jobject dexLoader_ = nullptr;
};

REGISTER_ZYGISK_MODULE(ADOFAIEditorModule)
