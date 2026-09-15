#include <jni.h>
#include <atomic>

#include "universe.h"

using namespace BNM;

namespace {
std::atomic<bool> g_bnmLoadRequested{false};
std::atomic<bool> g_bnmLoadedCallback{false};
}

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
    });
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (env == nullptr) return nullptr;
    const bool requested = g_bnmLoadRequested.load(std::memory_order_acquire);
    const bool loaded = g_bnmLoadedCallback.load(std::memory_order_acquire);
    const char* report = loaded
            ? "nativeProbe=cache-bnm-loader-only\nnativeStage=bnm-loader-only\nbnmLoadRequested=1\nbnmLoadedCallback=1\ngameHooksInstalled=0\n"
            : requested
                    ? "nativeProbe=cache-bnm-loader-only\nnativeStage=bnm-loader-only\nbnmLoadRequested=1\nbnmLoadedCallback=0\ngameHooksInstalled=0\n"
                    : "nativeProbe=cache-bnm-loader-only\nnativeStage=bnm-loader-only\nbnmLoadRequested=0\nbnmLoadedCallback=0\ngameHooksInstalled=0\n";
    return env->NewStringUTF(report);
}
