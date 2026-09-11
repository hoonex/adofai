#include <jni.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

#include "universe.h"
#include "Logger.h"

using namespace BNM;
using namespace BNM::Structures::Mono;
using namespace BNM::Structures::Unity;

namespace {
std::atomic<bool> g_enabled{true};
std::atomic<float> g_radiusPx{0.0f};
std::atomic<bool> g_installed{false};
std::atomic<bool> g_nativeApplyInProgress{false};
std::once_flag g_registrationOnce;

JavaVM* g_touchVm = nullptr;
jclass g_overlayClass = nullptr;
jmethodID g_refreshMethod = nullptr;
std::mutex g_jniStateMutex;

Method<int> g_getTouchCount;
Method<String*> g_getSceneName;
Property<Vector2> g_pointerPosition;
Property<int> g_listCount;
Method<void> g_listClear;
void (*g_oldRaycastAll)(IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*) = nullptr;

constexpr int64_t kEditorSceneCacheNs = 250000000LL;
std::atomic<int64_t> g_editorSceneCacheAtNs{0};
std::atomic<bool> g_editorSceneCacheValue{false};
thread_local bool g_insideRaycastHook = false;

int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
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

bool HasActiveTouch() {
    if (!g_getTouchCount.IsValid()) return false;
    return g_getTouchCount.Call() > 0;
}

void CaptureJavaRefresh(JNIEnv* env, jclass overlayClass) {
    if (!env || !overlayClass) return;
    std::lock_guard<std::mutex> lock(g_jniStateMutex);
    if (!g_touchVm) env->GetJavaVM(&g_touchVm);
    if (!g_overlayClass) {
        g_overlayClass = reinterpret_cast<jclass>(env->NewGlobalRef(overlayClass));
    }
    if (g_overlayClass && !g_refreshMethod) {
        g_refreshMethod = env->GetStaticMethodID(g_overlayClass, "refresh", "()V");
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            g_refreshMethod = nullptr;
        }
    }
}

void RequestJavaRefresh() {
    JavaVM* vm;
    jclass overlayClass;
    jmethodID refreshMethod;
    {
        std::lock_guard<std::mutex> lock(g_jniStateMutex);
        vm = g_touchVm;
        overlayClass = g_overlayClass;
        refreshMethod = g_refreshMethod;
    }
    if (!vm || !overlayClass || !refreshMethod) return;

    JNIEnv* env = nullptr;
    bool attached = false;
    const jint state = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (state == JNI_EDETACHED) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return;
        attached = true;
    } else if (state != JNI_OK || !env) {
        return;
    }

    env->CallStaticVoidMethod(overlayClass, refreshMethod);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        LOGW("V240: delayed touch-assist refresh failed");
    }
    if (attached) vm->DetachCurrentThread();
}

bool RaycastAt(IL2CPP::Il2CppObject* eventSystem,
               IL2CPP::Il2CppObject* eventData,
               IL2CPP::Il2CppObject* results,
               const Vector2& original,
               float dx,
               float dy) {
    if (!g_oldRaycastAll) return false;
    g_listClear[results].Call();
    g_pointerPosition[eventData].Set(Vector2(original.x + dx, original.y + dy));
    g_oldRaycastAll(eventSystem, eventData, results);
    return g_listCount[results].Get() > 0;
}

void HookRaycastAll(IL2CPP::Il2CppObject* eventSystem,
                    IL2CPP::Il2CppObject* eventData,
                    IL2CPP::Il2CppObject* results) {
    if (!g_oldRaycastAll) return;
    if (g_insideRaycastHook) {
        g_oldRaycastAll(eventSystem, eventData, results);
        return;
    }

    g_insideRaycastHook = true;
    g_oldRaycastAll(eventSystem, eventData, results);

    const float radius = g_radiusPx.load(std::memory_order_relaxed);
    const bool canExpand = g_enabled.load(std::memory_order_relaxed)
            && radius >= 1.0f
            && eventSystem
            && eventData
            && results
            && g_pointerPosition.IsValid()
            && g_listCount.IsValid()
            && g_listClear.IsValid()
            && HasActiveTouch()
            && IsEditorScene()
            && g_listCount[results].Get() == 0;

    if (canExpand) {
        const Vector2 original = g_pointerPosition[eventData].Get();
        const float quarter = radius * 0.25f;
        const float half = radius * 0.55f;
        const float diagonal = radius * 0.70710678f;
        bool hit = false;

        const float cardinals[][2] = {
                { quarter, 0.0f }, {-quarter, 0.0f}, {0.0f, quarter}, {0.0f, -quarter},
                { half, 0.0f }, {-half, 0.0f}, {0.0f, half}, {0.0f, -half},
                { radius, 0.0f }, {-radius, 0.0f}, {0.0f, radius}, {0.0f, -radius},
                { diagonal, diagonal }, {-diagonal, diagonal},
                { diagonal, -diagonal }, {-diagonal, -diagonal},
        };

        for (const auto& offset : cardinals) {
            if (RaycastAt(eventSystem, eventData, results, original, offset[0], offset[1])) {
                hit = true;
                break;
            }
        }

        g_pointerPosition[eventData].Set(original);
        if (!hit) g_listClear[results].Call();
    }

    g_insideRaycastHook = false;
}

bool InstallTouchAssistHook() {
    if (g_installed.load(std::memory_order_acquire)) return true;

    Class eventSystem("UnityEngine.EventSystems", "EventSystem");
    Class pointerEventData("UnityEngine.EventSystems", "PointerEventData");
    Class raycastResult("UnityEngine.EventSystems", "RaycastResult");
    Class list("System.Collections.Generic", "List`1");
    Class input("UnityEngine", "Input");

    if (!eventSystem || !pointerEventData || !raycastResult || !list || !input) {
        return false;
    }

    Class listRaycastResult = list.GetGeneric({raycastResult.GetCompileTimeClass()});
    if (!listRaycastResult) return false;

    auto raycastAll = eventSystem.GetMethod("RaycastAll");
    g_pointerPosition = pointerEventData.GetProperty("position");
    g_listCount = listRaycastResult.GetProperty("Count");
    g_listClear = listRaycastResult.GetMethod("Clear", 0);
    g_getTouchCount = input.GetMethod("get_touchCount", 0);
    g_getSceneName = Class("", "ADOBase").GetMethod("get_sceneName");

    if (!raycastAll.IsValid() || !g_pointerPosition.IsValid() ||
        !g_listCount.IsValid() || !g_listClear.IsValid() || !g_getTouchCount.IsValid()) {
        return false;
    }

    BasicHook(raycastAll, HookRaycastAll, g_oldRaycastAll);
    if (!g_oldRaycastAll) return false;

    g_installed.store(true, std::memory_order_release);
    LOGD("V240: enhanced touch EventSystem.RaycastAll hook installed");
    // If BNM became ready after Java already applied the legacy fallback scale,
    // ask Java to re-apply settings. It will now see this hook as installed and
    // neutralize the old proximity expansion instead of stacking both systems.
    if (!g_nativeApplyInProgress.load(std::memory_order_acquire)) RequestJavaRefresh();
    return true;
}

void RegisterTouchAssistHook() {
    std::call_once(g_registrationOnce, []() {
        Loading::AddOnLoadedEvent([]() {
            if (!InstallTouchAssistHook()) {
                LOGW("V240: enhanced touch hook unavailable; legacy touch fallback remains active");
            }
        });
    });
}
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(
        JNIEnv* env, jclass overlayClass, jboolean enabled, jfloat radiusPx) {
    CaptureJavaRefresh(env, overlayClass);
    g_enabled.store(enabled == JNI_TRUE, std::memory_order_relaxed);
    g_radiusPx.store(std::max(0.0f, std::min(160.0f, static_cast<float>(radiusPx))),
                     std::memory_order_relaxed);
    g_nativeApplyInProgress.store(true, std::memory_order_release);
    RegisterTouchAssistHook();
    g_nativeApplyInProgress.store(false, std::memory_order_release);
    return g_installed.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}
