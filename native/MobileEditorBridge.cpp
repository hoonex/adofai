#include "MobileEditorBridge.h"

#include <jni.h>
#include <mutex>
#include <string>

#include "Logger.h"
#include "universe.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

namespace {
std::mutex g_previewMutex;
std::string g_pendingPreviewPath;
bool g_previewHookInstalled = false;
int (*g_oldTouchCount)() = nullptr;

bool QueuePreview(const char* path) {
    if (!g_previewHookInstalled || !path || !path[0]) return false;
    std::lock_guard<std::mutex> lock(g_previewMutex);
    g_pendingPreviewPath = path;
    return true;
}

bool TakePendingPreview(std::string& out) {
    std::lock_guard<std::mutex> lock(g_previewMutex);
    if (g_pendingPreviewPath.empty()) return false;
    out.swap(g_pendingPreviewPath);
    return true;
}

bool SetRequiredPreviewState(const std::string& path) {
    auto gcs = Class("", "GCS");
    if (!gcs) {
        LOGE("Mobile editor preview: GCS not found");
        return false;
    }

    auto customLevelIndex = gcs.GetField("customLevelIndex");
    auto internalLevelName = gcs.GetField("internalLevelName");
    auto customLevelId = gcs.GetField("customLevelId");
    auto sceneToLoad = gcs.GetField("sceneToLoad");
    if (!customLevelIndex.IsValid() || !internalLevelName.IsValid() ||
        !customLevelId.IsValid() || !sceneToLoad.IsValid()) {
        LOGE("Mobile editor preview: required GCS fields are missing");
        return false;
    }

    auto controllerClass = Class("", "scrController");
    if (!controllerClass) {
        LOGE("Mobile editor preview: scrController not found");
        return false;
    }

    auto getInstance = controllerClass.GetMethod("get_instance", 0);
    auto loadCustomLevel = controllerClass.GetMethod("LoadCustomLevel", 3);
    if (!getInstance.IsValid() || !loadCustomLevel.IsValid()) {
        LOGE("Mobile editor preview: current custom-level methods are missing");
        return false;
    }

    auto controller = getInstance.cast<IL2CPP::Il2CppObject*>().Call();
    if (!controller) {
        LOGE("Mobile editor preview: scrController instance is null");
        return false;
    }

    String* levelPath = CreateMonoString(path);
    String* gameScene = CreateMonoString("scnGame");
    if (!levelPath || !gameScene) {
        LOGE("Mobile editor preview: required managed strings could not be created");
        return false;
    }

    // Mutate global game state only after every required class, field, method,
    // instance and managed string has been resolved. Validation failures above
    // therefore leave GCS untouched instead of partially entering custom-level mode.
    customLevelIndex.cast<int>().Set(0);
    internalLevelName.cast<String*>().Set(nullptr);
    customLevelId.cast<String*>().Set(nullptr);
    sceneToLoad.cast<String*>().Set(gameScene);

    auto fromBundle = gcs.GetField("loadCustomFromBundle");
    if (fromBundle.IsValid()) fromBundle.cast<bool>().Set(false);

    loadCustomLevel.cast<void>()[controller].Call(levelPath, static_cast<String*>(nullptr), false);
    LOGD("Mobile editor preview queued into current runtime: %s", path.c_str());
    return true;
}

void DrainPreviewQueueOnGameThread() {
    std::string path;
    if (!TakePendingPreview(path)) return;
    if (!SetRequiredPreviewState(path)) {
        LOGE("Mobile editor preview request failed closed: %s", path.c_str());
    }
}

int HookedTouchCount() {
    DrainPreviewQueueOnGameThread();
    return g_oldTouchCount ? g_oldTouchCount() : 0;
}
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_MobileEditorShell_nativeQueuePreview(
    JNIEnv* env, jclass, jstring path) {
    if (!env || !path) return JNI_FALSE;
    const char* raw = env->GetStringUTFChars(path, nullptr);
    if (!raw) return JNI_FALSE;
    bool queued = QueuePreview(raw);
    env->ReleaseStringUTFChars(path, raw);
    return queued ? JNI_TRUE : JNI_FALSE;
}

void InstallMobileEditorPreviewBridge() {
    if (g_previewHookInstalled) return;

    auto touchCount = Class("UnityEngine", "Input").GetMethod("get_touchCount", 0);
    if (!touchCount.IsValid()) {
        LOGE("Mobile editor preview disabled: UnityEngine.Input.get_touchCount missing");
        return;
    }

    BasicHook(touchCount, HookedTouchCount, g_oldTouchCount);
    if (!g_oldTouchCount) {
        LOGE("Mobile editor preview disabled: touchCount hook did not return original");
        return;
    }

    g_previewHookInstalled = true;
    LOGD("Mobile editor preview bridge installed on Unity game-thread input poll");
}
