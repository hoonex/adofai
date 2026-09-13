#include <jni.h>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "universe.h"
#include "Logger.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(
        JNIEnv* env, jclass overlayClass, jboolean enabled, jfloat radiusPx);

namespace {
constexpr char kSetFrameRateMarkerPrefix[] = "__V240_SET_FRAME_RATE__:";

std::once_flag g_registerOnce;
std::atomic<bool> g_probeComplete{false};
std::atomic<bool> g_setFrameRateBackportReady{false};

Field<String*> g_callMethodName;
Field<IL2CPP::Il2CppObject*> g_cameraInstanceField;
Property<IL2CPP::Il2CppObject*> g_cameraInstanceProperty;
Method<void> g_setCustomFrameRateInt;
Method<void> g_setCustomFrameRateFloat;
void (*g_oldCallMethodStartEffect)(IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*) = nullptr;

bool SameManagedType(const Class& left, const Class& right) {
    return left && right && left.GetClass() == right.GetClass();
}

bool IsLowerUuidToken(const std::string& value, std::size_t offset, std::size_t length) {
    if (length != 36 || offset + length > value.size()) return false;
    for (std::size_t i = 0; i < length; ++i) {
        const char ch = value[offset + i];
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (ch != '-') return false;
            continue;
        }
        const bool digit = ch >= '0' && ch <= '9';
        const bool hex = ch >= 'a' && ch <= 'f';
        if (!digit && !hex) return false;
    }
    return true;
}

bool ParseSetFrameRateMarker(String* value, bool* enabledOut, float* frameRateOut) {
    if (!value || !enabledOut || !frameRateOut) return false;
    const std::string text = value->str();
    const std::size_t prefixLength = sizeof(kSetFrameRateMarkerPrefix) - 1;
    if (text.size() <= prefixLength + 39 ||
        text.compare(0, prefixLength, kSetFrameRateMarkerPrefix) != 0) {
        return false;
    }

    const std::size_t tokenStart = prefixLength;
    const std::size_t tokenEnd = text.find(':', tokenStart);
    if (tokenEnd == std::string::npos ||
        !IsLowerUuidToken(text, tokenStart, tokenEnd - tokenStart)) {
        return false;
    }

    const std::size_t enabledIndex = tokenEnd + 1;
    if (enabledIndex + 2 >= text.size()) return false;
    const char enabled = text[enabledIndex];
    if ((enabled != '0' && enabled != '1') || text[enabledIndex + 1] != ':') return false;

    const char* number = text.c_str() + enabledIndex + 2;
    char* end = nullptr;
    const float frameRate = std::strtof(number, &end);
    if (!end || end == number || *end != '\0' || !std::isfinite(frameRate)) return false;

    *enabledOut = enabled == '1';
    *frameRateOut = frameRate;
    return true;
}

IL2CPP::Il2CppObject* GetCameraInstance() {
    if (g_cameraInstanceProperty.IsValid()) {
        Property<IL2CPP::Il2CppObject*> property = g_cameraInstanceProperty;
        IL2CPP::Il2CppObject* camera = property.Get();
        if (camera) return camera;
    }
    if (g_cameraInstanceField.IsValid()) {
        Field<IL2CPP::Il2CppObject*> field = g_cameraInstanceField;
        return field.Get();
    }
    return nullptr;
}

bool ApplySetFrameRate(bool enabled, float frameRate) {
    IL2CPP::Il2CppObject* camera = GetCameraInstance();
    if (!camera) return false;

    if (g_setCustomFrameRateFloat.IsValid()) {
        Method<void> method = g_setCustomFrameRateFloat;
        method[camera].Call(enabled, frameRate);
        return true;
    }
    if (g_setCustomFrameRateInt.IsValid()) {
        Method<void> method = g_setCustomFrameRateInt;
        method[camera].Call(enabled, static_cast<int>(std::lround(frameRate)));
        return true;
    }
    return false;
}

void HookCallMethodStartEffect(IL2CPP::Il2CppObject* self, IL2CPP::Il2CppObject* planet) {
    if (self && g_callMethodName.IsValid()) {
        Field<String*> methodNameField = g_callMethodName;
        String* methodName = methodNameField[self].Get();
        bool enabled = false;
        float frameRate = 0.0f;
        if (ParseSetFrameRateMarker(methodName, &enabled, &frameRate)) {
            // Marker names deliberately do not resolve to a real Level method. Once recognized,
            // never fall through to the original reflection path: failure must stay fail-closed.
            if (!ApplySetFrameRate(enabled, frameRate)) {
                LOGW("V240: SetFrameRate marker suppressed because camera runtime is unavailable");
            }
            return;
        }
    }

    if (g_oldCallMethodStartEffect) g_oldCallMethodStartEffect(self, planet);
}

void ProbeAndInstallEventCompat() {
    Class levelEvent("ADOFAI", "LevelEvent");
    Class scrPlanet("", "scrPlanet");
    Class ffxCallMethod("", "ffxCallMethod");
    Class scrCamera("", "scrCamera");
    Class scrFloor("", "scrFloor");
    Class scrDecorationManager("", "scrDecorationManager");
    Class scrTextDecoration("", "scrTextDecoration");

    auto callMethodName = ffxCallMethod ? ffxCallMethod.GetField("methodName") : FieldBase{};
    auto callMethodDecode = (ffxCallMethod && levelEvent)
            ? ffxCallMethod.GetMethod("Decode", {levelEvent.GetCompileTimeClass()})
            : MethodBase{};
    auto callMethodStartEffect = (ffxCallMethod && scrPlanet)
            ? ffxCallMethod.GetMethod("StartEffect", {scrPlanet.GetCompileTimeClass()})
            : MethodBase{};
    auto cameraInstanceField = scrCamera ? scrCamera.GetField("instance") : FieldBase{};
    auto cameraInstanceProperty = scrCamera ? scrCamera.GetProperty("instance") : PropertyBase{};
    auto setCustomFrameRateInt = scrCamera
            ? scrCamera.GetMethod("SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()})
            : MethodBase{};
    auto setCustomFrameRateFloat = scrCamera
            ? scrCamera.GetMethod("SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<float>()})
            : MethodBase{};
    auto floorLengthMult = scrFloor ? scrFloor.GetField("lengthMult") : FieldBase{};
    auto floorWidthMult = scrFloor ? scrFloor.GetField("widthMult") : FieldBase{};
    auto decorationManagerInstanceField = scrDecorationManager
            ? scrDecorationManager.GetField("instance") : FieldBase{};
    auto decorationManagerInstanceProperty = scrDecorationManager
            ? scrDecorationManager.GetProperty("instance") : PropertyBase{};
    auto getTaggedDecorations = scrDecorationManager
            ? scrDecorationManager.GetMethod("GetTaggedDecorations", 1) : MethodBase{};
    auto setTextString = scrTextDecoration
            ? scrTextDecoration.GetMethod("SetText", {Defaults::Get<String*>()}) : MethodBase{};

    const Class stringClass = Defaults::Get<String*>().ToClass();
    const Class floatClass = Defaults::Get<float>().ToClass();
    const Class voidClass = Defaults::Get<void>().ToClass();
    const bool methodNameString = callMethodName.IsValid()
            && SameManagedType(callMethodName.GetType(), stringClass);
    const bool cameraInstanceFieldTyped = cameraInstanceField.IsValid()
            && SameManagedType(cameraInstanceField.GetType(), scrCamera);
    const bool cameraInstancePropertyTyped = cameraInstanceProperty.IsValid()
            && SameManagedType(cameraInstanceProperty.GetType(), scrCamera);
    const bool cameraInstanceTyped = cameraInstanceFieldTyped || cameraInstancePropertyTyped;
    const bool frameRateMethod = setCustomFrameRateInt.IsValid() || setCustomFrameRateFloat.IsValid();
    const bool schedulerSurface = ffxCallMethod
            && methodNameString
            && callMethodDecode.IsValid()
            && callMethodStartEffect.IsValid();
    const bool floorLengthMultFloat = floorLengthMult.IsValid()
            && SameManagedType(floorLengthMult.GetType(), floatClass);
    const bool floorWidthMultFloat = floorWidthMult.IsValid()
            && SameManagedType(floorWidthMult.GetType(), floatClass);
    const bool tileDimensionsSurface = scrFloor
            && floorLengthMultFloat
            && floorWidthMultFloat;
    const bool decorationManagerFieldTyped = decorationManagerInstanceField.IsValid()
            && SameManagedType(decorationManagerInstanceField.GetType(), scrDecorationManager);
    const bool decorationManagerPropertyTyped = decorationManagerInstanceProperty.IsValid()
            && SameManagedType(decorationManagerInstanceProperty.GetType(), scrDecorationManager);
    const bool decorationManagerSingletonTyped = decorationManagerFieldTyped
            || decorationManagerPropertyTyped;
    const bool setTextStringVoid = setTextString.IsValid()
            && SameManagedType(setTextString.GetReturnType(), voidClass);
    const bool setTextSurface = scrDecorationManager
            && scrTextDecoration
            && decorationManagerSingletonTyped
            && getTaggedDecorations.IsValid()
            && setTextStringVoid;

    LOGD("V240: event compat probe ffxCallMethod=%d methodNameString=%d Decode=%d StartEffect=%d scrCamera=%d instanceField=%d instanceProperty=%d SetCustomFrameRate=%d bool-int=%d bool-float=%d",
         ffxCallMethod ? 1 : 0,
         methodNameString ? 1 : 0,
         callMethodDecode.IsValid() ? 1 : 0,
         callMethodStartEffect.IsValid() ? 1 : 0,
         scrCamera ? 1 : 0,
         cameraInstanceFieldTyped ? 1 : 0,
         cameraInstancePropertyTyped ? 1 : 0,
         frameRateMethod ? 1 : 0,
         setCustomFrameRateInt.IsValid() ? 1 : 0,
         setCustomFrameRateFloat.IsValid() ? 1 : 0);
    LOGD("V240: TileDimensions ABI scrFloor=%d lengthMultFloat=%d widthMultFloat=%d compatible=%d",
         scrFloor ? 1 : 0,
         floorLengthMultFloat ? 1 : 0,
         floorWidthMultFloat ? 1 : 0,
         tileDimensionsSurface ? 1 : 0);
    LOGD("V240: SetText ABI scrDecorationManager=%d instanceField=%d instanceProperty=%d GetTaggedDecorations=%d scrTextDecoration=%d SetTextStringVoid=%d compatible=%d",
         scrDecorationManager ? 1 : 0,
         decorationManagerFieldTyped ? 1 : 0,
         decorationManagerPropertyTyped ? 1 : 0,
         getTaggedDecorations.IsValid() ? 1 : 0,
         scrTextDecoration ? 1 : 0,
         setTextStringVoid ? 1 : 0,
         setTextSurface ? 1 : 0);

    if (!schedulerSurface || !cameraInstanceTyped || !frameRateMethod) {
        g_probeComplete.store(true, std::memory_order_release);
        LOGW("V240: SetFrameRate execution backport unavailable; opaque preserve-only mode retained");
        return;
    }

    g_callMethodName = callMethodName;
    if (cameraInstanceFieldTyped) g_cameraInstanceField = cameraInstanceField;
    if (cameraInstancePropertyTyped) g_cameraInstanceProperty = cameraInstanceProperty;
    g_setCustomFrameRateInt = setCustomFrameRateInt;
    g_setCustomFrameRateFloat = setCustomFrameRateFloat;
    BasicHook(callMethodStartEffect, HookCallMethodStartEffect, g_oldCallMethodStartEffect);

    const bool hooked = g_oldCallMethodStartEffect != nullptr;
    g_setFrameRateBackportReady.store(hooked, std::memory_order_release);
    g_probeComplete.store(true, std::memory_order_release);
    if (hooked) {
        LOGD("V240: SetFrameRate execution backport ready (fail-closed CallMethod marker hook)");
    } else {
        LOGW("V240: SetFrameRate scheduler hook failed; opaque preserve-only mode retained");
    }
}
} // namespace

void RegisterV240EventCompat() {
    std::call_once(g_registerOnce, []() {
        Loading::AddOnLoadedEvent([]() { ProbeAndInstallEventCompat(); });
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_unity3d_player_V240EventCompat_nativeRegister(JNIEnv* env, jclass) {
    // V240Bootstrap calls this synchronously immediately after System.loadLibrary(). Register
    // the enhanced touch callback here as well so it cannot lose the IL2CPP-loaded event while
    // waiting for UnityPlayer.currentActivity / the settings overlay to become ready.
    if (env) {
        Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(
                env, nullptr, JNI_TRUE, 0.0f);
    }
    RegisterV240EventCompat();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_V240EventCompat_nativeIsSetFrameRateBackportReady(
        JNIEnv*, jclass) {
    RegisterV240EventCompat();
    if (!g_probeComplete.load(std::memory_order_acquire)) return JNI_FALSE;
    return g_setFrameRateBackportReady.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}
