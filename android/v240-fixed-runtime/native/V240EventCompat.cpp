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
Field<IL2CPP::Il2CppObject*> g_cameraInstance;
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

bool ApplySetFrameRate(bool enabled, float frameRate) {
    Field<IL2CPP::Il2CppObject*> cameraInstance = g_cameraInstance;
    IL2CPP::Il2CppObject* camera = cameraInstance.Get();
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

    auto callMethodName = ffxCallMethod ? ffxCallMethod.GetField("methodName") : FieldBase{};
    auto callMethodDecode = (ffxCallMethod && levelEvent)
            ? ffxCallMethod.GetMethod("Decode", {levelEvent.GetCompileTimeClass()})
            : MethodBase{};
    auto callMethodStartEffect = (ffxCallMethod && scrPlanet)
            ? ffxCallMethod.GetMethod("StartEffect", {scrPlanet.GetCompileTimeClass()})
            : MethodBase{};
    auto cameraInstance = scrCamera ? scrCamera.GetField("instance") : FieldBase{};
    auto setCustomFrameRateInt = scrCamera
            ? scrCamera.GetMethod("SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()})
            : MethodBase{};
    auto setCustomFrameRateFloat = scrCamera
            ? scrCamera.GetMethod("SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<float>()})
            : MethodBase{};
    auto floorLengthMult = scrFloor ? scrFloor.GetField("lengthMult") : FieldBase{};
    auto floorWidthMult = scrFloor ? scrFloor.GetField("widthMult") : FieldBase{};

    const Class stringClass = Defaults::Get<String*>().ToClass();
    const Class floatClass = Defaults::Get<float>().ToClass();
    const bool methodNameString = callMethodName.IsValid()
            && SameManagedType(callMethodName.GetType(), stringClass);
    const bool cameraInstanceTyped = cameraInstance.IsValid()
            && SameManagedType(cameraInstance.GetType(), scrCamera);
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

    LOGD("V240: event compat probe ffxCallMethod=%d methodNameString=%d Decode=%d StartEffect=%d scrCamera=%d instance=%d SetCustomFrameRate=%d bool-int=%d bool-float=%d",
         ffxCallMethod ? 1 : 0,
         methodNameString ? 1 : 0,
         callMethodDecode.IsValid() ? 1 : 0,
         callMethodStartEffect.IsValid() ? 1 : 0,
         scrCamera ? 1 : 0,
         cameraInstanceTyped ? 1 : 0,
         frameRateMethod ? 1 : 0,
         setCustomFrameRateInt.IsValid() ? 1 : 0,
         setCustomFrameRateFloat.IsValid() ? 1 : 0);
    LOGD("V240: TileDimensions ABI scrFloor=%d lengthMultFloat=%d widthMultFloat=%d compatible=%d",
         scrFloor ? 1 : 0,
         floorLengthMultFloat ? 1 : 0,
         floorWidthMultFloat ? 1 : 0,
         tileDimensionsSurface ? 1 : 0);

    if (!schedulerSurface || !cameraInstanceTyped || !frameRateMethod) {
        g_probeComplete.store(true, std::memory_order_release);
        LOGW("V240: SetFrameRate execution backport unavailable; opaque preserve-only mode retained");
        return;
    }

    g_callMethodName = callMethodName;
    g_cameraInstance = cameraInstance;
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
