#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-fixed-native}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"
UPSTREAM="${ROOT}/.work/hitmargin-v240-native"
UPSTREAM_REPO="https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git"
UPSTREAM_SHA="74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"

if [[ -z "${SDK}" ]]; then echo 'ANDROID_SDK_ROOT is required' >&2; exit 2; fi
NDK_BUILD="${SDK}/ndk/${NDK_VERSION}/ndk-build"
test -x "${NDK_BUILD}"
mkdir -p "${ROOT}/.work"
if [[ ! -d "${UPSTREAM}/.git" ]]; then
  rm -rf "${UPSTREAM}"
  git clone --filter=blob:none "${UPSTREAM_REPO}" "${UPSTREAM}"
fi
git -C "${UPSTREAM}" fetch --prune origin
git -C "${UPSTREAM}" checkout --detach "${UPSTREAM_SHA}"
git -C "${UPSTREAM}" reset --hard "${UPSTREAM_SHA}"
git -C "${UPSTREAM}" clean -fdx
[[ "$(git -C "${UPSTREAM}" rev-parse HEAD)" == "${UPSTREAM_SHA}" ]]

EVENT_COMPAT_SOURCE="${ROOT}/android/v240-fixed-runtime/native/V240EventCompat.cpp"
POST_V240_PROBE_SOURCE="${ROOT}/android/v240-fixed-runtime/native/V240PostV240Probe.cpp"
python3 - "${EVENT_COMPAT_SOURCE}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
callback_marker = "Loading::AddOnLoadedEvent([]() {"
probe_call = "V240RunPostV240FeasibilityProbe();"
install_call = "ProbeAndInstallEventCompat();"
declaration = 'extern "C" void V240RunPostV240FeasibilityProbe();'

assert declaration in source, "post-v2.4 probe declaration missing"
start = source.index(callback_marker)
end = source.index("});", start)
callback = source[start:end]
assert probe_call in callback, "post-v2.4 probe is not gated by the proven BNM-loaded callback"
assert install_call in callback, "event compatibility install call missing from BNM-loaded callback"
assert callback.index(probe_call) < callback.index(install_call), \
    "read-only ABI probe must run before active event hook installation"
assert source.count(probe_call) == 2, \
    "only one post-BNM probe invocation plus its declaration are allowed"
PY

grep -q 'extern "C" void V240RunPostV240FeasibilityProbe()' "${POST_V240_PROBE_SOURCE}"
grep -q 'std::call_once(g_postV240ProbeOnce' "${POST_V240_PROBE_SOURCE}"

JNI="${UPSTREAM}/app/src/main/jni"
cp "${ROOT}/android/v240-fixed-runtime/native/V240Fix.cpp" "${JNI}/V240Fix.cpp"
cp "${ROOT}/android/v240-fixed-runtime/native/V240TouchAssist.cpp" "${JNI}/V240TouchAssist.cpp"
cp "${EVENT_COMPAT_SOURCE}" "${JNI}/V240EventCompat.cpp"
cp "${POST_V240_PROBE_SOURCE}" "${JNI}/V240PostV240Probe.cpp"

# The source snapshot was configured for a later Unity build. The user's audited APK
# is Unity 2021.3.10f1, so make BNM's IL2CPP layout match that exact runtime family.
python3 - "${JNI}/BNM/include/BNM/UserSettings/GlobalSettings.hpp" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('#define UNITY_VER 222 // 2022.2.x - 2022.3.x', '//#define UNITY_VER 222 // 2022.2.x - 2022.3.x')
s = s.replace('//#define UNITY_VER 213 // 2021.3.x', '#define UNITY_VER 213 // 2021.3.x')
s = s.replace('#define UNITY_PATCH_VER 32', '#define UNITY_PATCH_VER 10')
p.write_text(s)
PY

grep -q '^#define UNITY_VER 213' "${JNI}/BNM/include/BNM/UserSettings/GlobalSettings.hpp"
grep -q '^#define UNITY_PATCH_VER 10' "${JNI}/BNM/include/BNM/UserSettings/GlobalSettings.hpp"

cat > "${JNI}/Android.mk" <<'EOF'
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := dobby
LOCAL_SRC_FILES := libraries/$(TARGET_ARCH_ABI)/libdobby.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := v240fix
LOCAL_C_INCLUDES := $(LOCAL_PATH)/BNM/include \
    $(LOCAL_PATH)/BNM/external/include \
    $(LOCAL_PATH)/BNM/external \
    $(LOCAL_PATH)/BNM/external/utf8 \
    $(LOCAL_PATH)/BNM/src/BNM_data
LOCAL_STATIC_LIBRARIES := dobby
LOCAL_SRC_FILES := BNM/src/Class.cpp \
    BNM/src/ClassesManagement.cpp \
    BNM/src/Coroutine.cpp \
    BNM/src/Delegates.cpp \
    BNM/src/Defaults.cpp \
    BNM/src/EventBase.cpp \
    BNM/src/Exceptions.cpp \
    BNM/src/FieldBase.cpp \
    BNM/src/Hooks.cpp \
    BNM/src/Image.cpp \
    BNM/src/Internals.cpp \
    BNM/src/Loading.cpp \
    BNM/src/MethodBase.cpp \
    BNM/src/MonoStructures.cpp \
    BNM/src/PropertyBase.cpp \
    BNM/src/UnityStructures.cpp \
    BNM/src/Utils.cpp \
    V240Fix.cpp \
    V240TouchAssist.cpp \
    V240EventCompat.cpp \
    V240PostV240Probe.cpp \
    Logger.cpp
LOCAL_CPPFLAGS := -std=c++20 -fexceptions -O2
LOCAL_LDLIBS := -llog -ldl
include $(BUILD_SHARED_LIBRARY)
EOF

rm -rf "${OUT}"
mkdir -p "${OUT}"
"${NDK_BUILD}" -C "${JNI}" -j2
LIB="${UPSTREAM}/app/src/main/libs/arm64-v8a/libv240fix.so"
if [[ ! -s "${LIB}" ]]; then LIB="${UPSTREAM}/app/src/main/obj/local/arm64-v8a/libv240fix.so"; fi
test -s "${LIB}"
cp "${LIB}" "${OUT}/libv240fix.so"
readelf -h "${OUT}/libv240fix.so" | grep -q 'AArch64'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'JNI_OnLoad'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_com_unity3d_player_V240SettingsOverlay_nativeApply'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_com_unity3d_player_V240EventCompat_nativeIsSetFrameRateBackportReady'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'V240RunPostV240FeasibilityProbe'
strings "${OUT}/libv240fix.so" | grep -q 'V240: EmitParticle ABI'
strings "${OUT}/libv240fix.so" | grep -q 'V240: SetInputEvent feasibility'
strings "${OUT}/libv240fix.so" | grep -q 'V240: SetFilterAdvanced feasibility'
sha256sum "${OUT}/libv240fix.so" | tee "${OUT}/SHA256SUMS.txt"
