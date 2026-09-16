#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-cache-native}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"
UPSTREAM="${ROOT}/.work/hitmargin-v240-cache-native"
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

SRC="${ROOT}/android/v240-dynamic-runtime/native/V240CacheLoader.cpp"
test -f "${SRC}"
python3 - "${SRC}" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text(encoding='utf-8')
for marker in (
    'JNI_OnLoad', 'GetEnv', 'universe.h', 'Loading::TryLoadByJNI',
    'Loading::AddOnLoadedEvent', 'RunReadOnlyAbiProbe',
    'Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport',
    'nativeProbe=cache-post-bnm-probe-only-v2', 'nativeStage=post-bnm-read-only-abi',
    'probeComplete=1', 'gameHooksInstalled=0', 'sfbOpenFiltersHookInstalled=0',
    'sfbFilterMemoryRead=0', 'sfbHookPolicy=disabled-unproven-call-abi',
    'abi.SFB.OpenFilePanel.filtersExact=', 'abi.SFB.ExtensionFilter.Name=',
    'abi.SFB.ExtensionFilter.Extensions=', 'abi.Settings.PauseMenu.ShowSettingsMenu0=',
    'abi.Mobile.CanvasScaler.SetScaleFactor1=', 'abi.Touch.EventSystem.RaycastAll=',
    'abi.FPS.Application.setTargetFrameRate1=',
    'abi.Event.scrCamera.SetCustomFrameRateBoolInt=',
):
    assert marker in s, marker
for forbidden in (
    'BasicHook', 'InstallAllHooks', 'InstallSfbHooks', 'InstallMobileHooks',
    'InstallExactOpenFiltersHook', 'HookOpenFilePanelFilters',
    'V240SettingsOverlay', 'V240EventCompat', 'V240TouchAssist', 'FileSelector',
    'FindClass', 'CallStatic', 'CallObject', 'NewGlobalRef', 'pthread_create',
    '.Call(', '.Set(', 'CreateNewObject',
    'struct ExtensionFilterValue', 'reinterpret_cast<Array<ExtensionFilterValue>*>(',
    'm_Items[i].Extensions', 'RunOpenPicker(',
):
    assert forbidden not in s, forbidden
PY

JNI="${UPSTREAM}/app/src/main/jni"
cp "${SRC}" "${JNI}/V240CacheLoader.cpp"

# Match the exact audited APK: Unity 2021.3.10f1.
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
    V240CacheLoader.cpp
LOCAL_CPPFLAGS := -std=c++20 -fexceptions -Oz -fvisibility=hidden -Wall -Wextra
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
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport'
strings "${OUT}/libv240fix.so" | grep -q 'nativeProbe=cache-post-bnm-probe-only-v2'
strings "${OUT}/libv240fix.so" | grep -q 'nativeStage=post-bnm-read-only-abi'
strings "${OUT}/libv240fix.so" | grep -q 'sfbHookPolicy=disabled-unproven-call-abi'
strings "${OUT}/libv240fix.so" | grep -q 'abi.SFB.OpenFilePanel.filtersExact='
strings "${OUT}/libv240fix.so" | grep -q 'abi.Settings.PauseMenu.ShowSettingsMenu0='
strings "${OUT}/libv240fix.so" | grep -q 'gameHooksInstalled=0'
if readelf -Ws "${OUT}/libv240fix.so" | grep -Eq 'Java_com_unity3d_player_V240SettingsOverlay_|Java_com_unity3d_player_V240EventCompat_'; then
  echo 'cache ABI probe unexpectedly exported feature activation JNI' >&2
  exit 1
fi
sha256sum "${OUT}/libv240fix.so" | tee "${OUT}/SHA256SUMS.txt"
