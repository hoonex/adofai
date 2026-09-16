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
BRIDGE="${ROOT}/android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/DirectDocumentBridge.java"
ENTRY="${ROOT}/android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
test -f "${SRC}"
test -f "${BRIDGE}"
test -f "${ENTRY}"
python3 - "${SRC}" "${BRIDGE}" "${ENTRY}" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text(encoding='utf-8')
j = Path(sys.argv[2]).read_text(encoding='utf-8')
e = Path(sys.argv[3]).read_text(encoding='utf-8')
for marker in (
    'JNI_OnLoad', 'g_vm = vm', 'universe.h', 'Loading::TryLoadByJNI',
    'Loading::AddOnLoadedEvent', 'ReconcileInstallState',
    'Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport',
    'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeRegisterDynamicBridge',
    'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeReconcileDynamicRuntime',
    'nativeProbe=cache-post-bnm-sfb-dynamic-import-uihit-v1',
    'nativeStage=post-bnm-dynamic-document-and-uihit', 'abiProbeRevision=9',
    'sfbHookPolicy=dynamic-document-preprocess-before-bind',
    'sfbPickerBackend=dynamic-document', 'sfbEmbeddedBridgeBypassed=1',
    'uiHitPolicy=pinned-upstream-eventsystem-raycast',
    'uiHitSourceCommit=74bcc7a0d8c8be1267504e21e28a35e199b5d4eb',
    'BasicHook(openFilters, HookOpenFilePanelFilters, g_oldOpenFilters)',
    'BasicHook(uiMethod, HookUiHit, g_oldUiHit)',
    'ReadFilterExtensions', 'filters->m_Items[i].Extensions',
    'RunDynamicPicker(multiselect, extensions)',
    'g_pointerPosition[eventData].Set(position)',
    'g_raycastAll[eventSystem].Call(eventData, results)',
    'sfb-r9-install.pending', 'sfb-r9-call.pending',
    'uihit-r9-install.pending', 'uihit-r9-call.pending',
    'O_CREAT | O_EXCL | O_CLOEXEC', 'fsync(fd)',
):
    assert marker in s, marker
assert s.count('BasicHook(') == 2
assert 'g_oldOpenFilters(' not in s
assert '"com/unity3d/player/FileSelector"' not in s
assert '"com/unity3d/player/V240AndroidBridge"' not in s
for forbidden in ('InstallAllHooks', 'InstallSfbHooks', 'InstallMobileHooks', 'V240SettingsOverlay', 'V240EventCompat'):
    assert forbidden not in s, forbidden
for marker in (
    'Intent.ACTION_OPEN_DOCUMENT', 'V240ChartBackport', 'V240HallLegacyFix',
    'V240OpaqueEventBridge', 'V240MapCompatibility', 'V240AndroidBridge',
    'backportForV240', 'applyIfNeeded', 'prepareForV240', 'repairMap', 'bindSave',
    'BIND_SAVE.setAccessible(true)', 'MAX_FILES = 128',
    'MAX_BYTES = 512L * 1024L * 1024L',
    'directBridge=dynamic-document-v1',
):
    assert marker in j, marker
assert 'Intent.ACTION_OPEN_DOCUMENT_TREE' not in j
assert j.index('invokeBoolean(BACKPORT') < j.index('BIND_SAVE.invoke')
assert j.index('invokeBoolean(HALL_FIX') < j.index('BIND_SAVE.invoke')
assert j.index('invokeBoolean(OPAQUE_PREPARE') < j.index('BIND_SAVE.invoke')
assert j.index('invokeVoid(MAP_REPAIR') < j.index('BIND_SAVE.invoke')
for marker in ('nativeRegisterDynamicBridge(DirectDocumentBridge.class)', 'nativeReconcileDynamicRuntime()'):
    assert marker in e, marker
assert 'System.load(' not in e
assert 'System.loadLibrary(' not in e
PY

JNI="${UPSTREAM}/app/src/main/jni"
cp "${SRC}" "${JNI}/V240CacheLoader.cpp"
# Keep the committed source readable while fixing the JNI method-id/text diagnostic name
# collision in the isolated build copy. This is a bounded build-only reconciliation.
python3 - "${JNI}/V240CacheLoader.cpp" <<'PY_NATIVE_NAME_FIX'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
s = s.replace('jmethodID g_dynamicDiagnostics = nullptr;',
              'jmethodID g_dynamicDiagnosticsMethod = nullptr;')
s = s.replace('g_dynamicAwait != nullptr && g_dynamicDiagnostics != nullptr;',
              'g_dynamicAwait != nullptr && g_dynamicDiagnosticsMethod != nullptr;')
s = s.replace('diagnostics = g_dynamicDiagnostics;',
              'diagnostics = g_dynamicDiagnosticsMethod;')
s = s.replace('g_dynamicDiagnostics = diagnostics;',
              'g_dynamicDiagnosticsMethod = diagnostics;')
p.write_text(s, encoding='utf-8')
PY_NATIVE_NAME_FIX

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

cat > "${JNI}/Android.mk" <<'EOF_MK'
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
EOF_MK

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
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeRegisterDynamicBridge'
readelf -Ws "${OUT}/libv240fix.so" | grep -q 'Java_dev_hoonex_adofai_v240_dynamic_RuntimeEntry_nativeReconcileDynamicRuntime'
strings "${OUT}/libv240fix.so" | grep -q 'nativeProbe=cache-post-bnm-sfb-dynamic-import-uihit-v1'
strings "${OUT}/libv240fix.so" | grep -q 'sfbHookPolicy=dynamic-document-preprocess-before-bind'
strings "${OUT}/libv240fix.so" | grep -q 'uiHitPolicy=pinned-upstream-eventsystem-raycast'
strings "${OUT}/libv240fix.so" | grep -q 'uiHitSourceCommit=74bcc7a0d8c8be1267504e21e28a35e199b5d4eb'
sha256sum "${OUT}/libv240fix.so" | tee "${OUT}/SHA256SUMS.txt"
