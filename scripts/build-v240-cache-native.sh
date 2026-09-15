#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-cache-native}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"
API="${ADOFAI_ANDROID_MIN_API:-23}"

if [[ -z "${SDK}" ]]; then echo 'ANDROID_SDK_ROOT is required' >&2; exit 2; fi
NDK="${SDK}/ndk/${NDK_VERSION}"
NDK_BUILD="${NDK}/ndk-build"
TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/linux-x86_64"
LLVM_NM="${TOOLCHAIN}/bin/llvm-nm"
LLVM_READELF="${TOOLCHAIN}/bin/llvm-readelf"
SRC="${ROOT}/android/v240-dynamic-runtime/native/V240CacheLoader.c"
WORK="${ROOT}/.work/v240-cache-native"

test -x "${NDK_BUILD}"
test -x "${LLVM_NM}"
test -x "${LLVM_READELF}"
test -f "${SRC}"

python3 - "${SRC}" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text(encoding='utf-8')
for marker in ('JNI_OnLoad', 'GetEnv', 'JNI_VERSION_1_6'):
    assert marker in s, marker
for forbidden in (
    'universe.h', 'BNM', 'Loading::', 'BasicHook', 'Dobby', 'dlopen',
    'pthread_create', 'FindClass', 'CallStatic', 'CallObject', 'NewGlobalRef',
):
    assert forbidden not in s, forbidden
PY

rm -rf "${WORK}" "${OUT}"
mkdir -p "${WORK}/jni" "${OUT}"
cp "${SRC}" "${WORK}/jni/V240CacheLoader.c"
cat > "${WORK}/jni/Android.mk" <<'EOF'
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := v240fix
LOCAL_SRC_FILES := V240CacheLoader.c
LOCAL_CFLAGS := -Oz -fvisibility=hidden -Wall -Wextra -Werror
LOCAL_LDFLAGS := -Wl,-z,relro,-z,now,--no-undefined -Wl,--build-id=sha1
include $(BUILD_SHARED_LIBRARY)
EOF
cat > "${WORK}/jni/Application.mk" <<EOF
APP_ABI := arm64-v8a
APP_PLATFORM := android-${API}
APP_OPTIM := release
EOF

"${NDK_BUILD}" \
  NDK_PROJECT_PATH="${WORK}" \
  APP_BUILD_SCRIPT="${WORK}/jni/Android.mk" \
  NDK_APPLICATION_MK="${WORK}/jni/Application.mk" \
  V=1

cp "${WORK}/libs/arm64-v8a/libv240fix.so" "${OUT}/libv240fix.so"
test -s "${OUT}/libv240fix.so"

# Capture complete tool output before matching. With `set -o pipefail`, using
# `llvm-readelf|grep -q` or `llvm-nm|grep -q` can make LLVM observe SIGPIPE
# after grep exits on the first match and incorrectly fail a valid build.
ELF_HEADER="$("${LLVM_READELF}" -h "${OUT}/libv240fix.so")"
DYNAMIC_SYMBOLS="$("${LLVM_NM}" -D --defined-only "${OUT}/libv240fix.so")"

if ! grep -F 'AArch64' <<<"${ELF_HEADER}" >/dev/null; then
  echo 'cache loader verification failed: output is not AArch64 ELF' >&2
  exit 1
fi
if ! grep -E '[[:space:]]JNI_OnLoad$' <<<"${DYNAMIC_SYMBOLS}" >/dev/null; then
  echo 'cache loader verification failed: JNI_OnLoad is not exported' >&2
  exit 1
fi
# A cache candidate must not accidentally grow any old Java/native activation ABI.
if grep -E 'Java_com_unity3d_player_|BasicHook|Dobby|BNM' <<<"${DYNAMIC_SYMBOLS}" >/dev/null; then
  echo 'cache loader verification failed: unexpected activation ABI exported' >&2
  exit 1
fi

sha256sum "${OUT}/libv240fix.so" | tee "${OUT}/SHA256SUMS.txt"
