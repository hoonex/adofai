#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-cache-native}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"
API="${ADOFAI_ANDROID_MIN_API:-23}"

if [[ -z "${SDK}" ]]; then echo 'ANDROID_SDK_ROOT is required' >&2; exit 2; fi
NDK="${SDK}/ndk/${NDK_VERSION}"
HOST_TAG="linux-x86_64"
TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/${HOST_TAG}"
CC="${TOOLCHAIN}/bin/aarch64-linux-android${API}-clang"
LLVM_NM="${TOOLCHAIN}/bin/llvm-nm"
LLVM_READELF="${TOOLCHAIN}/bin/llvm-readelf"
SRC="${ROOT}/android/v240-dynamic-runtime/native/V240CacheLoader.c"

test -x "${CC}"
test -x "${LLVM_NM}"
test -x "${LLVM_READELF}"
test -f "${SRC}"

python3 - "${SRC}" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text(encoding='utf-8')
required = ['JNI_OnLoad', 'GetEnv', 'JNI_VERSION_1_6']
for marker in required:
    assert marker in s, marker
for forbidden in (
    'universe.h', 'BNM', 'Loading::', 'BasicHook', 'Dobby', 'dlopen',
    'pthread_create', 'FindClass', 'CallStatic', 'CallObject', 'NewGlobalRef',
):
    assert forbidden not in s, forbidden
PY

rm -rf "${OUT}"
mkdir -p "${OUT}"
"${CC}" \
  -shared -fPIC -Oz -fvisibility=hidden \
  -Wl,-z,relro,-z,now,--no-undefined \
  -Wl,--build-id=sha1 \
  "${SRC}" \
  -o "${OUT}/libv240fix.so"

test -s "${OUT}/libv240fix.so"
"${LLVM_READELF}" -h "${OUT}/libv240fix.so" | grep -Fq 'AArch64'
"${LLVM_NM}" -D --defined-only "${OUT}/libv240fix.so" | grep -Eq '[[:space:]]JNI_OnLoad$'
# A cache candidate must not accidentally grow any old Java/native activation ABI.
! "${LLVM_NM}" -D --defined-only "${OUT}/libv240fix.so" | grep -Eq 'Java_com_unity3d_player_|BasicHook|Dobby|BNM'
sha256sum "${OUT}/libv240fix.so" | tee "${OUT}/SHA256SUMS.txt"
