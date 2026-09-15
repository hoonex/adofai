#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-dynamic-runtime}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
PLATFORM="${ADOFAI_ANDROID_PLATFORM:-android-35}"
BUILD_TOOLS="${ADOFAI_BUILD_TOOLS:-36.0.0}"
if [[ -z "${SDK}" ]]; then echo 'ANDROID_SDK_ROOT is required' >&2; exit 2; fi
ANDROID_JAR="${SDK}/platforms/${PLATFORM}/android.jar"
D8="${SDK}/build-tools/${BUILD_TOOLS}/d8"
test -f "${ANDROID_JAR}"
test -x "${D8}"

SRC_ROOT="${ROOT}/android/v240-dynamic-runtime/java"
ENTRY="${SRC_ROOT}/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
test -f "${ENTRY}"
grep -Fq 'dev.hoonex.adofai.v240.dynamic' "${ENTRY}"
grep -Fq 'public static void install(Context context)' "${ENTRY}"

rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex"
mapfile -t SRC < <(find "${SRC_ROOT}" -name '*.java' -print | sort)
javac -source 8 -target 8 -bootclasspath "${ANDROID_JAR}" -d "${OUT}/classes" "${SRC[@]}"
jar cf "${OUT}/v240-dynamic-runtime.jar" -C "${OUT}/classes" .
"${D8}" --min-api 23 --output "${OUT}/dex" "${OUT}/v240-dynamic-runtime.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/runtime.dex"
test -s "${OUT}/runtime.dex"
strings "${OUT}/runtime.dex" | grep -Fq 'Ldev/hoonex/adofai/v240/dynamic/RuntimeEntry;'
sha256sum "${OUT}/runtime.dex" | tee "${OUT}/SHA256SUMS.txt"
rm -rf "${OUT}/classes" "${OUT}/dex"
