#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-fixed-java}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
PLATFORM="${ADOFAI_ANDROID_PLATFORM:-android-35}"
BUILD_TOOLS="${ADOFAI_BUILD_TOOLS:-36.0.0}"
if [[ -z "${SDK}" ]]; then echo 'ANDROID_SDK_ROOT is required' >&2; exit 2; fi
ANDROID_JAR="${SDK}/platforms/${PLATFORM}/android.jar"
D8="${SDK}/build-tools/${BUILD_TOOLS}/d8"
test -f "${ANDROID_JAR}"
test -x "${D8}"
rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex"
mapfile -t SRC < <(find "${ROOT}/android/v240-fixed-runtime/java" -name '*.java' -print | sort)
javac -source 8 -target 8 -bootclasspath "${ANDROID_JAR}" -d "${OUT}/classes" "${SRC[@]}"
jar cf "${OUT}/v240-fixed-runtime.jar" -C "${OUT}/classes" .
"${D8}" --min-api 23 --output "${OUT}/dex" "${OUT}/v240-fixed-runtime.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/v240-fixed-runtime.dex"
test -s "${OUT}/v240-fixed-runtime.dex"
sha256sum "${OUT}/v240-fixed-runtime.dex" | tee "${OUT}/SHA256SUMS.txt"
rm -rf "${OUT}/classes" "${OUT}/dex"
