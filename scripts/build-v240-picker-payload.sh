#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-picker-payload}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
PLATFORM="${ADOFAI_ANDROID_PLATFORM:-android-35}"
BUILD_TOOLS="${ADOFAI_BUILD_TOOLS:-36.0.0}"

if [[ -z "${SDK}" ]]; then
  echo "ANDROID_SDK_ROOT (or ANDROID_HOME) is required" >&2
  exit 2
fi
ANDROID_JAR="${SDK}/platforms/${PLATFORM}/android.jar"
D8="${SDK}/build-tools/${BUILD_TOOLS}/d8"
for required in "${ANDROID_JAR}" "${D8}"; do
  [[ -e "${required}" ]] || { echo "missing: ${required}" >&2; exit 2; }
done

bash "${ROOT}/scripts/prepare-v240-bugfix-upstream.sh"
SRC="${ROOT}/.work/hitmargin-v240-bugfix/app/src/main/java/com/unity3d/player"
rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex"

javac -source 8 -target 8 -bootclasspath "${ANDROID_JAR}" \
  -d "${OUT}/classes" \
  "${SRC}/CustomFileChooser.java" \
  "${SRC}/FileSelector.java"
jar cf "${OUT}/picker.jar" -C "${OUT}/classes" .
"${D8}" --min-api 23 --output "${OUT}/dex" "${OUT}/picker.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/classes2.dex"

test -s "${OUT}/classes2.dex"
rm -rf "${OUT}/classes" "${OUT}/dex" "${OUT}/picker.jar"
printf 'v2.4 optional picker payload ready: %s\n' "${OUT}/classes2.dex"
