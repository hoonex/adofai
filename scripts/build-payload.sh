#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/payload}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
PLATFORM="${ADOFAI_ANDROID_PLATFORM:-android-35}"
BUILD_TOOLS="${ADOFAI_BUILD_TOOLS:-36.0.0}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"

if [[ -z "${SDK}" ]]; then
  echo "ANDROID_SDK_ROOT (or ANDROID_HOME) is required" >&2
  exit 2
fi

ANDROID_JAR="${SDK}/platforms/${PLATFORM}/android.jar"
D8="${SDK}/build-tools/${BUILD_TOOLS}/d8"
NDK_BUILD="${SDK}/ndk/${NDK_VERSION}/ndk-build"

for required in "${ANDROID_JAR}" "${D8}" "${NDK_BUILD}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required Android toolchain component missing: ${required}" >&2
    exit 2
  fi
done

command -v javac >/dev/null
command -v jar >/dev/null
command -v sha256sum >/dev/null
command -v python3 >/dev/null

bash "${ROOT}/scripts/prepare-upstream.sh"
SRC="${ROOT}/.work/hitmargin-mobile-mod"
JAVA_SRC="${SRC}/app/src/main/java/com/unity3d/player"
EDITOR_SHELL_TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"
GENERATED_JAVA_DIR="${OUT}/generated/com/unity3d/player"
EDITOR_SHELL="${GENERATED_JAVA_DIR}/MobileEditorShell.java"
JNI="${SRC}/app/src/main/jni"

if [[ ! -f "${EDITOR_SHELL_TEMPLATE}" ]]; then
  echo "mobile editor shell source missing: ${EDITOR_SHELL_TEMPLATE}" >&2
  exit 2
fi

rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex" "${GENERATED_JAVA_DIR}"

python3 "${ROOT}/tools/apply_mobile_editor_document_safety.py" \
  "${EDITOR_SHELL_TEMPLATE}" \
  "${EDITOR_SHELL}"

javac \
  -source 8 -target 8 \
  -bootclasspath "${ANDROID_JAR}" \
  -d "${OUT}/classes" \
  "${JAVA_SRC}/CustomFileChooser.java" \
  "${JAVA_SRC}/FileSelector.java" \
  "${EDITOR_SHELL}"

jar cf "${OUT}/filepicker.jar" -C "${OUT}/classes" .
"${D8}" \
  --min-api 23 \
  --output "${OUT}/dex" \
  "${OUT}/filepicker.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/classes2.dex"

"${NDK_BUILD}" -C "${JNI}" -j2
LIB="${SRC}/app/src/main/libs/arm64-v8a/libOctober.so"
if [[ ! -s "${LIB}" ]]; then
  echo "native build did not produce ${LIB}" >&2
  exit 3
fi
cp "${LIB}" "${OUT}/libOctober.so"

test -s "${OUT}/classes2.dex"
test -s "${OUT}/libOctober.so"
(
  cd "${OUT}"
  sha256sum classes2.dex libOctober.so > SHA256SUMS.txt
)

rm -rf "${OUT}/classes" "${OUT}/dex" "${OUT}/filepicker.jar" "${OUT}/generated"
printf 'Payload ready: %s\n' "${OUT}"
cat "${OUT}/SHA256SUMS.txt"
