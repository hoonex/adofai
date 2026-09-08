#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/v240-bugfix-payload}"
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

bash "${ROOT}/scripts/prepare-v240-bugfix-upstream.sh"
SRC="${ROOT}/.work/hitmargin-v240-bugfix"
JAVA_SRC="${SRC}/app/src/main/java/com/unity3d/player"
JNI="${SRC}/app/src/main/jni"

rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex"

javac \
  -source 8 -target 8 \
  -bootclasspath "${ANDROID_JAR}" \
  -d "${OUT}/classes" \
  "${JAVA_SRC}/CustomFileChooser.java" \
  "${JAVA_SRC}/FileSelector.java"

jar cf "${OUT}/filepicker.jar" -C "${OUT}/classes" .
"${D8}" --min-api 23 --output "${OUT}/dex" "${OUT}/filepicker.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/classes2.dex"

"${NDK_BUILD}" -C "${JNI}" -j2
LIB="${SRC}/app/src/main/libs/arm64-v8a/libOctober.so"
if [[ ! -s "${LIB}" ]]; then
  LIB="${SRC}/app/src/main/obj/local/arm64-v8a/libOctober.so"
fi
if [[ ! -s "${LIB}" ]]; then
  echo "native build did not produce libOctober.so" >&2
  exit 3
fi
cp "${LIB}" "${OUT}/libOctober.so"

cat > "${OUT}/PATCHSET.txt" <<'EOF'
ADOFAI v2.4 Custom bugfix payload

Pinned hook source:
  HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level
  74bcc7a0d8c8be1267504e21e28a35e199b5d4eb

Applied fixes only:
  1fcb41684bc9f50679ef1937157f4912a1b94495  editor desktop-mode hook
  01df912cbc65c75d03b92880154388cade87b600  Open/Save/Folder file-dialog bridge
  5d33b38572c1545ff7907c070f2a0ed267ad4725  Android storage + Activity acquisition
  3b2b0b109f6b954ec168354b7b5f040602b21fbb  Back/cancel completion guard

Explicitly excluded:
  ADOFAI 3.3 safe-runtime profile
  3.3 UnityFileDialog backend
  modern native editor shell
  Companion Editor / official handoff / Custom Player
EOF

(
  cd "${OUT}"
  sha256sum classes2.dex libOctober.so > SHA256SUMS.txt
)

test -s "${OUT}/classes2.dex"
test -s "${OUT}/libOctober.so"
rm -rf "${OUT}/classes" "${OUT}/dex" "${OUT}/filepicker.jar"
printf 'v2.4 bugfix payload ready: %s\n' "${OUT}"
cat "${OUT}/SHA256SUMS.txt"
