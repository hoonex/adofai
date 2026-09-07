#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/dist/zygisk-runtime}"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
PLATFORM="${ADOFAI_ANDROID_PLATFORM:-android-35}"
BUILD_TOOLS="${ADOFAI_BUILD_TOOLS:-36.0.0}"
NDK_VERSION="${ADOFAI_NDK_VERSION:-29.0.14206865}"
ZYGISK_SHA="7bb941ac8edfcffd1d23761e401c45ca95409dc1"

if [[ -z "${SDK}" ]]; then
  echo "ANDROID_SDK_ROOT (or ANDROID_HOME) is required" >&2
  exit 2
fi
ANDROID_JAR="${SDK}/platforms/${PLATFORM}/android.jar"
D8="${SDK}/build-tools/${BUILD_TOOLS}/d8"
NDK_BUILD="${SDK}/ndk/${NDK_VERSION}/ndk-build"
for required in "${ANDROID_JAR}" "${D8}" "${NDK_BUILD}"; do
  [[ -e "${required}" ]] || { echo "missing Android toolchain component: ${required}" >&2; exit 2; }
done
for command in git javac jar python3 sha256sum zip; do command -v "${command}" >/dev/null; done

rm -rf "${OUT}"
mkdir -p "${OUT}"

# Build the identity-pinned current-runtime native preview bridge, but remove the
# APK-injection Java bootstrap. Zygisk owns Java/Dex loading in this distribution.
bash "${ROOT}/scripts/prepare-upstream.sh"
UPSTREAM="${ROOT}/.work/hitmargin-mobile-mod"
python3 "${ROOT}/tools/apply_zygisk_native_mode.py" "${UPSTREAM}"

GEN="${OUT}/generated/com/unity3d/player"
CLASSES="${OUT}/java-classes"
DEXDIR="${OUT}/dex"
mkdir -p "${GEN}" "${CLASSES}" "${DEXDIR}"
SHELL_TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"
SHELL="${GEN}/MobileEditorShell.java"

python3 "${ROOT}/tools/apply_mobile_editor_document_safety.py" "${SHELL_TEMPLATE}" "${SHELL}"
python3 "${ROOT}/tools/apply_mobile_editor_json_strictness.py" "${SHELL}" "${SHELL}"
python3 "${ROOT}/tools/apply_mobile_editor_picker_serialization.py" "${SHELL}" "${SHELL}"
python3 "${ROOT}/tools/apply_mobile_editor_dirty_close_guard.py" "${SHELL}" "${SHELL}"
python3 "${ROOT}/tools/apply_zygisk_editor_mode.py" "${SHELL}" "${SHELL}"

javac \
  -source 8 -target 8 \
  -bootclasspath "${ANDROID_JAR}" \
  -d "${CLASSES}" \
  "${ROOT}/android/zygisk-runtime/java/com/unity3d/player/FileSelector.java" \
  "${ROOT}/android/zygisk-runtime/java/com/unity3d/player/ZygiskEditorBootstrap.java" \
  "${SHELL}"
jar cf "${OUT}/editor.jar" -C "${CLASSES}" .
"${D8}" --min-api 26 --output "${DEXDIR}" "${OUT}/editor.jar"
mv "${DEXDIR}/classes.dex" "${OUT}/editor.dex"

"${NDK_BUILD}" -C "${UPSTREAM}/app/src/main/jni" -j2
OCTOBER="${UPSTREAM}/app/src/main/libs/arm64-v8a/libOctober.so"
[[ -s "${OCTOBER}" ]] || { echo "native preview payload missing: ${OCTOBER}" >&2; exit 3; }
cp "${OCTOBER}" "${OUT}/libOctober.so"

# Pin the published Zygisk API v4 (Magisk 26+) instead of building against an
# unversioned moving header. Newer Magisk versions retain backwards compatibility.
ZYGISK_SRC="${ROOT}/.work/zygisk-module-sample"
if [[ ! -d "${ZYGISK_SRC}/.git" ]]; then
  rm -rf "${ZYGISK_SRC}"
  git clone --filter=blob:none https://github.com/topjohnwu/zygisk-module-sample.git "${ZYGISK_SRC}"
fi
git -C "${ZYGISK_SRC}" fetch --prune origin
git -C "${ZYGISK_SRC}" checkout --detach "${ZYGISK_SHA}"
git -C "${ZYGISK_SRC}" reset --hard "${ZYGISK_SHA}"
[[ "$(git -C "${ZYGISK_SRC}" rev-parse HEAD)" == "${ZYGISK_SHA}" ]] || {
  echo "Zygisk API identity mismatch" >&2; exit 3;
}

MODULE_PROJECT="${ROOT}/android/zygisk-runtime/module"
rm -rf "${MODULE_PROJECT}/libs" "${MODULE_PROJECT}/obj"
"${NDK_BUILD}" \
  NDK_PROJECT_PATH="${MODULE_PROJECT}" \
  APP_BUILD_SCRIPT="${MODULE_PROJECT}/jni/Android.mk" \
  NDK_APPLICATION_MK="${MODULE_PROJECT}/jni/Application.mk" \
  ZYGISK_API_DIR="${ZYGISK_SRC}/module/jni" \
  -j2
ZYGISK_SO="${MODULE_PROJECT}/libs/arm64-v8a/libadofai_editor_zygisk.so"
[[ -s "${ZYGISK_SO}" ]] || { echo "Zygisk arm64 module was not produced" >&2; exit 3; }

STAGE="${OUT}/module-stage"
mkdir -p "${STAGE}/zygisk" "${STAGE}/payload"
cp "${MODULE_PROJECT}/module.prop" "${STAGE}/module.prop"
cp "${ZYGISK_SO}" "${STAGE}/zygisk/arm64-v8a.so"
cp "${OUT}/editor.dex" "${STAGE}/payload/editor.dex"
cp "${OUT}/libOctober.so" "${STAGE}/payload/libOctober.so"
cat > "${STAGE}/README.txt" <<'EOF'
ADOFAI 3.3.1 Mobile Editor Runtime

Requires Magisk 26+ with Zygisk enabled and the official Play-installed
com.fizzd.connectedworlds version 3.3.1 (versionCode 300382).

This module does not replace, patch, resign, or bundle the commercial game APK.
Install this ZIP from Magisk's Modules screen and reboot. Disable/remove the module
to return to the untouched official runtime.
EOF

ZIP="${OUT}/ADOFAI-3.3.1-Zygisk-Editor.zip"
(
  cd "${STAGE}"
  zip -qr "${ZIP}" .
)
sha256sum "${ZIP}" "${OUT}/editor.dex" "${OUT}/libOctober.so" "${ZYGISK_SO}" \
  > "${OUT}/SHA256SUMS.txt"

rm -rf "${CLASSES}" "${DEXDIR}" "${OUT}/editor.jar" "${OUT}/generated" "${STAGE}"
printf 'Zygisk runtime module ready: %s\n' "${ZIP}"
cat "${OUT}/SHA256SUMS.txt"
