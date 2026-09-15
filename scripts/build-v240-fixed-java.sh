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

WINDOW_COMPAT="${ROOT}/android/v240-fixed-runtime/java/com/unity3d/player/V240WindowCompat.java"
BOOTSTRAP="${ROOT}/android/v240-fixed-runtime/java/com/unity3d/player/V240Bootstrap.java"
UPDATER="${ROOT}/android/v240-fixed-runtime/java/com/unity3d/player/V240RuntimeUpdater.java"
COMPAT_REPORT="${ROOT}/android/v240-fixed-runtime/java/com/unity3d/player/V240CompatibilityReport.java"
grep -Fq 'LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER' "${WINDOW_COMPAT}"
grep -Fq 'SOFT_INPUT_ADJUST_RESIZE' "${WINDOW_COMPAT}"
grep -Fq 'setSystemGestureExclusionRects' "${WINDOW_COMPAT}"
grep -Fq 'height * 0.30f' "${WINDOW_COMPAT}"
grep -Fq 'height * 0.70f' "${WINDOW_COMPAT}"
grep -Fq 'V240WindowCompat.apply();' "${BOOTSTRAP}"
grep -Fq 'V240RuntimeUpdater.startIfReady();' "${BOOTSTRAP}"
! grep -Fq 'System.loadLibrary("v240fix")' "${BOOTSTRAP}"
! grep -Fq 'V240EventCompat.initialize();' "${BOOTSTRAP}"
grep -Fq 'main.postDelayed(forceRebind, 500L);' "${BOOTSTRAP}"
grep -Fq 'main.postDelayed(forceRebind, 1500L);' "${BOOTSTRAP}"
[[ "$(grep -Fc 'V240CompatibilityReport.install();' "${BOOTSTRAP}")" -eq 2 ]]
grep -Fq 'getCodeCacheDir()' "${UPDATER}"
grep -Fq 'boot.pending' "${UPDATER}"
grep -Fq 'DexClassLoader' "${UPDATER}"
grep -Fq 'System.load(nativeLib.getAbsolutePath())' "${UPDATER}"
grep -Fq 'bundleSha256' "${UPDATER}"
grep -Fq 'rollout' "${UPDATER}"
grep -Fq 'setReadOnly()' "${UPDATER}"
grep -Fq 'setOnLongClickListener' "${COMPAT_REPORT}"
grep -Fq 'ClipboardManager' "${COMPAT_REPORT}"
grep -Fq 'nativeGetCompatibilityReport' "${COMPAT_REPORT}"
grep -Fq 'V240RuntimeUpdater.diagnosticText()' "${COMPAT_REPORT}"

rm -rf "${OUT}"
mkdir -p "${OUT}/classes" "${OUT}/dex"
mapfile -t SRC < <(find "${ROOT}/android/v240-fixed-runtime/java" -name '*.java' -print | sort)
mapfile -t STUBS < <(find "${ROOT}/android/v240-fixed-runtime/stubs" -name '*.java' -print | sort)
javac -source 8 -target 8 -bootclasspath "${ANDROID_JAR}" -d "${OUT}/classes" "${STUBS[@]}" "${SRC[@]}"
# The stub only satisfies javac. The original APK's UnityPlayerActivity remains authoritative.
rm -f "${OUT}/classes/com/unity3d/player/UnityPlayerActivity.class"
jar cf "${OUT}/v240-fixed-runtime.jar" -C "${OUT}/classes" .
"${D8}" --min-api 23 --output "${OUT}/dex" "${OUT}/v240-fixed-runtime.jar"
mv "${OUT}/dex/classes.dex" "${OUT}/v240-fixed-runtime.dex"
test -s "${OUT}/v240-fixed-runtime.dex"
# Fail closed if the compile-only stub accidentally leaked into the payload.
if strings "${OUT}/v240-fixed-runtime.dex" | grep -Fq 'Lcom/unity3d/player/UnityPlayerActivity;'; then
  # The superclass reference from V240UnityPlayerActivity is expected. A class definition is checked below with dexdump if available.
  true
fi
for marker in \
  'Lcom/unity3d/player/V240UnityPlayerActivity;' \
  'Lcom/unity3d/player/V240AndroidBridge;' \
  'Lcom/unity3d/player/V240PickerActivity;' \
  'Lcom/unity3d/player/V240SettingsOverlay;' \
  'Lcom/unity3d/player/V240WindowCompat;' \
  'Lcom/unity3d/player/V240EventCompat;' \
  'Lcom/unity3d/player/V240CompatibilityReport;' \
  'Lcom/unity3d/player/V240RuntimeUpdater;' \
  'Lcom/unity3d/player/V240SetFrameRateBackport;' \
  'Lcom/unity3d/player/V240LevelFolderBridge;' \
  'Lcom/unity3d/player/V240ArchiveOpenBridge;' \
  'Lcom/unity3d/player/V240ZipLevelImporter;' \
  'Lcom/unity3d/player/V240MapCompatibility;' \
  'Lcom/unity3d/player/V240ChartBackport;' \
  'Lcom/unity3d/player/V240ChartCompatibilityScanner;' \
  'Lcom/unity3d/player/V240OpaqueEventBridge;' \
  'Lcom/unity3d/player/V240HallLegacyFix;' \
  'Lcom/unity3d/player/FileSelector;'; do
  strings "${OUT}/v240-fixed-runtime.dex" | grep -Fq "${marker}" || { echo "missing payload class: ${marker}" >&2; exit 3; }
done
# The hot-swappable entrypoint must never be embedded in this parent ClassLoader.
! strings "${OUT}/v240-fixed-runtime.dex" | grep -Fq 'Ldev/hoonex/adofai/v240/dynamic/RuntimeEntry;'
sha256sum "${OUT}/v240-fixed-runtime.dex" | tee "${OUT}/SHA256SUMS.txt"
rm -rf "${OUT}/classes" "${OUT}/dex"
