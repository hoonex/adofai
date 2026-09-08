#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/android/editor-harness/app"
GEN="${APP}/generated/java/com/unity3d/player"
TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"
SAF_SELECTOR="${APP}/src/main/java/com/unity3d/player/FileSelector.java"
BUNDLE_WORKSPACE="${APP}/src/main/java/com/unity3d/player/BundleWorkspace.java"
LOOPBACK_SERVER="${APP}/src/main/java/com/unity3d/player/LoopbackZipServer.java"
OFFICIAL_BRIDGE="${APP}/src/main/java/com/unity3d/player/OfficialGameBridge.java"
OFFICIAL_PROVIDER="${APP}/src/main/java/dev/hoonex/adofai/companion/OfficialChartProvider.java"
MAIN_ACTIVITY="${APP}/src/main/java/dev/hoonex/adofai/editorharness/MainActivity.java"

rm -rf "${APP}/generated"
mkdir -p "${GEN}"

python3 "${ROOT}/tools/apply_mobile_editor_document_safety.py" \
  "${TEMPLATE}" \
  "${GEN}/MobileEditorShell.java"
python3 "${ROOT}/tools/apply_mobile_editor_json_strictness.py" \
  "${GEN}/MobileEditorShell.java" \
  "${GEN}/MobileEditorShell.java"
python3 "${ROOT}/tools/apply_mobile_editor_picker_serialization.py" \
  "${GEN}/MobileEditorShell.java" \
  "${GEN}/MobileEditorShell.java"
python3 "${ROOT}/tools/apply_mobile_editor_dirty_close_guard.py" \
  "${GEN}/MobileEditorShell.java" \
  "${GEN}/MobileEditorShell.java"
python3 "${ROOT}/tools/apply_companion_editor_mode.py" \
  "${GEN}/MobileEditorShell.java" \
  "${GEN}/MobileEditorShell.java"

# Canonical product contract: standalone Companion Editor + unmodified official
# Play ADOFAI handoff. ZIP bundles are first-class because charts commonly
# reference sibling audio/images by relative path.
grep -q 'ADOFAI Companion Editor' "${GEN}/MobileEditorShell.java"
grep -q 'openStandalonePath' "${GEN}/MobileEditorShell.java"
grep -q 'showOfficialHandoffDiagnostic' "${GEN}/MobileEditorShell.java"
grep -q 'syncSavedPath' "${GEN}/MobileEditorShell.java"
grep -q 'makeAction("ZIP URL"' "${GEN}/MobileEditorShell.java"
grep -q 'FileSelector.selectUrlBundle' "${GEN}/MobileEditorShell.java"
grep -q 'makeAction("공식 ADOFAI"' "${GEN}/MobileEditorShell.java"
grep -q 'makeAction("원본 HTTPS"' "${GEN}/MobileEditorShell.java"
grep -q 'OfficialGameBridge.open(currentPath)' "${GEN}/MobileEditorShell.java"
grep -q 'OfficialGameBridge.openOriginalHttps(currentPath)' "${GEN}/MobileEditorShell.java"
grep -q 'pickerInFlight' "${GEN}/MobileEditorShell.java"
grep -q 'Unexpected trailing content after JSON value' "${GEN}/MobileEditorShell.java"
grep -q 'requestClose' "${GEN}/MobileEditorShell.java"
grep -q 'confirmOpenPath' "${GEN}/MobileEditorShell.java"
grep -q 'confirmSaveAndPreview' "${GEN}/MobileEditorShell.java"
grep -q 'Unsaved changes' "${GEN}/MobileEditorShell.java"
grep -q 'Discard & open' "${GEN}/MobileEditorShell.java"
grep -q 'Save & preview' "${GEN}/MobileEditorShell.java"
if grep -q 'CustomPlayerBridge' "${GEN}/MobileEditorShell.java"; then
  echo 'canonical companion shell must not target the clean-room CustomPlayer' >&2
  exit 1
fi

test -s "${SAF_SELECTOR}"
grep -q 'Intent.ACTION_OPEN_DOCUMENT' "${SAF_SELECTOR}"
grep -q 'Intent.ACTION_CREATE_DOCUMENT' "${SAF_SELECTOR}"
grep -q 'takePersistableUriPermission' "${SAF_SELECTOR}"
grep -q 'BundleWorkspace.importZip' "${SAF_SELECTOR}"
grep -q 'selectUrlBundle' "${SAF_SELECTOR}"
if grep -q 'MANAGE_EXTERNAL_STORAGE' "${SAF_SELECTOR}"; then
  echo 'companion SAF bridge must not request raw all-files access' >&2
  exit 1
fi

test -s "${BUNDLE_WORKSPACE}"
grep -q 'ZipInputStream' "${BUNDLE_WORKSPACE}"
grep -q 'main.adofai' "${BUNDLE_WORKSPACE}"
grep -q 'ZIP path traversal rejected' "${BUNDLE_WORKSPACE}"
grep -q 'MAX_EXTRACTED_BYTES' "${BUNDLE_WORKSPACE}"
grep -q 'SOURCE_URL_BY_CHART' "${BUNDLE_WORKSPACE}"
grep -q 'sourceHttpsUrlForChart' "${BUNDLE_WORKSPACE}"

test -s "${LOOPBACK_SERVER}"
grep -q '127.0.0.1' "${LOOPBACK_SERVER}"
grep -q 'application/zip' "${LOOPBACK_SERVER}"
grep -q 'ProbeState' "${LOOPBACK_SERVER}"
grep -q 'diagnosticFor' "${LOOPBACK_SERVER}"
grep -q 'noteRequest' "${LOOPBACK_SERVER}"
grep -q 'noteBytes' "${LOOPBACK_SERVER}"

test -s "${OFFICIAL_BRIDGE}"
grep -q 'com.fizzd.connectedworlds' "${OFFICIAL_BRIDGE}"
grep -q 'com.unity3d.player.UnityPlayerActivity' "${OFFICIAL_BRIDGE}"
grep -q 'EXPECTED_VERSION_CODE = 300382L' "${OFFICIAL_BRIDGE}"
grep -q 'FLAG_GRANT_READ_URI_PERMISSION' "${OFFICIAL_BRIDGE}"
grep -q 'BundleWorkspace.packageBundle' "${OFFICIAL_BRIDGE}"
grep -q 'BundleWorkspace.sourceHttpsUrlForChart' "${OFFICIAL_BRIDGE}"
grep -q 'LoopbackZipServer.publish' "${OFFICIAL_BRIDGE}"
grep -q 'LoopbackZipServer.diagnosticFor' "${OFFICIAL_BRIDGE}"
grep -q 'consumeReturnDiagnostic' "${OFFICIAL_BRIDGE}"
grep -q 'openOriginalHttps' "${OFFICIAL_BRIDGE}"
grep -q 'buildUrlIntent' "${OFFICIAL_BRIDGE}"
grep -q 'application/zip' "${OFFICIAL_BRIDGE}"
grep -q 'OfficialChartProvider.publish' "${OFFICIAL_BRIDGE}"

test -s "${MAIN_ACTIVITY}"
grep -q 'RETURN_PROBE_DELAY_MS' "${MAIN_ACTIVITY}"
grep -q 'OfficialGameBridge.consumeReturnDiagnostic' "${MAIN_ACTIVITY}"
grep -q 'MobileEditorShell.showOfficialHandoffDiagnostic' "${MAIN_ACTIVITY}"

test -s "${OFFICIAL_PROVIDER}"
grep -q 'Read-only provider' "${OFFICIAL_PROVIDER}"
grep -q 'MODE_READ_ONLY' "${OFFICIAL_PROVIDER}"

printf 'Prepared ADOFAI Companion Editor + ZIP URL official-game handoff sources at %s\n' "${GEN}"
