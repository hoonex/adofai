#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/android/editor-harness/app"
GEN="${APP}/generated/java/com/unity3d/player"
TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"
SAF_SELECTOR="${APP}/src/main/java/com/unity3d/player/FileSelector.java"
PLAYER="${APP}/src/main/java/dev/hoonex/adofai/companion/PlayerActivity.java"
PLAYER_BRIDGE="${APP}/src/main/java/com/unity3d/player/CustomPlayerBridge.java"

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
python3 "${ROOT}/tools/apply_custom_game_mode.py" \
  "${GEN}/MobileEditorShell.java" \
  "${GEN}/MobileEditorShell.java"

# Fail closed if either the generated shell or SAF/player bridge drifts away from
# the standalone custom-game contract.
grep -q 'ADOFAI Custom Editor' "${GEN}/MobileEditorShell.java"
grep -q 'openStandalonePath' "${GEN}/MobileEditorShell.java"
grep -q 'syncSavedPath' "${GEN}/MobileEditorShell.java"
grep -q 'makeAction("Play"' "${GEN}/MobileEditorShell.java"
grep -q 'CustomPlayerBridge.open' "${GEN}/MobileEditorShell.java"
grep -q 'pickerInFlight' "${GEN}/MobileEditorShell.java"
grep -q 'Unexpected trailing content after JSON value' "${GEN}/MobileEditorShell.java"
grep -q 'requestClose' "${GEN}/MobileEditorShell.java"
grep -q 'confirmOpenPath' "${GEN}/MobileEditorShell.java"
grep -q 'confirmSaveAndPreview' "${GEN}/MobileEditorShell.java"
grep -q 'Unsaved changes' "${GEN}/MobileEditorShell.java"
grep -q 'Discard & open' "${GEN}/MobileEditorShell.java"
grep -q 'Save & preview' "${GEN}/MobileEditorShell.java"

test -s "${SAF_SELECTOR}"
grep -q 'Intent.ACTION_OPEN_DOCUMENT' "${SAF_SELECTOR}"
grep -q 'Intent.ACTION_CREATE_DOCUMENT' "${SAF_SELECTOR}"
grep -q 'takePersistableUriPermission' "${SAF_SELECTOR}"
if grep -q 'MANAGE_EXTERNAL_STORAGE' "${SAF_SELECTOR}"; then
  echo 'custom game SAF bridge must not request raw all-files access' >&2
  exit 1
fi

test -s "${PLAYER}"
test -s "${PLAYER_BRIDGE}"
grep -q 'SetSpeed' "${PLAYER}"
grep -q 'Twirl' "${PLAYER}"
grep -q 'Pause' "${PLAYER}"
grep -q 'Hold' "${PLAYER}"
grep -q 'MediaPlayer' "${PLAYER}"
grep -q 'PlayerActivity.EXTRA_CHART_PATH' "${PLAYER_BRIDGE}"

printf 'Prepared standalone ADOFAI Custom editor + gameplay sources at %s\n' "${GEN}"
