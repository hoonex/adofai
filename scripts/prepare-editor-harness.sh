#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/android/editor-harness/app"
GEN="${APP}/generated/java/com/unity3d/player"
TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"
SAF_SELECTOR="${APP}/src/main/java/com/unity3d/player/FileSelector.java"

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

# Fail closed if either the generated shell or SAF bridge drifts away from the
# independent companion-editor contract.
grep -q 'ADOFAI Companion Editor' "${GEN}/MobileEditorShell.java"
grep -q 'openStandalonePath' "${GEN}/MobileEditorShell.java"
grep -q 'syncSavedPath' "${GEN}/MobileEditorShell.java"
grep -q 'ADOFAI / 공유' "${GEN}/MobileEditorShell.java"
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
grep -q 'openInAdofaiOrShare' "${SAF_SELECTOR}"
if grep -q 'MANAGE_EXTERNAL_STORAGE' "${SAF_SELECTOR}"; then
  echo 'companion SAF bridge must not request raw all-files access' >&2
  exit 1
fi

printf 'Prepared standalone ADOFAI companion editor sources at %s\n' "${GEN}"
