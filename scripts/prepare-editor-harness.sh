#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/android/editor-harness/app"
GEN="${APP}/generated/java/com/unity3d/player"
TEMPLATE="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java"

rm -rf "${APP}/generated"
mkdir -p "${GEN}"

# Reuse the exact identity-pinned and hardened Java picker sources that the real
# injected payload uses. This includes modern storage handling and Back/cancel
# completion, but does not copy any proprietary ADOFAI game code or assets.
bash "${ROOT}/scripts/prepare-upstream.sh"
UPSTREAM_JAVA="${ROOT}/.work/hitmargin-mobile-mod/app/src/main/java/com/unity3d/player"

# Generate the same reviewed shell source as the canonical injected payload.
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

cp "${UPSTREAM_JAVA}/FileSelector.java" "${GEN}/FileSelector.java"
cp "${UPSTREAM_JAVA}/CustomFileChooser.java" "${GEN}/CustomFileChooser.java"

# Fail closed if the generated standalone host stops matching the safety surface
# proven for the real payload.
grep -q 'pickerInFlight' "${GEN}/MobileEditorShell.java"
grep -q 'Unexpected trailing content after JSON value' "${GEN}/MobileEditorShell.java"
grep -q "This chart's settings field is not an object" "${GEN}/MobileEditorShell.java"
grep -q 'requestClose' "${GEN}/MobileEditorShell.java"
grep -q 'Unsaved changes' "${GEN}/MobileEditorShell.java"
grep -q 'Discard & close' "${GEN}/MobileEditorShell.java"
grep -q 'confirmOpenPath' "${GEN}/MobileEditorShell.java"
grep -q 'Discard & open' "${GEN}/MobileEditorShell.java"
grep -q 'KEYCODE_BACK' "${GEN}/MobileEditorShell.java"
grep -q 'setOnCancelListener' "${GEN}/CustomFileChooser.java"
grep -q 'Environment.isExternalStorageManager' "${GEN}/FileSelector.java"

printf 'Prepared standalone editor harness sources at %s\n' "${GEN}"
