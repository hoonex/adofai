#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.work/hitmargin-v240-bugfix"
REPO="https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git"
SHA="74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"

mkdir -p "${ROOT}/.work"
if [[ ! -d "${DEST}/.git" ]]; then
  rm -rf "${DEST}"
  git clone --filter=blob:none "${REPO}" "${DEST}"
fi

git -C "${DEST}" fetch --prune origin
git -C "${DEST}" checkout --detach "${SHA}"
git -C "${DEST}" reset --hard "${SHA}"
git -C "${DEST}" clean -fdx

ACTUAL="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${SHA}" ]]; then
  echo "upstream identity mismatch: expected ${SHA}, got ${ACTUAL}" >&2
  exit 2
fi

# Deliberately keep this patchset limited to the old custom runtime fixes.
# Do not apply the later 3.3 runtime profile, UnityFileDialog backend,
# Companion editor shell, or official-game handoff code here.
python3 "${ROOT}/tools/apply_hitmargin_editor_mode.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_file_dialogs.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_storage_guard.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_picker_cancel_guard.py" "${DEST}"

git -C "${DEST}" diff --check

grep -q 'enableEditorDesktopMode' "${DEST}/app/src/main/jni/Config.h"
grep -q 'NeedsDesktopEditorMode' "${DEST}/app/src/main/jni/Hooks.cpp"
grep -q 'g_method_saveAs' "${DEST}/app/src/main/jni/FilePicker.cpp"
grep -q 'Hooked_OpenFolderPanel' "${DEST}/app/src/main/jni/FilePicker.cpp"
grep -q 'ensureStorageAccess' "${DEST}/app/src/main/java/com/unity3d/player/FileSelector.java"
grep -q 'setOnCancelListener' "${DEST}/app/src/main/java/com/unity3d/player/CustomFileChooser.java"

# Guard against accidentally pulling the abandoned/newer product paths into v2.4.
if grep -R -q 'OfficialGameBridge\|MobileEditorShell\|enableModernSafeProfile' \
  "${DEST}/app/src/main/java" "${DEST}/app/src/main/jni"; then
  echo "unexpected modern/Companion code in v2.4 bugfix source" >&2
  exit 3
fi

printf 'Prepared v2.4 custom bugfix source at %s\n' "${DEST}"
printf 'Pinned upstream: %s\n' "${ACTUAL}"
printf 'Patchset: editor desktop mode + Open/Save/Folder + Android storage/Activity + Back cancellation\n'
