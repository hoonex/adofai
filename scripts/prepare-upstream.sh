#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.work/hitmargin-mobile-mod"
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

python3 "${ROOT}/tools/apply_hitmargin_editor_mode.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_file_dialogs.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_unity_file_dialogs.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_storage_guard.py" "${DEST}"
python3 "${ROOT}/tools/apply_hitmargin_modern_safe_profile.py" "${DEST}"
python3 "${ROOT}/tools/apply_mobile_editor_shell.py" "${DEST}" --repo-root "${ROOT}"
git -C "${DEST}" diff --check

printf 'Prepared pinned mobile hook baseline at %s\n' "${DEST}"
printf 'Base commit: %s\n' "${ACTUAL}"
printf 'Working tree contains only identity-checked local transforms on top of that source.\n'
