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

ACTUAL="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${SHA}" ]]; then
  echo "upstream identity mismatch: expected ${SHA}, got ${ACTUAL}" >&2
  exit 2
fi

printf 'Prepared pinned mobile hook baseline at %s\n' "${DEST}"
printf 'Commit: %s\n' "${ACTUAL}"
printf 'Next: inspect/patch only against this exact source identity.\n'
