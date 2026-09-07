#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.work/hitmargin-mobile-mod"
REPO="https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level.git"
SHA="74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
PATCH_DIR="${ROOT}/patches/hitmargin"

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

if compgen -G "${PATCH_DIR}/*.patch" >/dev/null; then
  for patch in "${PATCH_DIR}"/*.patch; do
    git -C "${DEST}" apply --check "${patch}"
    git -C "${DEST}" apply "${patch}"
    printf 'Applied patch: %s\n' "$(basename "${patch}")"
  done
fi

printf 'Prepared pinned mobile hook baseline at %s\n' "${DEST}"
printf 'Base commit: %s\n' "${ACTUAL}"
printf 'Working tree now contains only reviewed local patches on top of that identity.\n'
