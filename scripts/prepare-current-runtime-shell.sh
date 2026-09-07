#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.work/adofai-331-runtime-shell"
REPO="https://github.com/Harrot114514/ADOFAI-MobileLevelLoder.git"
SHA="401b69f26b2f607181d00b07273bac6bb0524638"

mkdir -p "${ROOT}/.work"

if [[ ! -d "${DEST}/.git" ]]; then
  rm -rf "${DEST}"
  git clone --filter=blob:none --no-checkout "${REPO}" "${DEST}"
  git -C "${DEST}" sparse-checkout init --cone
fi

# Consume only the MIT-licensed loader source/documentation needed for the editor
# shell. Do not checkout gameassest/, gamelibs/, levelexample/ or prebuilt .so files.
git -C "${DEST}" sparse-checkout set \
  LICENSE \
  ReverseDocumentation.md \
  loader/build.sh \
  loader/src \
  loader/tests

git -C "${DEST}" fetch --prune origin "${SHA}"
git -C "${DEST}" checkout --detach "${SHA}"
git -C "${DEST}" reset --hard "${SHA}"

ACTUAL="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${SHA}" ]]; then
  echo "current-runtime shell identity mismatch: expected ${SHA}, got ${ACTUAL}" >&2
  exit 2
fi

for forbidden in gameassest gamelibs levelexample "install方法以及示例" loader/out; do
  if [[ -e "${DEST}/${forbidden}" ]]; then
    echo "refusing source preparation: proprietary/prebuilt path was checked out: ${forbidden}" >&2
    exit 3
  fi
done

for required in \
  LICENSE \
  ReverseDocumentation.md \
  loader/build.sh \
  loader/src/main.cpp \
  loader/src/game.cpp \
  loader/src/input.cpp \
  loader/src/render.cpp \
  loader/src/overlay.cpp; do
  if [[ ! -s "${DEST}/${required}" ]]; then
    echo "required source-only runtime file missing: ${required}" >&2
    exit 3
  fi
done

printf 'Prepared source-only ADOFAI 3.3.1 runtime shell at %s\n' "${DEST}"
printf 'Pinned commit: %s\n' "${ACTUAL}"
printf 'No game assets, game native libraries, sample copyrighted media or prebuilt loader binaries were checked out.\n'
