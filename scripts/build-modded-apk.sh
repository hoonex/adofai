#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <owned-current-game.apk> <output.apk>" >&2
  exit 2
fi

BASE="$1"
OUTPUT="$2"
PAYLOAD="${ROOT}/dist/payload"

bash "${ROOT}/scripts/build-payload.sh" "${PAYLOAD}"
bash "${ROOT}/scripts/repack-apk.sh" "${BASE}" "${PAYLOAD}" "${OUTPUT}"
