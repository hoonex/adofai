#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <base.apk> <payload-dir> <output.apk>" >&2
  exit 2
fi

BASE="$(realpath "$1")"
PAYLOAD="$(realpath "$2")"
OUTPUT="$(realpath -m "$3")"
DEX="${PAYLOAD}/classes2.dex"
LIB="${PAYLOAD}/libOctober.so"

if [[ ! -f "${BASE}" ]]; then
  echo "base APK not found: ${BASE}" >&2
  exit 2
fi
if [[ ! -s "${DEX}" || ! -s "${LIB}" ]]; then
  echo "payload must contain classes2.dex and libOctober.so" >&2
  exit 2
fi
if [[ "${BASE}" == "${OUTPUT}" ]]; then
  echo "refusing to overwrite the input APK in place" >&2
  exit 2
fi

APKTOOL_BIN="${APKTOOL_BIN:-$(command -v apktool || true)}"
if [[ -z "${APKTOOL_BIN}" ]]; then
  echo "apktool is required (or set APKTOOL_BIN)" >&2
  exit 2
fi

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
find_build_tool() {
  local tool="$1"
  local explicit="$2"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return
  fi
  local from_path
  from_path="$(command -v "${tool}" || true)"
  if [[ -n "${from_path}" ]]; then
    printf '%s\n' "${from_path}"
    return
  fi
  if [[ -n "${SDK}" && -d "${SDK}/build-tools" ]]; then
    local newest
    newest="$(find "${SDK}/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n1)"
    if [[ -n "${newest}" && -x "${newest}/${tool}" ]]; then
      printf '%s\n' "${newest}/${tool}"
      return
    fi
  fi
  return 1
}

ZIPALIGN_BIN="$(find_build_tool zipalign "${ZIPALIGN_BIN:-}")" || {
  echo "zipalign is required (or set ZIPALIGN_BIN)" >&2
  exit 2
}
APKSIGNER_BIN="$(find_build_tool apksigner "${APKSIGNER_BIN:-}")" || {
  echo "apksigner is required (or set APKSIGNER_BIN)" >&2
  exit 2
}
command -v keytool >/dev/null || {
  echo "keytool is required" >&2
  exit 2
}

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
DECODED="${TMP}/decoded"
REB="${TMP}/rebuilt.apk"
INJECTED="${TMP}/injected.apk"
ALIGNED="${TMP}/aligned.apk"

"${APKTOOL_BIN}" d -f "${BASE}" -o "${DECODED}"
python3 "${ROOT}/tools/patch_android_manifest.py" "${DECODED}/AndroidManifest.xml"
"${APKTOOL_BIN}" b "${DECODED}" -o "${REB}"

python3 "${ROOT}/tools/inject_apk_payload.py" \
  "${REB}" "${INJECTED}" \
  --dex "${DEX}" \
  --library "${LIB}"

"${ZIPALIGN_BIN}" -f -p 4 "${INJECTED}" "${ALIGNED}"

KEYSTORE="${ADOFAI_KEYSTORE:-${HOME}/.adofai-mobile-editor/debug.keystore}"
KEY_ALIAS="${ADOFAI_KEY_ALIAS:-adofai-editor}"
KS_PASS="${ADOFAI_KS_PASS:-android}"
KEY_PASS="${ADOFAI_KEY_PASS:-${KS_PASS}}"

if [[ ! -f "${KEYSTORE}" ]]; then
  if [[ -n "${ADOFAI_KEYSTORE:-}" ]]; then
    echo "configured ADOFAI_KEYSTORE does not exist: ${KEYSTORE}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${KEYSTORE}")"
  keytool -genkeypair \
    -keystore "${KEYSTORE}" \
    -storepass "${KS_PASS}" \
    -keypass "${KEY_PASS}" \
    -alias "${KEY_ALIAS}" \
    -keyalg RSA \
    -keysize 3072 \
    -validity 10000 \
    -dname "CN=ADOFAI Mobile Editor Local Debug,O=Local Development,C=KR" \
    -noprompt
  echo "Created persistent local sideload key: ${KEYSTORE}"
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${APKSIGNER_BIN}" sign \
  --ks "${KEYSTORE}" \
  --ks-key-alias "${KEY_ALIAS}" \
  --ks-pass "pass:${KS_PASS}" \
  --key-pass "pass:${KEY_PASS}" \
  --out "${OUTPUT}" \
  "${ALIGNED}"

"${APKSIGNER_BIN}" verify --verbose --print-certs "${OUTPUT}"
printf 'Signed patched APK: %s\n' "${OUTPUT}"
printf 'Note: a differently signed build cannot update an official-store install in place.\n'
