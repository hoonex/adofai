#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <installed-splits-dir> <payload-dir> <output-dir>" >&2
  echo "expected current ADOFAI 3.3 layout: base.apk + split_config.arm64_v8a.apk + other installed splits" >&2
  exit 2
fi

INPUT_DIR="$(realpath "$1")"
PAYLOAD="$(realpath "$2")"
OUTPUT_DIR="$(realpath -m "$3")"
BASE="${INPUT_DIR}/base.apk"
ARM64="${INPUT_DIR}/split_config.arm64_v8a.apk"
DEX="${PAYLOAD}/classes2.dex"
LIB="${PAYLOAD}/libOctober.so"
OUTPUT_APKS="${OUTPUT_DIR}/apks"

for path in "$BASE" "$ARM64" "$DEX" "$LIB"; do
  if [[ ! -s "$path" ]]; then
    echo "required input missing or empty: $path" >&2
    exit 2
  fi
done

mapfile -t INPUT_APKS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.apk' -print | sort)
if [[ ${#INPUT_APKS[@]} -lt 2 ]]; then
  echo "expected a split APK set, found ${#INPUT_APKS[@]} APK(s) in $INPUT_DIR" >&2
  exit 2
fi

APKTOOL_BIN="${APKTOOL_BIN:-$(command -v apktool || true)}"
if [[ -z "$APKTOOL_BIN" ]]; then
  echo "apktool is required (or set APKTOOL_BIN)" >&2
  exit 2
fi

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
find_build_tool() {
  local tool="$1"
  local explicit="$2"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return
  fi
  local from_path
  from_path="$(command -v "$tool" || true)"
  if [[ -n "$from_path" ]]; then
    printf '%s\n' "$from_path"
    return
  fi
  if [[ -n "$SDK" && -d "$SDK/build-tools" ]]; then
    local newest
    newest="$(find "$SDK/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n1)"
    if [[ -n "$newest" && -x "$newest/$tool" ]]; then
      printf '%s\n' "$newest/$tool"
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

KEYSTORE="${ADOFAI_KEYSTORE:-${HOME}/.adofai-mobile-editor/debug.keystore}"
KEY_ALIAS="${ADOFAI_KEY_ALIAS:-adofai-editor}"
KS_PASS="${ADOFAI_KS_PASS:-android}"
KEY_PASS="${ADOFAI_KEY_PASS:-${KS_PASS}}"

if [[ ! -f "$KEYSTORE" ]]; then
  if [[ -n "${ADOFAI_KEYSTORE:-}" ]]; then
    echo "configured ADOFAI_KEYSTORE does not exist: $KEYSTORE" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$KEYSTORE")"
  keytool -genkeypair \
    -keystore "$KEYSTORE" \
    -storepass "$KS_PASS" \
    -keypass "$KEY_PASS" \
    -alias "$KEY_ALIAS" \
    -keyalg RSA \
    -keysize 3072 \
    -validity 10000 \
    -dname "CN=ADOFAI Mobile Editor Local Debug,O=Local Development,C=KR" \
    -noprompt
  echo "Created persistent local sideload key: $KEYSTORE"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
DECODED_BASE="$TMP/base-decoded"
REBUILT_BASE="$TMP/base-rebuilt.apk"
INJECTED_BASE="$TMP/base-injected.apk"
INJECTED_ARM64="$TMP/arm64-injected.apk"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_APKS"

# Base split owns the manifest, UnityPlayerActivity and managed DEX payload.
"$APKTOOL_BIN" d -f "$BASE" -o "$DECODED_BASE"
python3 "$ROOT/tools/patch_android_manifest.py" "$DECODED_BASE/AndroidManifest.xml"
python3 "$ROOT/tools/patch_unity_activity.py" "$DECODED_BASE"
"$APKTOOL_BIN" b "$DECODED_BASE" -o "$REBUILT_BASE"
python3 "$ROOT/tools/inject_split_component.py" \
  "$REBUILT_BASE" "$INJECTED_BASE" \
  --dex "$DEX"
"$ZIPALIGN_BIN" -P 16 -f 4 "$INJECTED_BASE" "$OUTPUT_APKS/base.apk"

# ABI split owns native Unity/IL2CPP libraries. Keep October uncompressed and
# 16 KiB-aligned so modern Android can mmap native libraries safely.
python3 "$ROOT/tools/inject_split_component.py" \
  "$ARM64" "$INJECTED_ARM64" \
  --library "$LIB"
"$ZIPALIGN_BIN" -P 16 -f 4 "$INJECTED_ARM64" "$OUTPUT_APKS/$(basename "$ARM64")"

# Copy every other installed split byte-for-byte before resigning. In particular,
# do not decode/rebuild the multi-gigabyte asset split.
for apk in "${INPUT_APKS[@]}"; do
  name="$(basename "$apk")"
  if [[ "$name" == "base.apk" || "$name" == "$(basename "$ARM64")" ]]; then
    continue
  fi
  cp -p "$apk" "$OUTPUT_APKS/$name"
done

# Every member of a split install must use the same signing identity. Signing is
# intentionally the final mutation; zipalign must run before apksigner.
CERT_DIGEST=""
for apk in "$OUTPUT_APKS"/*.apk; do
  "$APKSIGNER_BIN" sign \
    --ks "$KEYSTORE" \
    --ks-key-alias "$KEY_ALIAS" \
    --ks-pass "pass:$KS_PASS" \
    --key-pass "pass:$KEY_PASS" \
    --v4-signing-enabled false \
    "$apk"

  VERIFY_OUTPUT="$($APKSIGNER_BIN verify --verbose --print-certs "$apk")"
  printf '%s\n' "$VERIFY_OUTPUT"
  digest="$(printf '%s\n' "$VERIFY_OUTPUT" | sed -n 's/^Signer #1 certificate SHA-256 digest: //p' | head -n1)"
  if [[ -z "$digest" ]]; then
    echo "could not resolve signer digest for $apk" >&2
    exit 2
  fi
  if [[ -z "$CERT_DIGEST" ]]; then
    CERT_DIGEST="$digest"
  elif [[ "$digest" != "$CERT_DIGEST" ]]; then
    echo "split signer mismatch: $apk has $digest, expected $CERT_DIGEST" >&2
    exit 2
  fi
done

# Mutated APK alignment is verified after signing because signing itself does not
# change ZIP entry offsets. Pass-through Play splits retain their original layout.
"$ZIPALIGN_BIN" -c -P 16 4 "$OUTPUT_APKS/base.apk"
"$ZIPALIGN_BIN" -c -P 16 4 "$OUTPUT_APKS/$(basename "$ARM64")"

(
  cd "$OUTPUT_APKS"
  sha256sum ./*.apk > "$OUTPUT_DIR/SHA256SUMS.txt"
)

{
  echo "ADOFAI split mod package"
  echo "Signer SHA-256: $CERT_DIGEST"
  echo
  echo "The Play-signed installed app cannot be updated in place with this locally signed mod."
  echo "Back up any app-local data you care about before uninstalling the Play build."
  echo
  printf 'After the original package is removed, install all splits together with:\n  adb install-multiple --no-incremental'
  for apk in "$OUTPUT_APKS"/*.apk; do
    printf ' "%s"' "$apk"
  done
  printf '\n'
} > "$OUTPUT_DIR/INSTALL.txt"

printf 'Prepared signed split package set: %s\n' "$OUTPUT_APKS"
printf 'All split APKs use signer SHA-256: %s\n' "$CERT_DIGEST"
printf 'Installation is intentionally not automatic; read %s/INSTALL.txt first.\n' "$OUTPUT_DIR"
