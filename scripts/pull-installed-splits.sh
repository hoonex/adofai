#!/usr/bin/env bash
set -euo pipefail

PACKAGE="${ADOFAI_PACKAGE:-com.fizzd.connectedworlds}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-dir>" >&2
  exit 2
fi

OUTPUT="$(realpath -m "$1")"
ADB_BIN="${ADB_BIN:-$(command -v adb || true)}"
if [[ -z "$ADB_BIN" ]]; then
  echo "adb is required (or set ADB_BIN)" >&2
  exit 2
fi

state="$($ADB_BIN get-state 2>/dev/null || true)"
if [[ "$state" != "device" ]]; then
  echo "no authorized Android device is connected through adb" >&2
  exit 2
fi

mapfile -t REMOTE_APKS < <(
  "$ADB_BIN" shell pm path "$PACKAGE" \
    | tr -d '\r' \
    | sed -n 's/^package://p'
)
if [[ ${#REMOTE_APKS[@]} -eq 0 ]]; then
  echo "package $PACKAGE is not installed or pm path returned no APKs" >&2
  exit 2
fi

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"
: > "$OUTPUT/INVENTORY.txt"

for remote in "${REMOTE_APKS[@]}"; do
  name="$(basename "$remote")"
  printf '%s -> %s\n' "$remote" "$name" | tee -a "$OUTPUT/INVENTORY.txt"
  "$ADB_BIN" pull "$remote" "$OUTPUT/$name"
done

if [[ ! -s "$OUTPUT/base.apk" ]]; then
  echo "base.apk was not collected" >&2
  exit 2
fi
if [[ ! -s "$OUTPUT/split_config.arm64_v8a.apk" ]]; then
  echo "arm64 split was not collected; current patch payload only supports arm64-v8a" >&2
  exit 2
fi

(
  cd "$OUTPUT"
  sha256sum ./*.apk > SHA256SUMS.txt
)

printf 'Collected %d installed ADOFAI split APK(s) into %s\n' "${#REMOTE_APKS[@]}" "$OUTPUT"
printf 'The files stay local; this script does not upload or redistribute the game.\n'
