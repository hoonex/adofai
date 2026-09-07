#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <work-dir>" >&2
  echo "" >&2
  echo "Validate and collect the currently installed, user-owned ADOFAI split APKs through adb," >&2
  echo "build the modern editor payload, and produce a locally signed split set." >&2
  echo "This script never uninstalls or installs the game automatically." >&2
  exit 2
fi

WORK="$(realpath -m "$1")"
SOURCE_DIR="$WORK/installed-splits"
PAYLOAD_DIR="$WORK/payload"
OUTPUT_DIR="$WORK/modded-splits"
RUNTIME_REPORT="$WORK/installed-runtime.txt"

for command in adb realpath; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required" >&2
    exit 2
  fi
done

mkdir -p "$WORK"

printf '[1/4] Verifying the installed runtime matches the source-validated hook target...\n'
bash "$ROOT/scripts/verify-installed-runtime.sh" "$RUNTIME_REPORT"

printf '[2/4] Collecting the installed ADOFAI split set via adb...\n'
bash "$ROOT/scripts/pull-installed-splits.sh" "$SOURCE_DIR"

printf '[3/4] Building the current mobile editor DEX/native payload...\n'
bash "$ROOT/scripts/build-payload.sh" "$PAYLOAD_DIR"

printf '[4/4] Repacking and signing the complete split set...\n'
bash "$ROOT/scripts/repack-split-apks.sh" "$SOURCE_DIR" "$PAYLOAD_DIR" "$OUTPUT_DIR"

cat > "$WORK/README-NEXT.txt" <<EOF
ADOFAI current-install mobile editor build completed.

Validated installed runtime report:
  $RUNTIME_REPORT

Original installed splits (local copy):
  $SOURCE_DIR

Built editor payload:
  $PAYLOAD_DIR

Locally signed modded split set:
  $OUTPUT_DIR/apks

IMPORTANT:
- The patch profile is currently validated for ADOFAI 3.3.1 / versionCode 300382.
- Version drift fails closed before the game APKs are pulled or modified; reconcile a newer runtime in source first.
- The Play-installed package and this local build use different signing keys.
- This workflow intentionally did NOT uninstall or install anything.
- Back up app-local data you care about before replacing the Play build.
- Read $OUTPUT_DIR/INSTALL.txt for the exact adb install-multiple command.
- Keep the generated keystore if you want later local updates to install over this modded build.
EOF

printf '\nBuild complete: %s\n' "$OUTPUT_DIR/apks"
printf 'Validated runtime identity is recorded in %s\n' "$RUNTIME_REPORT"
printf 'No package was installed or removed. Read %s and %s/INSTALL.txt before device installation.\n' \
  "$WORK/README-NEXT.txt" "$OUTPUT_DIR"
