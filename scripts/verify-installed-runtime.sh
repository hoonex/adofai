#!/usr/bin/env bash
set -euo pipefail

PACKAGE="${ADOFAI_PACKAGE:-com.fizzd.connectedworlds}"
EXPECTED_VERSION_NAME="${ADOFAI_EXPECTED_VERSION_NAME:-3.3.1}"
EXPECTED_VERSION_CODE="${ADOFAI_EXPECTED_VERSION_CODE:-300382}"
ADB_BIN="${ADB_BIN:-$(command -v adb || true)}"

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [report-file]" >&2
  exit 2
fi

REPORT="${1:-}"
if [[ -z "$ADB_BIN" ]]; then
  echo "adb is required (or set ADB_BIN)" >&2
  exit 2
fi

state="$($ADB_BIN get-state 2>/dev/null || true)"
if [[ "$state" != "device" ]]; then
  echo "no authorized Android device is connected through adb" >&2
  exit 2
fi

PM_PATHS="$($ADB_BIN shell pm path "$PACKAGE" 2>/dev/null | tr -d '\r')"
if ! grep -q '^package:' <<<"$PM_PATHS"; then
  echo "package $PACKAGE is not installed or pm path returned no APKs" >&2
  exit 2
fi

DUMPSYS="$($ADB_BIN shell dumpsys package "$PACKAGE" 2>/dev/null | tr -d '\r')"
VERSION_NAME="$(sed -n 's/^[[:space:]]*versionName=//p' <<<"$DUMPSYS" | head -n1 | xargs)"
VERSION_CODE="$(sed -n 's/^[[:space:]]*versionCode=\([0-9][0-9]*\).*$/\1/p' <<<"$DUMPSYS" | head -n1)"
PRIMARY_ABI="$(sed -n 's/^[[:space:]]*primaryCpuAbi=//p' <<<"$DUMPSYS" | head -n1 | xargs)"

BASE_COUNT="$(grep -c '/base\.apk$' <<<"$PM_PATHS" || true)"
ARM64_COUNT="$(grep -c '/split_config\.arm64_v8a\.apk$' <<<"$PM_PATHS" || true)"
APK_COUNT="$(grep -c '^package:' <<<"$PM_PATHS" || true)"

STATUS="compatible"
REASON="exact validated runtime identity"

if [[ -z "$VERSION_NAME" || -z "$VERSION_CODE" ]]; then
  STATUS="blocked"
  REASON="could not resolve installed versionName/versionCode from dumpsys package"
elif [[ "$VERSION_NAME" != "$EXPECTED_VERSION_NAME" || "$VERSION_CODE" != "$EXPECTED_VERSION_CODE" ]]; then
  STATUS="blocked"
  REASON="installed runtime differs from the source-verified hook target"
elif [[ "$BASE_COUNT" -ne 1 ]]; then
  STATUS="blocked"
  REASON="expected exactly one base.apk in installed package paths"
elif [[ "$ARM64_COUNT" -ne 1 ]]; then
  STATUS="blocked"
  REASON="expected split_config.arm64_v8a.apk; current payload is arm64-v8a only"
elif [[ -n "$PRIMARY_ABI" && "$PRIMARY_ABI" != "arm64-v8a" && "$PRIMARY_ABI" != "null" ]]; then
  STATUS="blocked"
  REASON="installed primaryCpuAbi is not arm64-v8a"
fi

emit_report() {
  cat <<EOF
ADOFAI installed runtime preflight
status=$STATUS
reason=$REASON
package=$PACKAGE
versionName=${VERSION_NAME:-unknown}
versionCode=${VERSION_CODE:-unknown}
expectedVersionName=$EXPECTED_VERSION_NAME
expectedVersionCode=$EXPECTED_VERSION_CODE
primaryCpuAbi=${PRIMARY_ABI:-unknown}
installedApkCount=$APK_COUNT
baseApkCount=$BASE_COUNT
arm64SplitCount=$ARM64_COUNT
EOF
}

if [[ -n "$REPORT" ]]; then
  REPORT="$(realpath -m "$REPORT")"
  mkdir -p "$(dirname "$REPORT")"
  emit_report > "$REPORT"
fi
emit_report

if [[ "$STATUS" != "compatible" ]]; then
  cat >&2 <<EOF
Refusing to build the runtime patch for this installed package.
The current hook/profile is evidence-backed for ADOFAI ${EXPECTED_VERSION_NAME} / versionCode ${EXPECTED_VERSION_CODE};
version drift must be inspected and reconciled in source before repacking.
EOF
  exit 3
fi

printf 'Installed runtime matches the validated patch target.\n'
