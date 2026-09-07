#!/usr/bin/env bash
set -euo pipefail

PACKAGE="${ADOFAI_PACKAGE:-com.fizzd.connectedworlds}"
ADB_BIN="${ADB_BIN:-$(command -v adb || true)}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <evidence-dir>" >&2
  echo "" >&2
  echo "Runs a guided, non-destructive real-device smoke test against an already-installed" >&2
  echo "ADOFAI mobile-editor build. It never installs, uninstalls, clears app data, or changes signing state." >&2
  exit 2
fi

if [[ -z "$ADB_BIN" ]]; then
  echo "adb is required (or set ADB_BIN)" >&2
  exit 2
fi

state="$($ADB_BIN get-state 2>/dev/null || true)"
if [[ "$state" != "device" ]]; then
  echo "no authorized Android device is connected through adb" >&2
  exit 2
fi

if ! "$ADB_BIN" shell pm path "$PACKAGE" >/dev/null 2>&1; then
  echo "package $PACKAGE is not installed" >&2
  exit 2
fi

OUT="$(realpath -m "$1")"
rm -rf "$OUT"
mkdir -p "$OUT"

capture_ui() {
  local name="$1"
  local remote="/data/local/tmp/adofai-editor-ui.xml"
  if "$ADB_BIN" shell uiautomator dump --compressed "$remote" >/dev/null 2>&1; then
    "$ADB_BIN" exec-out cat "$remote" > "$OUT/$name.xml" 2>/dev/null || printf '<unavailable/>\n' > "$OUT/$name.xml"
    "$ADB_BIN" shell rm -f "$remote" >/dev/null 2>&1 || true
  else
    printf '<unavailable/>\n' > "$OUT/$name.xml"
  fi
  "$ADB_BIN" exec-out screencap -p > "$OUT/$name.png" 2>/dev/null || true
}

wait_for_process() {
  local pid=""
  for _ in $(seq 1 30); do
    pid="$($ADB_BIN shell pidof "$PACKAGE" 2>/dev/null | tr -d '\r' | awk '{print $1}')"
    if [[ -n "$pid" ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
    sleep 0.5
  done
  return 1
}

ui_state() {
  local file="$1"
  local marker="$2"
  if [[ ! -s "$file" || "$(cat "$file")" == '<unavailable/>' ]]; then
    printf 'UNPROVEN'
  elif grep -Fq "$marker" "$file"; then
    printf 'PASS'
  else
    printf 'FAIL'
  fi
}

log_state() {
  local marker="$1"
  if [[ ! -s "$OUT/runtime.log" ]]; then
    printf 'UNPROVEN'
  elif grep -Fq "$marker" "$OUT/runtime.log"; then
    printf 'PASS'
  else
    printf 'FAIL'
  fi
}

{
  echo "package=$PACKAGE"
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "serial=$($ADB_BIN get-serialno 2>/dev/null || true)"
  echo "model=$($ADB_BIN shell getprop ro.product.model 2>/dev/null | tr -d '\r')"
  echo "device=$($ADB_BIN shell getprop ro.product.device 2>/dev/null | tr -d '\r')"
  echo "android=$($ADB_BIN shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')"
  echo "sdk=$($ADB_BIN shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')"
} > "$OUT/device.txt"

"$ADB_BIN" shell pm path "$PACKAGE" | tr -d '\r' > "$OUT/pm-paths.txt"
"$ADB_BIN" shell dumpsys package "$PACKAGE" > "$OUT/package.txt"

printf '[1/6] Restarting the already-installed app (no data is cleared)...\n'
"$ADB_BIN" shell am force-stop "$PACKAGE"
"$ADB_BIN" shell monkey -p "$PACKAGE" -c android.intent.category.LAUNCHER 1 >/dev/null
PID="$(wait_for_process)" || {
  echo "ADOFAI did not start within the bounded launch window" >&2
  exit 1
}
printf '%s\n' "$PID" > "$OUT/pid.txt"
sleep 2
capture_ui "01-launcher"

printf '\n[2/6] On the phone, tap the floating Editor button.\n'
read -r -p 'When the full-screen ADOFAI Mobile Editor is visible, press Enter here... ' _
capture_ui "02-editor-open"

printf '\n[3/6] In the editor, tap Open and choose a representative modern .adofai chart.\n'
read -r -p 'When the chart has loaded and the status says Loaded successfully, press Enter... ' _
capture_ui "03-chart-loaded"

printf '\n[4/6] Change one harmless field (for example a test setting or geometry), then tap Save.\n'
read -r -p 'When the status says Saved, press Enter... ' _
capture_ui "04-chart-saved"

printf '\n[5/6] Close the editor, reopen it, use Open, and select the saved chart again.\n'
read -r -p 'When the saved chart has reopened successfully, press Enter... ' _
capture_ui "05-chart-reopened"

printf '\n[6/6] Tap Preview. The editor dialog should close and the current runtime should load the chart.\n'
read -r -p 'After the preview attempt has completed or visibly failed, press Enter... ' _
sleep 1
capture_ui "06-after-preview"

CURRENT_PID="$($ADB_BIN shell pidof "$PACKAGE" 2>/dev/null | tr -d '\r' | awk '{print $1}')"
if [[ -n "$CURRENT_PID" ]]; then
  if ! "$ADB_BIN" logcat -d --pid "$CURRENT_PID" -v threadtime > "$OUT/runtime.log" 2>/dev/null; then
    "$ADB_BIN" logcat -d -v threadtime 'ADOFAI.MobileEditor:*' 'IL2CPP_EXPORTS:*' '*:S' > "$OUT/runtime.log" 2>/dev/null || true
  fi
  "$ADB_BIN" shell dumpsys meminfo "$PACKAGE" > "$OUT/meminfo.txt" 2>/dev/null || true
  "$ADB_BIN" shell dumpsys gfxinfo "$PACKAGE" framestats > "$OUT/gfxinfo-framestats.txt" 2>/dev/null || true
else
  printf 'Target process exited before evidence collection.\n' > "$OUT/runtime.log"
fi

LAUNCHER_STATE="$(ui_state "$OUT/01-launcher.xml" 'text="Editor"')"
EDITOR_STATE="$(ui_state "$OUT/02-editor-open.xml" 'ADOFAI Mobile Editor')"
LOAD_STATE="$(ui_state "$OUT/03-chart-loaded.xml" 'Loaded successfully')"
SAVE_STATE="$(ui_state "$OUT/04-chart-saved.xml" 'Saved')"
REOPEN_STATE="$(ui_state "$OUT/05-chart-reopened.xml" 'Loaded successfully')"
BRIDGE_STATE="$(log_state 'Mobile editor preview bridge installed on Unity game-thread input poll')"
PREVIEW_STATE="$(log_state 'Mobile editor preview queued into current runtime')"
PREVIEW_FAIL_STATE="$(log_state 'Mobile editor preview request failed closed')"

cat > "$OUT/REPORT.md" <<EOF
# ADOFAI Mobile Editor device smoke evidence

Package: \`$PACKAGE\`
PID at launch: \`$PID\`

| Boundary | Result | Evidence |
| --- | --- | --- |
| Floating Editor launcher visible | $LAUNCHER_STATE | \`01-launcher.xml/png\` |
| Android-native editor shell visible | $EDITOR_STATE | \`02-editor-open.xml/png\` |
| Modern chart open reports success | $LOAD_STATE | \`03-chart-loaded.xml/png\` |
| Save reports success | $SAVE_STATE | \`04-chart-saved.xml/png\` |
| Saved chart reopens | $REOPEN_STATE | \`05-chart-reopened.xml/png\` |
| Native preview bridge installed | $BRIDGE_STATE | \`runtime.log\` |
| Preview reached current runtime LoadCustomLevel call | $PREVIEW_STATE | \`runtime.log\` |
| Preview fail-closed marker observed | $PREVIEW_FAIL_STATE | \`runtime.log\` (PASS here means a failure marker was present) |

Interpretation rules:

- \`PASS\` means the captured UI/log contains the exact repository-defined marker.
- \`FAIL\` means evidence was captured but the marker was absent.
- \`UNPROVEN\` means the relevant Android evidence could not be captured.
- A successful \`LoadCustomLevel\` call is not by itself proof that every chart event rendered correctly.
- Screenshots and raw PID-scoped logs remain in this directory for manual reconciliation.
- This harness does not prove performance, thermal, battery, every event type, every storage provider, or every device shape.
EOF

printf '\nEvidence captured in %s\n' "$OUT"
printf 'Summary: launcher=%s editor=%s load=%s save=%s reopen=%s bridge=%s preview=%s\n' \
  "$LAUNCHER_STATE" "$EDITOR_STATE" "$LOAD_STATE" "$SAVE_STATE" "$REOPEN_STATE" "$BRIDGE_STATE" "$PREVIEW_STATE"
printf 'Read %s/REPORT.md and inspect the screenshots/log before promoting device-runtime claims.\n' "$OUT"
