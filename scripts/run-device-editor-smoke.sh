#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="${ADOFAI_PACKAGE:-com.fizzd.connectedworlds}"
ADB_BIN="${ADB_BIN:-$(command -v adb || true)}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <evidence-dir>" >&2
  echo "" >&2
  echo "Runs a guided, non-destructive real-device smoke test against an already-installed" >&2
  echo "ADOFAI mobile-editor build. It never installs, uninstalls, clears app data, or changes signing state." >&2
  echo "The guided edit is performed only after Save As creates a separate smoke-test chart copy." >&2
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

PM_PATHS="$($ADB_BIN shell pm path "$PACKAGE" 2>/dev/null | tr -d '\r')"
if ! grep -q '^package:' <<<"$PM_PATHS"; then
  echo "package $PACKAGE is not installed or pm path returned no APKs" >&2
  exit 2
fi

OUT="$(realpath -m "$1")"
is_same_or_parent() {
  local parent="$1"
  local child="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

# OUT is deliberately recreated for each run. Refuse any path whose recursive
# deletion could consume a filesystem root, repository/work directory, or home.
PWD_REAL="$(pwd -P)"
HOME_REAL="$(realpath -m "${HOME:-/__adofai_no_home__}")"
if [[ -z "$OUT" || "$OUT" == "/" || "$OUT" =~ ^/[^/]+$ ]]; then
  echo "refusing dangerous smoke evidence directory: ${OUT:-<empty>}" >&2
  exit 2
fi
for protected in "$ROOT" "$PWD_REAL" "$HOME_REAL"; do
  if is_same_or_parent "$OUT" "$protected"; then
    echo "refusing smoke evidence directory that contains protected path $protected: $OUT" >&2
    exit 2
  fi
done

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
  if [[ ! -s "$file" ]] || grep -Fxq '<unavailable/>' "$file"; then
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

log_presence() {
  local marker="$1"
  if [[ ! -s "$OUT/runtime.log" ]]; then
    printf 'UNPROVEN'
  elif grep -Fq "$marker" "$OUT/runtime.log"; then
    printf 'PRESENT'
  else
    printf 'ABSENT'
  fi
}

capture_runtime_logs() {
  local launch_pid="$1"
  local current_pid="$2"
  : > "$OUT/runtime.log"

  if ! "$ADB_BIN" logcat -d --pid "$launch_pid" -v threadtime > "$OUT/runtime.log" 2>/dev/null; then
    "$ADB_BIN" logcat -d -v threadtime 'ADOFAI.MobileEditor:*' 'IL2CPP_EXPORTS:*' '*:S' > "$OUT/runtime.log" 2>/dev/null || true
    return
  fi

  if [[ -n "$current_pid" && "$current_pid" != "$launch_pid" ]]; then
    {
      printf '\n--- process changed after launch: %s -> %s ---\n' "$launch_pid" "$current_pid"
      "$ADB_BIN" logcat -d --pid "$current_pid" -v threadtime 2>/dev/null || true
    } >> "$OUT/runtime.log"
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

printf '%s\n' "$PM_PATHS" > "$OUT/pm-paths.txt"
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
printf 'Do not edit the source chart yet; step 4 creates a separate smoke-test copy first.\n'
read -r -p 'When the chart has loaded and the status says Loaded successfully, press Enter... ' _
capture_ui "03-chart-loaded"

printf '\n[4/6] First tap Save As and choose a DIFFERENT disposable .adofai filename/path.\n'
printf 'After Save As reports Saved, change one harmless field on that copy and tap Save.\n'
read -r -p 'When the copied chart has been edited and the status says Saved again, press Enter... ' _
capture_ui "04-chart-saved"

printf '\n[5/6] Close the editor, reopen it, use Open, and select the saved SMOKE COPY again.\n'
read -r -p 'When the saved smoke copy has reopened successfully, press Enter... ' _
capture_ui "05-chart-reopened"

printf '\n[6/6] Tap Preview on the smoke copy. The editor dialog should close and the current runtime should load it.\n'
read -r -p 'After the preview attempt has completed or visibly failed, press Enter... ' _
sleep 1
capture_ui "06-after-preview"

CURRENT_PID="$($ADB_BIN shell pidof "$PACKAGE" 2>/dev/null | tr -d '\r' | awk '{print $1}')"
capture_runtime_logs "$PID" "$CURRENT_PID"
if [[ -n "$CURRENT_PID" ]]; then
  "$ADB_BIN" shell dumpsys meminfo "$PACKAGE" > "$OUT/meminfo.txt" 2>/dev/null || true
  "$ADB_BIN" shell dumpsys gfxinfo "$PACKAGE" framestats > "$OUT/gfxinfo-framestats.txt" 2>/dev/null || true
else
  printf 'Target process was not running at final evidence collection.\n' > "$OUT/process-exit.txt"
fi

LAUNCHER_STATE="$(ui_state "$OUT/01-launcher.xml" 'text="Editor"')"
EDITOR_STATE="$(ui_state "$OUT/02-editor-open.xml" 'ADOFAI Mobile Editor')"
LOAD_STATE="$(ui_state "$OUT/03-chart-loaded.xml" 'Loaded successfully')"
SAVE_STATE="$(ui_state "$OUT/04-chart-saved.xml" 'Saved')"
REOPEN_STATE="$(ui_state "$OUT/05-chart-reopened.xml" 'Loaded successfully')"
SHELL_BOOTSTRAP_STATE="$(log_state 'MobileEditorShell launcher installed through injected DEX loader')"
BRIDGE_STATE="$(log_state 'Mobile editor preview bridge installed on Unity game-thread input poll')"
PREVIEW_STATE="$(log_state 'Mobile editor preview queued into current runtime')"
PREVIEW_FAILURE_MARKER="$(log_presence 'Mobile editor preview request failed closed')"

cat > "$OUT/REPORT.md" <<EOF
# ADOFAI Mobile Editor device smoke evidence

Package: \`$PACKAGE\`
PID at launch: \`$PID\`
PID at final capture: \`${CURRENT_PID:-not-running}\`

| Boundary | Result | Evidence |
| --- | --- | --- |
| Injected DEX editor bootstrap executed | $SHELL_BOOTSTRAP_STATE | \`runtime.log\` |
| Floating Editor launcher visible | $LAUNCHER_STATE | \`01-launcher.xml/png\` |
| Android-native editor shell visible | $EDITOR_STATE | \`02-editor-open.xml/png\` |
| Modern chart open reports success | $LOAD_STATE | \`03-chart-loaded.xml/png\` |
| Save-As smoke copy + edit/save reports success | $SAVE_STATE | \`04-chart-saved.xml/png\` |
| Saved smoke copy reopens | $REOPEN_STATE | \`05-chart-reopened.xml/png\` |
| Native preview bridge installed | $BRIDGE_STATE | \`runtime.log\` |
| Preview reached current runtime LoadCustomLevel call | $PREVIEW_STATE | \`runtime.log\` |
| Preview fail-closed marker | $PREVIEW_FAILURE_MARKER | \`runtime.log\` |

Interpretation rules:

- \`PASS\` means the captured UI/log contains the exact repository-defined success marker.
- \`FAIL\` means evidence was captured but the success marker was absent.
- \`UNPROVEN\` means the relevant Android evidence could not be captured.
- The fail-closed row uses \`PRESENT\` / \`ABSENT\`; \`PRESENT\` is a runtime failure signal, not a pass.
- The harness requires Save As to a different disposable path before any edit; preserve the original chart unchanged.
- The UI evidence proves the Save status marker, but the operator must still verify the Save As destination is different from the source path.
- A successful \`LoadCustomLevel\` call is not by itself proof that every chart event rendered correctly.
- Screenshots and raw PID-scoped logs remain in this directory for manual reconciliation.
- This harness does not prove performance, thermal, battery, every event type, every storage provider, or every device shape.
EOF

printf '\nEvidence captured in %s\n' "$OUT"
printf 'Summary: bootstrap=%s launcher=%s editor=%s load=%s save=%s reopen=%s bridge=%s preview=%s preview-failure=%s\n' \
  "$SHELL_BOOTSTRAP_STATE" "$LAUNCHER_STATE" "$EDITOR_STATE" "$LOAD_STATE" "$SAVE_STATE" "$REOPEN_STATE" \
  "$BRIDGE_STATE" "$PREVIEW_STATE" "$PREVIEW_FAILURE_MARKER"
printf 'Read %s/REPORT.md and inspect the screenshots/log before promoting device-runtime claims.\n' "$OUT"
