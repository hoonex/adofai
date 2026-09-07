# ADOFAI Modern Mobile Editor

Source-first modernization of the old ADOFAI Android custom editor. This repository contains patching/build/test tooling only; it does **not** contain or redistribute the proprietary game APK or game assets.

## Architecture

The old `V2.4.0 Custom.apk` is a behavioral reference, not the target runtime. Modern `.adofai` charts contain runtime/editor behavior that did not exist in 2.4.0, so teaching that old parser to ignore new keys would not make the missing event implementations work.

The current architecture is:

```text
current ADOFAI Android engine (validated against 3.3.1 / versionCode 300382)
    + Android-native editor shell in injected secondary DEX
    + current-runtime IL2CPP preview bridge in libOctober.so
    + split-APK-safe repack/signing tooling
```

Android 3.3.1 still exposes current chart/runtime symbols, but the serialized `scnEditor` scene itself is not packaged. The modern path therefore does **not** pretend that a hidden Unity editor scene can simply be enabled.

Pinned public source/model references are recorded in `upstream.lock.json`.

## Editor shell

`android/mobile-editor-shell/src/com/unity3d/player/MobileEditorShell.java` provides the first usable mobile editing slice:

- floating **Editor** launcher over the current Unity activity;
- full-screen Android-native editor surface with normal Android IME/touch handling;
- **Open** `.adofai` files through the Android file bridge;
- tolerant load handling for UTF-8 BOMs, trailing commas and raw JSON control characters;
- **Chart** tab for `pathData` and `angleData`;
- **Settings** tab for inspect/edit/add/delete;
- **Events** tab for `actions` and `decorations` object editing;
- **Raw** tab as a future-compatible fallback;
- unknown root fields and unknown event payloads preserved unless the user explicitly replaces them;
- **Save / Save As** through sibling temporary file + flush + `fsync` + atomic rename;
- **Preview** through a JNI/native queue drained on the Unity game thread.

The preview bridge resolves current `GCS` state and `scrController.LoadCustomLevel` through BNM at runtime and fails closed when the required surface is missing instead of invoking guessed addresses.

## Modern chart tooling

`tools/adofai_compat.py` provides source-side compatibility diagnostics/normalization:

- UTF-8 BOM handling;
- trailing-comma tolerance;
- raw control-character repair inside JSON strings;
- `pathData` and `angleData`;
- optional legacy path conversion;
- current action/decorations inventory;
- preservation of unknown/future payloads.

```bash
python3 tools/adofai_compat.py level.adofai
python3 tools/adofai_compat.py level.adofai --normalize normalized.adofai
```

## Reproducible payload build

CI and local builds produce:

```text
classes2.dex
libOctober.so
SHA256SUMS.txt
```

Pinned Android build environment:

- JDK 17
- Android platform 35
- build-tools 36.0.0
- Android NDK `29.0.14206865`

Local build:

```bash
bash scripts/build-payload.sh dist/payload
```

## Build from the current installed game

The recommended path for a user-owned Play installation is the split-aware flow. With USB debugging/ADB enabled:

```bash
bash scripts/build-from-installed-current.sh dist/device-build
```

That command:

1. pulls the installed `com.fizzd.connectedworlds` split APK set through ADB;
2. builds the DEX/native editor payload;
3. patches the base and arm64 splits;
4. preserves the large asset split instead of rebuilding it;
5. signs **every** split with the same persistent local key;
6. emits `INSTALL.txt` with the exact `adb install-multiple` command.

It intentionally does **not** uninstall or install anything. The Play-signed build and locally signed mod have different signing identities, so replacing the Play build is an explicit user action. Back up any app-local data you care about first, and keep the generated local keystore if you want later local updates to install over the same modded build.

Single-APK builds remain available when the target really is a monolithic APK:

```bash
bash scripts/build-modded-apk.sh /path/to/owned-current.apk dist/ADOFAI-Mobile-Editor.apk
```

## Real-device smoke evidence

After a locally patched build is installed, run:

```bash
bash scripts/run-device-editor-smoke.sh dist/device-smoke
```

The guided harness is deliberately non-destructive. It does not install, uninstall or clear app data. It captures evidence for:

- floating Editor launcher visibility;
- full-screen editor shell visibility;
- modern chart Open success;
- Save success;
- saved-chart reopen;
- native preview bridge installation;
- Preview reaching the current runtime `LoadCustomLevel` call.

For each stage it stores UIAutomator XML and screenshots. It also records package/device identity, PID-scoped logcat, `meminfo` and `gfxinfo` framestats, then emits `REPORT.md` with `PASS`, `FAIL` or `UNPROVEN` per evidence boundary.

A `LoadCustomLevel` call is **not** proof that every modern event rendered correctly. Representative chart/event/assets behavior still needs manual reconciliation on the device.

## Current verification boundary

Repository/CI evidence proves source preparation, tests, Java -> DEX compilation, arm64 NDK compilation and payload artifact generation. It does **not** by itself prove:

- installation of the locally signed split set on a physical device;
- touch/IME ergonomics across device shapes;
- every storage provider/path;
- every current/future event type;
- sibling audio/image/font/video asset behavior;
- large-chart runtime behavior;
- performance, thermal or battery quality.

Keep PR #1 in draft until the real-device smoke/evidence loop is completed on the exact target package and representative modern charts have been opened, edited, saved, reopened and previewed successfully.

## Binary policy

No proprietary ADOFAI APK, game library or game asset is committed here. Public third-party source references are pinned by immutable commit where applicable. User-owned game packages remain local to the user's machine/device.
