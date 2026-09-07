# ADOFAI Modern Mobile Editor

Source-first modernization of the old ADOFAI Android custom editor. The project patches an **APK supplied by the user**; it does not contain or redistribute the game.

## Why this is not based on 2.4.0

The old `V2.4.0 Custom.apk` can be useful as a behavioral reference, but modern `.adofai` charts contain event/runtime behavior that did not exist in the 2.4.0 engine. Making its JSON parser accept new keys would not make the old engine render, edit, preview, or save those features correctly.

The patch therefore uses a current-enough Android ADOFAI engine and repairs/exposes its embedded editor.

Pinned references:

- Android IL2CPP hook baseline: `HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level@74bcc7a0d8c8be1267504e21e28a35e199b5d4eb`
- Modern `.adofai` model/event reference: `adofaiex/ADOFAI-JS@1f66bfa8c4146853d239c80c47ed2168d9208d02`
- exact pins: `upstream.lock.json`

## Implemented

### Modern chart compatibility

`tools/adofai_compat.py` is lossless-first. It handles:

- UTF-8 BOM;
- trailing commas;
- raw control characters inside strings;
- both `pathData` and `angleData`;
- optional legacy `pathData` -> `angleData` conversion;
- current action/decorations inventory, including newer input/particle event families;
- unknown future top-level/settings/event payloads without silently deleting them.

```bash
python3 tools/adofai_compat.py level.adofai
python3 tools/adofai_compat.py level.adofai --normalize normalized.adofai
```

### Mobile editor runtime fixes

The source preparation layer verifies exact upstream Git blob identities before making each change. It currently fixes:

- missing installation of the existing `ADOBase.get_isMobile` hook;
- editor scenes using the desktop editor layout/input branch rather than the restricted mobile branch;
- `StandaloneFileBrowser.OpenFilePanel` Android bridge;
- previously-unwired `SaveFilePanel` support;
- previously-unwired `OpenFolderPanel` support;
- safe empty-array/string cancellation results instead of null-return edge cases;
- serialized file-dialog calls;
- JNI attach/error paths that previously could leave the caller waiting;
- `UnityPlayer.currentActivity` as the primary Activity source, with the older reflection route only as fallback;
- Android 11+ all-files-access guidance for the raw-path level browser;
- immediate completion on Activity/permission failures instead of leaving `isDone=false` until timeout.

The transforms live in:

- `tools/apply_hitmargin_editor_mode.py`
- `tools/apply_hitmargin_file_dialogs.py`
- `tools/apply_hitmargin_storage_guard.py`

Run all of them reproducibly with:

```bash
bash scripts/prepare-upstream.sh
```

### Verified Android payload build

CI builds both payload components with a pinned Android toolchain:

- `classes2.dex` — Java file-browser bridge
- `libOctober.so` — arm64-v8a IL2CPP/JNI hook library

Current build pin is Android NDK `29.0.14206865`. CI also records SHA-256 digests and uploads the two files as a workflow artifact.

Local equivalent:

```bash
bash scripts/build-payload.sh
```

Output:

```text
dist/payload/classes2.dex
dist/payload/libOctober.so
dist/payload/SHA256SUMS.txt
```

## Build a patched APK from your own current game APK

Requirements:

- JDK 17
- Android SDK with platform 35, build-tools 36.0.0 and NDK 29.0.14206865
- `apktool`
- `zipalign`
- `apksigner`

Then:

```bash
bash scripts/build-modded-apk.sh /path/to/your-current-adofai.apk dist/ADOFAI-Mobile-Editor.apk
```

The pipeline:

1. resolves the exact pinned source;
2. applies the identity-checked runtime fixes;
3. builds DEX + native arm64 payloads;
4. decodes the user-supplied APK with apktool;
5. idempotently adds required storage manifest settings;
6. rebuilds the APK;
7. chooses a free `classesN.dex` slot instead of overwriting an existing secondary dex;
8. injects the payload and replaces `lib/arm64-v8a/libOctober.so`;
9. zipaligns and signs the result;
10. verifies the final APK signature.

By default the script creates/reuses a local debug signing key under `~/.adofai-mobile-editor/`. A build signed with a different key cannot update the official-store installation in place, so keep that key for subsequent patched builds or provide your own through the documented environment variables in `scripts/repack-apk.sh`.

## Verification status

Proven on branch commit `416c7521bac3a1dad223b538fbf9821c10a4cae6` by GitHub Actions run `34072702544`:

- Python and shell tooling syntax checks;
- compatibility tests and APK packaging unit tests;
- exact pinned-source preparation and all runtime transforms;
- editor-critical hook/storage guard presence checks;
- Java -> DEX compilation;
- arm64 native NDK compilation;
- SHA-256 generation and workflow-artifact upload.

The resulting `mobile-editor-patch-binaries` artifact is tied to that exact source commit. This proves source transformation and payload compilation, not in-game behavior.

Still requires a user-supplied **current** ADOFAI Android APK for the last evidence boundary:

- repack/sign that exact game build;
- install on a real Android device;
- open representative modern maps;
- edit and save them;
- close/reopen and compare preserved chart data/assets;
- exercise touch/keyboard/viewport/editor controls.

The currently supplied `V2.4.0 Custom.apk` is intentionally not used as that final target because its engine predates modern chart/runtime features. Until a current APK and real-device checks are available, the repository must not claim every editor function is device-verified.

## Binary policy

No proprietary ADOFAI APK or game assets are committed here. The old uploaded 2.4.0 custom build is not redistributed and is not used as the target engine for modern chart support.
