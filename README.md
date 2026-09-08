# ADOFAI Companion Editor

Non-root Android companion editor for user-authored ADOFAI custom levels. The canonical product keeps the official Google Play ADOFAI installation untouched and runs the editor as a separate app (`dev.hoonex.adofai.companion`). This repository does **not** contain or redistribute the proprietary ADOFAI APK or game assets.

## Canonical architecture

```text
ADOFAI Companion Editor
    ├─ Open .adofai / local .zip through Android SAF
    ├─ ZIP URL (historical Open From URL-style input)
    ├─ safe app-private bundle workspace
    │   └─ main.adofai + sibling audio/images/decorations stay together
    ├─ loss-preserving chart editor
    ├─ repackage current workspace as ZIP
    ├─ loopback-only http://127.0.0.1:<port>/.../level.zip
    └─ explicit ZIP-URL handoff to official ADOFAI 3.3.1

Official Google Play ADOFAI
    └─ com.fizzd.connectedworlds / 3.3.1 / versionCode 300382
       remains installed and unmodified
```

The canonical path does **not** root the device, use Magisk/Zygisk, patch or resign the Play APK, replace its signing identity, bypass licensing, or bundle a clean-room gameplay player.

Legacy patcher/Zygisk experiments remain repository history/reference only. Their workflows are manual-only and are not part of the normal Companion Editor build.

## Why ZIP bundles are first-class

Real custom levels are commonly not a standalone JSON file. A package can contain:

```text
Level Name/
├─ main.adofai
├─ song.ogg
├─ background.jpg
├─ decoration.png
└─ ...
```

`main.adofai` refers to those assets by relative filename. Detaching only the chart breaks song/background/decoration resolution.

Older mobile editor builds exposed **Open From URL** for this purpose: a direct `.zip` URL was downloaded and extracted before the chart was opened. The Companion Editor therefore preserves the complete ZIP hierarchy and edits the chart in place inside that bundle workspace.

### Bundle safety

ZIP import is bounded and rejects unsafe archives:

- canonical-path check against ZIP path traversal / Zip Slip;
- entry-count limit;
- download-size limit;
- extracted-size limit;
- unique `main.adofai` preferred; otherwise exactly one `.adofai` is required.

A URL-imported level remains inside the Companion app's private workspace. A locally opened ZIP is repackaged and synchronized back to that selected SAF document after a successful chart save.

## Editor

The Companion Editor supports:

- **New** chart creation;
- **Open** local `.adofai`, `.zip`, or `.adozip` through Android's Storage Access Framework;
- **ZIP URL** for a direct HTTP/HTTPS level archive;
- **Save / Save As** for chart documents and in-place synchronization of opened ZIP bundles;
- **Chart** editing for `pathData` / `angleData`;
- **Settings** editing;
- **Events** editing for `actions` and `decorations`;
- **Raw** JSON editing as a future-compatible fallback;
- UTF-8 BOM, trailing-comma and raw-control-character compatibility handling;
- preservation of unknown root fields and unknown event payloads unless explicitly changed;
- dirty-document guards around close/open/handoff flows.

## Official-game handoff

The **공식 ADOFAI** action saves the current chart, keeps all sibling bundle assets in their original relative layout, repackages the workspace as a ZIP, and exposes that ZIP only on the device loopback interface (`127.0.0.1`). The user's level is not uploaded to an external server.

`OfficialGameBridge` verifies the installed target is exactly:

```text
package:     com.fizzd.connectedworlds
version:     3.3.1
versionCode: 300382
activity:    com.unity3d.player.UnityPlayerActivity
```

It explicitly launches the official Unity activity with the loopback ZIP URL as the Intent data (`application/zip`) and supplies the same URL through multiple URL-oriented extras. A temporary read-only `content://` URI for the extracted chart is attached as a fallback with a package-scoped read grant.

### Why this route is technically plausible

The exact 3.3.1 runtime still contains the custom-level gameplay pipeline: `scrController.LoadCustomLevel(path, id, fromBundle)` populates `GCS.customLevelPaths`, transitions to `scnGame`, and the level loader resolves sibling resources relative to the chart file's directory.

The missing part on current Android is a proven public external entry point into that internal loader. Explicitly launching `UnityPlayerActivity` with a ZIP URL recreates the historical URL-shaped input without modifying the official package, but **build/CI success does not prove that 3.3.1 consumes that launch URL**. A real-device run is required for that final boundary.

A successful official-app launch by itself must not be reported as successful custom-level gameplay.

## Build

Pinned build environment:

- JDK 17
- Android platform 35
- build-tools 35.0.0
- Gradle 8.9

Local build:

```bash
bash scripts/prepare-editor-harness.sh
gradle --no-daemon -p android/editor-harness :app:assembleDebug
```

GitHub Actions workflow:

```text
Build ADOFAI Companion Editor
```

Artifact:

```text
adofai-companion-editor-apk/
└─ ADOFAI-Companion-Editor.apk
```

## Compatibility tooling

`tools/adofai_compat.py` provides source-side chart diagnostics/normalization:

```bash
python3 tools/adofai_compat.py level.adofai
python3 tools/adofai_compat.py level.adofai --normalize normalized.adofai
```

It handles BOMs, trailing commas, raw control characters, `pathData`/`angleData`, event inventories, optional legacy path conversion, and preservation of unknown/future payloads.

## Binary policy

No proprietary ADOFAI APK, native game library, or game asset is committed here. The Companion Editor is built independently. The installed Play app remains the user's original installation.
