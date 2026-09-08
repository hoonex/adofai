# ADOFAI Companion Editor

Non-root Android companion editor for user-authored `.adofai` charts and ZIP level bundles. The canonical product keeps the official Google Play ADOFAI installation untouched and runs the editor as a separate app (`dev.hoonex.adofai.companion`). This repository does **not** contain or redistribute the proprietary ADOFAI APK or game assets.

## Canonical architecture

```text
ADOFAI Companion Editor
    ├─ Android Storage Access Framework (Open / Save / Save As)
    ├─ local .adofai / .zip / .adozip input
    ├─ ZIP URL import
    ├─ loss-preserving .adofai editor
    ├─ app-private bundle workspace
    ├─ sibling audio/image/decor asset preservation
    ├─ loopback-only ZIP URL publisher
    ├─ read-only, grant-scoped content:// chart fallback
    └─ explicit handoff to installed official ADOFAI 3.3.1

Official Google Play ADOFAI
    └─ com.fizzd.connectedworlds / 3.3.1 / versionCode 300382
       remains installed and unmodified
```

The canonical path does **not** root the device, use Magisk/Zygisk, patch or resign the Play APK, replace its signing identity, bypass licensing, or bundle a clean-room gameplay player.

Legacy patcher/Zygisk experiments remain repository history/reference only. Their workflows are manual-only and are not part of the normal Companion Editor build.

## ZIP bundle model

Custom ADOFAI levels commonly consist of more than one file. A representative package is:

```text
Level Name/
├─ main.adofai
├─ song.ogg
├─ background.jpg
└─ decoration.png
```

The chart references sibling files by relative filename, so the Companion treats the complete directory hierarchy as the authoritative level bundle rather than flattening the chart into a standalone JSON document.

Supported inputs:

- local `.adofai`;
- local `.zip` / `.adozip`;
- a direct ZIP URL through **ZIP URL**.

ZIP import prefers a unique `main.adofai`. If no `main.adofai` exists, exactly one `.adofai` file must be present. The importer rejects canonical-path traversal, excessive entry counts, oversized downloads, and oversized expanded archives.

When a bundled chart is saved, the edited chart remains in the same workspace as its sibling assets. A locally opened ZIP is repackaged and synchronized back to its selected Android document.

## Editor

The Companion Editor supports:

- **New** chart creation;
- **Open** through Android's Storage Access Framework;
- **ZIP URL** import;
- **Save / Save As**;
- **Chart** editing for `pathData` / `angleData`;
- **Settings** editing;
- **Events** editing for `actions` and `decorations`;
- **Raw** JSON editing as a future-compatible fallback;
- UTF-8 BOM, trailing-comma and raw-control-character compatibility handling;
- preservation of unknown root fields and unknown event payloads unless explicitly changed;
- dirty-document guards around close/open/handoff flows.

## Official-game handoff

The **공식 ADOFAI** action reproduces the historical ZIP-URL-shaped input as closely as possible without modifying the official game:

1. save/synchronize the current chart and bundle;
2. repackage the full level directory, including sibling assets;
3. publish the ZIP only on `http://127.0.0.1:<ephemeral-port>/bundle/<token>/level.zip`;
4. verify the installed target is exactly `com.fizzd.connectedworlds` 3.3.1 / versionCode 300382;
5. explicitly launch `com.unity3d.player.UnityPlayerActivity` with the ZIP URL as `ACTION_VIEW` data and `application/zip`;
6. also supply URL-oriented extras and a read-only package-granted `content://` URI for the extracted chart as a fallback.

Nothing is uploaded to an external server by this path.

### Runtime consumption diagnostic

Launching the official Activity is not considered proof that the game consumed the level. The loopback server therefore records requests for the exact published ZIP, including:

- `HEAD` count;
- `GET` count;
- response-body bytes served;
- the most recent `User-Agent` when supplied.

When the user returns from official ADOFAI to the Companion, the result is surfaced in the editor and as a toast.

Interpretation:

```text
GET > 0
  => strong evidence that the official process or a component it invoked
     actually requested the ZIP URL body.

HEAD > 0, GET = 0
  => the URL was touched, but a ZIP-body download was not demonstrated.

HEAD = 0, GET = 0
  => no loopback request was observed. This alone does not distinguish
     "the game ignored the URL" from a target-side cleartext-HTTP/network
     policy that rejected http://127.0.0.1 before a request reached the server.
```

If the zero-request case occurs on a physical device, the next bounded experiment is a direct HTTPS ZIP URL using the same explicit official-Activity handoff. That removes the localhost-cleartext ambiguity before considering any different architecture.

## Runtime evidence boundary

Reverse analysis of the exact 3.3.1 runtime confirms that the custom gameplay pipeline still exists internally: `scrController.LoadCustomLevel(path, id, fromBundle)` populates custom-level state, transitions to `scnGame`, and the custom-level path resolves sibling assets relative to the chart directory.

What remains device-runtime-unverified is the final external entry boundary: whether the unmodified Android 3.3.1 Activity consumes the supplied ZIP URL and routes it into that internal loader. A successful app launch alone must never be described as official preview success.

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
