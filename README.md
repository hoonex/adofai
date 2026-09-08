# ADOFAI Companion Editor

Non-root Android companion editor for user-authored `.adofai` charts. The canonical product keeps the official Google Play ADOFAI installation untouched and runs the editor as a separate app (`dev.hoonex.adofai.companion`). This repository does **not** contain or redistribute the proprietary ADOFAI APK or game assets.

## Canonical architecture

```text
ADOFAI Companion Editor
    ├─ Android Storage Access Framework (Open / Save / Save As)
    ├─ loss-preserving .adofai editor
    ├─ app-private working copy
    ├─ read-only, grant-scoped content:// chart provider
    └─ explicit handoff to the installed official ADOFAI 3.3.1

Official Google Play ADOFAI
    └─ com.fizzd.connectedworlds / 3.3.1 / versionCode 300382
       remains installed and unmodified
```

The canonical path does **not** root the device, use Magisk/Zygisk, patch or resign the Play APK, replace its signing identity, bypass licensing, or bundle a clean-room gameplay player.

Legacy patcher/Zygisk experiments remain repository history/reference only. Their workflows are manual-only and are not part of the normal Companion Editor build.

## Editor

The Companion Editor supports:

- **New** chart creation;
- **Open** through Android's Storage Access Framework;
- **Save / Save As** back to the selected Android document;
- **Chart** editing for `pathData` / `angleData`;
- **Settings** editing;
- **Events** editing for `actions` and `decorations`;
- **Raw** JSON editing as a future-compatible fallback;
- UTF-8 BOM, trailing-comma and raw-control-character compatibility handling;
- preservation of unknown root fields and unknown event payloads unless explicitly changed;
- dirty-document guards around close/open/handoff flows.

The editor works on an app-private mirror so its atomic-save semantics remain independent of the external document provider. Successful saves are synchronized back to the selected SAF document.

## Official-game handoff

The **공식 ADOFAI** action first saves/synchronizes the current chart, then creates a temporary read-only `content://` URI from `OfficialChartProvider`. Read permission is granted only to `com.fizzd.connectedworlds`.

`OfficialGameBridge` verifies the installed target is exactly:

```text
package:     com.fizzd.connectedworlds
version:     3.3.1
versionCode: 300382
activity:    com.unity3d.player.UnityPlayerActivity
```

It then explicitly launches that exported official activity with the chart URI, `application/json`, a URI permission grant, `ClipData`, `EXTRA_STREAM`, and compatibility URI extras in one handoff attempt.

### Verification boundary

Real-device inventory from the exact Play build shows that ADOFAI 3.3.1 exposes `UnityPlayerActivity`, but does **not** advertise a normal public `ACTION_VIEW` or `ACTION_SEND` file-import handler for `.adofai` data. Therefore Android can launch the exported activity explicitly, but repository/CI evidence alone cannot prove that the unmodified game will consume the supplied chart URI and enter gameplay.

If the official game ignores the supplied Intent data, a separate non-root app cannot directly invoke its private Unity/IL2CPP level loader or write its private app data. Doing that would require changing the constraints (for example modifying/injecting into the game process), which is intentionally outside the canonical product.

The project must not label a mere successful app launch as “official preview success.”

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
