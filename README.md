# ADOFAI Modern Mobile Editor

This repository is being rebuilt as a source-first patch project for a user-supplied A Dance of Fire and Ice Android APK.

## Goal

The old 2.4.0 custom mobile editor is not a good base for modern charts. Newer `.adofai` files contain schema and event types that the 2.4.0 engine does not know how to decode or render. A parser shim alone cannot add engine features that do not exist in that binary.

The implementation therefore targets a newer mobile game engine and exposes/fixes its embedded editor rather than backporting years of editor/runtime behavior into 2.4.0.

Current evidence-backed Android hook baseline:

- HitMargin mobile custom-level project, pinned at commit `74bcc7a0d8c8be1267504e21e28a35e199b5d4eb`.
- That project documents support for ADOFAI mobile 2.8.3–2.10.1 and hooks the IL2CPP runtime to expose level loading/editor behavior.
- Modern `.adofai` compatibility is modeled against ADOFAI-JS pinned at `1f66bfa8c4146853d239c80c47ed2168d9208d02`.

Exact pins are stored in `upstream.lock.json`.

## What is implemented now

### Modern map compatibility tooling

`tools/adofai_compat.py` is a lossless-first parser/normalizer for `.adofai` files. It:

- strips UTF-8 BOM safely;
- accepts trailing commas;
- repairs raw control characters embedded in strings;
- accepts both `pathData` and `angleData`;
- converts legacy `pathData` to `angleData` on request, including relative path symbols;
- preserves unknown top-level settings and event payload fields instead of deleting them;
- reports action/decorations event types and unknown modern event types;
- writes normalized UTF-8 JSON without intentionally downgrading the chart.

Examples:

```bash
python3 tools/adofai_compat.py level.adofai
python3 tools/adofai_compat.py level.adofai --normalize normalized.adofai
python3 tools/adofai_compat.py level.adofai --normalize normalized.adofai --convert-path-data
```

### Reproducible modern hook baseline

`scripts/prepare-upstream.sh` resolves the exact pinned HitMargin source into `.work/hitmargin-mobile-mod`. The repository never silently follows upstream `HEAD`.

## Planned runtime layers

The patch is intentionally split into separate owners so one workaround does not hide another bug:

1. **APK identity / IL2CPP discovery** — resolve exact game version, ABI, Unity/IL2CPP metadata and hook signatures.
2. **Editor exposure** — expose the current engine's editor scene and editor-only controls without globally pretending every scene is the Unity editor.
3. **Android file I/O** — reliable open/save/folder selection under scoped storage, with a filesystem path the Unity/Mono code can actually use.
4. **Map compatibility** — tolerate valid modern chart syntax without deleting unrecognized data.
5. **Mobile input/UI fixes** — touch hit testing, keyboard/IME, viewport/safe-area, scrolling and selection behavior.
6. **Verification** — fixture charts plus real-device editor open/edit/save/reopen tests.

## Binary policy

This repository does **not** commit or redistribute proprietary ADOFAI APK/game assets. Build/patch tooling operates on an APK supplied by the user.

The uploaded `V2.4.0 Custom.apk` remains useful as a behavioral/reference binary, but it is not the target engine for full modern map support.
