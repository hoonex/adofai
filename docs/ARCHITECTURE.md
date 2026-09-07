# Architecture

## Why the 2.4.0 APK is a reference, not the final runtime

A `.adofai` file is not just geometry. Modern charts can contain settings, actions and decorations whose runtime implementations were added after 2.4.0. Making an old JSON reader ignore those fields may stop a parse exception, but it cannot make a missing runtime event render, preview, edit or serialize correctly.

The compatibility boundary is therefore:

```text
modern chart file
    -> tolerant syntax parser / diagnostics
    -> current-enough ADOFAI engine
    -> editor scene exposure
    -> Android-specific input/storage fixes
    -> save/reopen verification
```

not:

```text
modern chart
    -> delete fields until 2.4.0 accepts it
```

The latter silently corrupts levels.

## Evidence-backed upstream baseline

The pinned HitMargin source already provides useful Android/IL2CPP primitives:

- `ADOBase.get_isUnityEditor` override for exposing load/editor behavior;
- `SFB.StandaloneFileBrowser.OpenFilePanel` replacement;
- Java file chooser bridge loaded from `classes2.dex`;
- custom UI hit testing through Unity `EventSystem.RaycastAll`;
- hooks for pause/menu/mobile-specific behavior.

This is a better base than rediscovering every hook from the 2.4.0 binary.

## Known baseline risks to fix rather than inherit blindly

1. The upstream code defines an `ADOBase.get_isMobile` replacement but the pinned `Main.cpp` does not install that hook. Any intended scene-specific desktop/mobile editor behavior is therefore not proven active.
2. The custom Java chooser starts from `Environment.getExternalStorageDirectory()` and works with raw filesystem paths. On modern Android, scoped-storage behavior and permission state must be tested explicitly instead of assuming every directory is readable/writable.
3. `get_isUnityEditor` is forced globally by the baseline. Global editor identity can enable unrelated desktop/editor-only branches. The final patch should narrow overrides to the scenes/features that actually need them.
4. File-open success is not enough. A chart can reference sibling audio/images/fonts/video. The import path must preserve relative assets or the editor will appear to load while media is missing.
5. An event name being parseable is not proof that its current runtime behavior is supported. Modern event fixtures must be opened, edited, saved and reopened on-device.

## Modern format inventory

The pinned ADOFAI-JS reference currently models 57 event types, including newer families such as `SetFilterAdvanced`, `SetConditionalEvents`, `FreeRoam*`, `SetObject`, particles, and `SetInputEvent`.

`tools/adofai_compat.py` uses that inventory only for diagnostics. Unknown events are preserved; they are never dropped automatically.

## Verification matrix

A release claim requires separate evidence for:

- APK repack/install success;
- editor scene entry;
- touch selection/dragging/scrolling;
- hardware keyboard where supported;
- open from shared storage;
- save to writable storage;
- reopen saved map;
- relative song/background/decor assets;
- legacy `pathData` chart;
- modern `angleData` chart;
- actions and decorations including at least one post-2.4 feature;
- large map load without UI deadlock;
- portrait/landscape behavior actually supported by the product.

Until real-device evidence exists, those states remain unverified even if source compiles.
