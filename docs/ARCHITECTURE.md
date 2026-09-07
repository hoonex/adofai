# Architecture

## Why the 2.4.0 APK is a reference, not the final runtime

A `.adofai` file is not just geometry. Modern charts can contain settings, actions and decorations whose runtime implementations were added after 2.4.0. Making an old JSON reader ignore those fields may stop a parse exception, but it cannot make a missing runtime event render, preview, edit or serialize correctly.

The modern compatibility boundary is therefore:

```text
modern chart file
    -> tolerant syntax handling / lossless document model
    -> Android-native editor shell
    -> atomic save / reopen
    -> current ADOFAI 3.3.x engine for preview
    -> real-device verification
```

not:

```text
modern chart
    -> delete fields until 2.4.0 accepts it
```

The latter silently corrupts levels.

## Why the editor shell is Android-native

Inspection of the current Android 3.3.1 package changed the design.

The IL2CPP metadata still contains current editor/chart classes and methods, but the serialized `scnEditor` scene itself is not packaged in the Android player. An independent reverse-engineering baseline for the exact same 3.3.1 runtime reaches the same scene-table conclusion: the current custom-level gameplay pipeline is present, while `scnEditor` / `scnCLS` are not.

Therefore the modern branch does **not** rely on making the game report `isUnityEditor=true` and then loading a hidden editor scene. The modern path is:

```text
ADOFAI 3.3.1 Android runtime
    + injected secondary DEX
        -> MobileEditorShell (Android Views + Android IME)
        -> file open / save / save-as
        -> geometry/settings/event/raw editing
    + injected arm64 libOctober.so
        -> file-dialog hooks
        -> game-thread preview request bridge
        -> current scrController.LoadCustomLevel path
```

The old HitMargin editor/gameplay hooks remain only as an older-runtime compatibility baseline. A 3.3.x runtime is detected through the modern `UnityFileDialog` surface and takes an ABI-safe path that returns before legacy hooks whose method signatures changed.

## Evidence-backed source baselines

### Android hook/file-dialog baseline

Pinned source:

`HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level@74bcc7a0d8c8be1267504e21e28a35e199b5d4eb`

Useful primitives retained from that MIT baseline include:

- BNM/IL2CPP runtime lookup and hooking;
- native/Java bridge infrastructure;
- injected secondary-DEX loading through `DexClassLoader`;
- Android raw-path file chooser support;
- older SFB file-dialog compatibility.

Local exact-identity transforms add:

- the missing `ADOBase.get_isMobile` installation for the older profile;
- open/save/folder bridge completion;
- modern Android Activity/storage handling;
- `UnityFileDialog.FileBrowser` support for current 3.3.x builds;
- the modern ABI-safe profile;
- `MobileEditorShell` launch and current-runtime Preview bridge.

### Exact 3.3.1 runtime baseline

Pinned source/docs reference:

`Harrot114514/ADOFAI-MobileLevelLoder@401b69f26b2f607181d00b07273bac6bb0524638`

This MIT project targets ADOFAI Android 3.3.1 / Unity 6000.3.10f1 and documents the current custom-level chain. `scripts/prepare-current-runtime-shell.sh` sparse-checks only the source/documentation portions needed as evidence/reference and fails if proprietary game assets, game libraries, sample media or prebuilt loader binaries appear in the prepared tree.

### Modern chart-format reference

Pinned source:

`adofaiex/ADOFAI-JS@1f66bfa8c4146853d239c80c47ed2168d9208d02`

`tools/adofai_compat.py` uses the modern event inventory for diagnostics only. Unknown events are preserved; they are never silently deleted.

## Mobile editor shell data model

The first vertical slice intentionally keeps the root chart as a `JSONObject` instead of deserializing it into a closed, old-version schema.

Structured tabs edit only the fields they own:

- **Chart**: `pathData` or `angleData`;
- **Settings**: arbitrary settings keys with JSON values;
- **Events**: raw `actions` and `decorations` objects;
- **Raw**: full-document fallback for fields and future event shapes not yet represented by structured controls.

Unknown root fields and unknown event members remain in memory and are serialized back unless the user explicitly replaces or deletes them.

The loader also tolerates common real-world chart syntax problems already covered by the compatibility tooling:

- UTF-8 BOM;
- trailing commas outside strings;
- raw JSON control characters inside strings.

## Save boundary

A successful Save writes to a sibling temporary file, flushes the writer, calls `fsync`, and then replaces the target with `android.system.Os.rename`.

This keeps the original file intact until a complete replacement has been written. Save As goes through the same final write path.

The current implementation still requires real-device checks for Android storage-policy edge cases and sibling asset access.

## Preview boundary

Preview must not call Unity scene/game functions from the Android UI thread.

`MobileEditorShell.nativeQueuePreview(path)` only places the requested path in a native queue. `MobileEditorBridge.cpp` drains that queue from a Unity game-thread input poll, then resolves the required current runtime names through BNM:

- `GCS.customLevelIndex`;
- `GCS.internalLevelName`;
- `GCS.customLevelId`;
- `GCS.sceneToLoad`;
- optional `GCS.loadCustomFromBundle`;
- `scrController.get_instance`;
- `scrController.LoadCustomLevel`.

If the required current runtime surface is absent, Preview fails closed rather than calling a guessed hard-coded RVA.

This is compile-verified but remains device-runtime-unverified.

## Split APK packaging boundary

The inspected 3.3.1 install is a split package, not a single monolithic APK. The branch therefore separates placement by owner split:

- base split: manifest, Unity activity bootstrap and injected DEX;
- arm64 split: `libOctober.so`;
- asset/other splits: preserved instead of decoded/rebuilt.

All output splits are signed with one local identity. Because that local identity differs from the Play signing key, the modded package cannot update the Play-installed package in place.

`scripts/build-from-installed-current.sh` now composes the safe local workflow:

```text
adb pull installed user-owned split set
    -> build current editor payload
    -> repack base + arm64 owners
    -> preserve other splits
    -> align/sign all output splits with one local key
    -> stop before uninstall/install
```

The script intentionally never removes or installs the game automatically.

## Verification matrix

Build/CI evidence can prove:

- source transforms apply to exact pinned inputs;
- Python/shell/unit tests pass;
- Java editor shell compiles to DEX;
- native current-runtime bridge compiles/links for arm64;
- payload artifacts are produced;
- split tooling syntax and component-level tests pass.

A release/runtime claim additionally requires real-device evidence for:

- complete split repack and install;
- Editor launcher appearing in the real 3.3.1 activity;
- launcher surviving relevant scene/activity transitions;
- open from shared storage;
- `pathData` and `angleData` editing;
- settings and modern/unknown event preservation;
- Save and Save As;
- close/reopen equivalence;
- sibling song/background/decor asset resolution;
- Preview through the current 3.3.1 engine;
- touch selection/scrolling;
- Android keyboard/IME;
- large chart behavior;
- storage permission edge cases;
- supported orientation/viewport behavior.

Until those on-device checks are completed, source/build success must not be described as editor runtime success.
