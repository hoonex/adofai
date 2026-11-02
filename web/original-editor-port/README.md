# ADOFAI original PC `scnEditor` browser port — working baseline

This project **does not** claim to run the game editor in browser. It starts from the **actual original PC files** supplied by the user, not from a hand-designed clone.

## What runs

- Fetches exact ZIP byte ranges for `level5` and `Assembly-CSharp.dll` **on demand** from the user's existing read-only Drive ZIP.
- Verifies the extracted PC bytes by SHA-256 before parsing.
- Reads the genuine Unity 6000.3.10f1 `scnEditor` object hierarchy, `RectTransform` hierarchy/position/anchors, and GameObject metadata with UnityPy.
- Reads genuine `scnEditor`, `scrFloor`, `FloorMesh` **method names** from the PC Mono assembly with dnfile.
- Exposes *metadata only* to a browser inspector at `/api/scene` and `/api/scene?view=methods`; doesn't publish original game files or the C# method bodies.

## Verified offline against actual supplied PC game

- `level5`: 602,692 bytes, SHA256 `45cffa740bd852b8ad9821eba5984f8c93b9016123a7ca1b0c645e36ef8d4800`.
- `Assembly-CSharp.dll`: 2,785,280 bytes, SHA256 `0d1524dabf0f8c67aa58ba1992d6ed858522b9094490ebbb00e5e8aeff7f40cd`.
- 3,727 serialized objects / 867 GameObjects / 1,220 MonoBehaviours.
- 369 `scnEditor` methods / 55 `scrFloor` methods / 51 `FloorMesh` methods.

## What is still missing

The original PC game was compiled to Windows native Unity player + managed Mono C# DLL. Those binaries cannot be executed as a Unity WebGL game by copying files. A real port needs a browser-compatible runtime and implementations of the PC editor's C# logic, Unity UI components, shaders/materials/textures/audio and chart serialization. The current inspector is an **authentic source-scene import/analysis step**, not the finished interactive editor.

Do not run the unofficial archive's EXE, BAT or other bundled launchers. The source archive, its game assets and DLL are not republished in this repository or deployment.

## Source

Source is the user's existing PC ZIP `A-Dance-Of-Fire-And-Ice-AnkerGames.zip`. All matching bytes are fetched from that file read-only. If Drive access changes, this development site will stop loading and no source files will be exposed.


## Browser chart editing layer (2026-11-02)

The original serialized `scnEditor` object hierarchy now runs beside a **browser-side chart editing layer**. This is source-scene-linked reimplementation, **not Unity C# methods executing in WebAssembly**.

- UI handlers are tied to GameObject IDs read from the original scene: `buttonOpen`, `buttonSave`, `buttonSaveAs`, `AddTile`, `settingsPanel`, `levelEventsPanel`.
- User-selected `.adofai` and JSON files are handled fully in the browser, never uploaded to the API.
- Parse `pathData` and `angleData`; preserve original char paths until the user explicitly modifies tile geometry, then normalize to explicit `angleData`. Unknown directions fail closed, not guessed.
- Edit selected tile direction, insert/delete tile, and shift matching `actions` and `decorations` floor IDs. Removing a tile with events asks confirmation.
- Edit JSON of each selected event, settings and raw chart; Save/Save As downloads `.adofai`. Unknown chart and event keys remain in the document unless explicitly modified.
- Undo/redo of document mutations. Render simplified tile positions; not full original Unity geometry or gameplay.
- Test with `node web/original-editor-port/tests/chart-port.test.cjs` and `node --check web/original-editor-port/chart-port.js`.

**Limits:** Floor positions ignore `PositionTrack` and other Unity rendering events; `angleData` and special path symbols get a conservative geometric preview. The web browser is *not* a replacement for the proprietary Unity player. This work does not reproduce the actual Unity C# runtime or distribute original textures/audio.

