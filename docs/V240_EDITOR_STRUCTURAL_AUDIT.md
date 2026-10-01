# ADOFAI v2.4 Android editor structural audit

Status: **STRUCTURAL AUDIT ACTIVE — NO TILE FIX CLAIM**

This document replaces the narrow “guess one cause, add one probe” debugging model for the
phone-only v2.4 editor-selection defect. The goal is to reconstruct the editor as a complete
input/selection system, identify Android-hostile assumptions, and only then make a repair.

The healthy tablet remains the control. Do not modify it merely to collect comparison data.

## Evidence grades

Use these grades throughout this document:

- **V240-EXACT** — established from the authoritative v2.4 APK, its exact IL2CPP metadata /
  arm64 binary, or the pinned v2.4 runtime build.
- **DEVICE** — observed on the affected phone.
- **PINNED-UPSTREAM** — from
  `HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level@74bcc7a0d8c8be1267504e21e28a35e199b5d4eb`,
  which is the exact upstream baseline used by this repository.
- **CROSS-VERSION** — public ADOFAI source-derived documentation or mods from another game
  build. Useful to recover responsibilities and state-machine shape, but never sufficient
  for an ABI mutation by itself.
- **HYPOTHESIS** — structurally plausible but not yet demonstrated on this v2.4 phone.

## System boundary

This is not a normal Android-native editor. It is a Unity/IL2CPP editor whose selection path
was designed around desktop-style mouse semantics, running inside an Android player with
touch-to-legacy-mouse emulation, Android window/inset behavior, and injected native/DEX
compatibility code.

Historical context also matters: 7th Beat Games publicly described the mobile editor as
unsupported in 2019. This is context, not proof of current v2.4 behavior:
https://steamcommunity.com/app/977950/discussions/0/1778262124937541640/?ctp=6

The pinned Android mod baseline itself is explicitly an Android NDK/IL2CPP hook project:
https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level/tree/74bcc7a0d8c8be1267504e21e28a35e199b5d4eb

## Exact v2.4 selection surface

V240-EXACT metadata/binary evidence:

- `scnEditor.HandleMouseActions`: RVA `0x22E3EB0`
- `scnEditor.ObjectsAtMouse`: token `0x060006F2`, RVA `0x22E8DF4`
- `scnEditor.SelectFloor`: RVA `0x22E7DD0`
- `FloorMesh.GenerateCollider`: RVA `0x10835F8`
- `scrFloor.GenerateCollider`: RVA `0x0C61EFC`
- exact `Physics2D.RaycastAll(Vector2, Vector2, float, int)`: RVA `0x1B9FF44`

The exact device metadata also exposes the following `scnEditor` state/methods:

- `ObjectsAtMouse`, `GizmoAtMouse`, `SmartObjectSelect`, `SelectFloor`
- `DragCamera`, `DragTilesStart`, `DragTiles`
- `mousePosition0`, `cameraPositionAtDragStart`
- `camera`, `floorLayerMask`
- `previouslyFoundObjects`, `foundObjects`, `selectedObjectIndexOfBunch`
- `decorationWasSelected`, `hoveringFindFloorPanel`
- `selectedFloors`, `selectedDecorations`, `multiSelectPoint`

This proves that floor selection is not a stateless
`touch -> raycast -> SelectFloor` path. It shares state with dragging, camera movement,
decoration/gizmo selection, overlap cycling, previous-frame hit sets, and UI state.

## Cross-version reconstruction of editor responsibilities

Public source-derived documentation for `scnEditor` describes
`PointerDownObjectType = Decoration / Floor / Gizmo / None`, persistent floor/decoration
selection state, and a large editor state machine:
https://github.com/memsys-lizi/7thRhythmDocuments/blob/94af62b6be641f9ddc10be08c1aa5ac867da1ff3/adofai/docs/api/core/scnEditor.md

Public mods provide two especially useful behavioral anchors:

1. Iridium replaces `ObjectsAtMouse` with a spatially bounded implementation that converts
   `Input.mousePosition` through the editor camera, generates/enables floor colliders, and
   performs point-overlap queries. Its comments explicitly preserve the original
   null-on-empty contract so callers can deselect:
   https://github.com/adofaiex/Iridium/blob/12e7c92671c884f98ad4a1f10d9359c923098096/v3/Patches/SceneOptimizationPatches.cs

2. Sapphire patches `HandleMouseActions` and documents a branch where right-button state,
   current selection and editor mode control free-angle dragging:
   https://github.com/PrismMods/Sapphire/blob/c8987c33d941307c2664baddf7d216b6e9bbd2bc/Sapphire/Patches/Patches.cs

Sapphire also intentionally resolves some editor click meaning on the following frame because
the result cannot be known until the game has processed the click. Its editor helpers use
`editor.camera.ScreenToWorldPoint(Input.mousePosition)` and include a world-distance floor
fallback:
https://github.com/PrismMods/Sapphire/blob/c8987c33d941307c2664baddf7d216b6e9bbd2bc/Sapphire/Editor/Chrome/EditorToolbar.cs

These are CROSS-VERSION references. They define useful failure classes, not exact v2.4 patch
targets.

## Pinned upstream changes that alter the platform model

PINNED-UPSTREAM `Main.cpp` installs several global hooks. Two matter structurally:

- when `enableLoadLevel` is enabled, `ADOBase.get_isUnityEditor` is forced to return true;
- when `enableCustomUIHitTest` is enabled, `scrController.IsScreenPointInsideUIElements`
  is replaced with an EventSystem `PointerEventData + RaycastAll` implementation.

Source:
https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level/blob/74bcc7a0d8c8be1267504e21e28a35e199b5d4eb/app/src/main/jni/Main.cpp

The matching `Hooks.cpp` shows that `get_isUnityEditor` is not a narrow file-picker return;
it is a global property replacement. The project defines an `IsMobile()` wrapper but this
pinned commit does not install it as a hook:
https://github.com/HitMargin/A-Dance-of-Fire-and-Ice-Mobile---Load-Custom-Level/blob/74bcc7a0d8c8be1267504e21e28a35e199b5d4eb/app/src/main/jni/Hooks.cpp

Current v2.4 hot-runtime evidence says the broad custom UI-hit hook is disabled on the active
tile path, so it is not the leading explanation for the current floor-selection defect.
The forced `isUnityEditor` state remains a structural audit item because it can activate
desktop/editor-only branches outside the file-picker feature.

## Unity legacy mobile-input contract

Unity 2021.3 documents `Input.mousePosition` as pixel coordinates in the current
`Screen.width / Screen.height` space, with the bottom-left as the origin:
https://docs.unity3d.com/2021.3/Documentation/ScriptReference/Input-mousePosition.html

Unity's legacy mobile input also enables touch-to-mouse simulation by default. Multiple
concurrent touches can be translated into different mouse buttons:
https://docs.unity3d.com/2021.3/Documentation/ScriptReference/Input-simulateMouseWithTouches.html

The mobile-input manual explicitly warns that simulated mouse input is materially different
from native touch input and recommends moving to touch input rather than relying on mouse
simulation for a finished mobile interaction model:
https://docs.unity3d.com/2021.3/Documentation/Manual/MobileInput.html

This makes three structural assumptions explicit:

- a floor tap depends on the legacy mouse state machine receiving the intended touch phase;
- a second concurrent touch can become a different mouse-button semantic rather than merely
  another touch point;
- the screen-pixel space used by the editor must stay synchronized with the Android surface
  and editor-camera viewport.

These are platform contracts, not proof that any one of them is currently broken.

## Android window / viewport boundary

The current compatibility layer changes Android window behavior after Unity activity
creation, including display-cutout policy, soft-input behavior, gesture exclusion and
repeated viewport recovery. The hot runtime also reasserts viewport normalization after
startup and lifecycle/layout changes.

That creates a boundary that must be treated as one coordinate system, not separate fixes:

`Android window/surface -> Unity Screen dimensions -> legacy Input.mousePosition ->
Camera.pixelRect / ScreenToWorldPoint -> Physics2D query`

A mismatch anywhere in this chain can make ordinary Unity UI appear correct while
world-space editor picking is wrong. UI and world picking do not necessarily consume the
same coordinate transform.

This risk is stronger on phones than tablets because cutouts, navigation modes, aspect
ratios, density, IME resize and edge gestures differ.

However, the current phone failure survived the repository's full-width SHORT_EDGES policy,
its delayed reassertions, and the later activity/lifecycle/layout viewport guard. Therefore
"the Android cutout shrank the Unity surface" is not sufficient as the current root cause.
A subtler disagreement between Input.mousePosition, Unity Screen dimensions, the editor
camera pixel rect, or cached editor coordinates remains open and must be measured as one
transaction rather than inferred from the window policy alone.

## Device-falsified hypotheses

Do not revisit these without new contradictory evidence.

### Missing touch-to-mouse edge

Earlier device evidence showed legacy mouse-down edges while floor selection still rarely
reached `SelectFloor`. The primary failure is not simply “Android never generates
GetMouseButtonDown”.

### Physics2D stale-transform synchronization

Runtime `3701b119a389bde4e0bcb0d503b78426869e6fb2` proved on the affected phone that:

- the exact `ObjectsAtMouse` and exact four-argument `RaycastAll` hooks installed;
- ABI guards passed;
- `Physics2D.SyncTransforms` resolved through the already-loaded libil2cpp owner;
- `ObjectsAtMouse`, `RaycastAll` and `SyncTransforms` executed;
- tile selection still failed.

Therefore “generate collider and immediately query stale Physics2D state” is not an active
repair theory.

R30 removes that mutation and observes the original query path only.

## Important counter contradiction

The r29 phone report recorded:

- `tileR21ObjectsCalls=60`
- `tileR21RaycastCalls=60`
- `tileR21SyncCalls=30`

The hook increments `ObjectsCalls` once per outer `ObjectsAtMouse` invocation and
`RaycastCalls` once per hooked exact RaycastAll invocation. Therefore it is not valid to
describe v2.4 as “every ObjectsAtMouse call always executes exactly two RaycastAll calls”.

The known callsites at `0x22E9400` and `0x22E9454` exist, but control flow is conditional.
The old prose interpretation was too strong. Any final reconstruction must explain the
60/60/30 device counts rather than hiding the mismatch.

This is a strong reason to reconstruct the whole `HandleMouseActions -> ObjectsAtMouse`
control flow instead of adding another isolated physics patch.

### Exact v2.4 call-graph refinement

A second pass over the authoritative arm64 `libil2cpp.so` resolves more of the exact
selection graph:

- `HandleMouseActions` directly calls `ObjectsAtMouse` at `0x22E480C` and
  `0x22E71C0`;
- both direct paths then call the function at `0x22E9910`, with the bool argument
  respectively `false` and `true`;
- `0x22E9910` itself calls `ObjectsAtMouse` at `0x22E994C`, compares the new object
  array with a cached previous array when cycling is allowed, maintains a modulo selection
  index, and returns one object from the hit set. Together with the exact metadata ABI/name,
  this identifies `scnEditor.SmartObjectSelect(bool)` at RVA `0x22E9910`;
- the function at `0x22EA828` reads the current pointer/camera state, calls the exact
  four-argument `Physics2D.RaycastAll` at `0x22EA914`, filters the hit to the
  `TransformGizmo` type and returns it. Together with the exact metadata ABI/name, this
  identifies `scnEditor.GizmoAtMouse()` at RVA `0x22EA828`;
- `HandleMouseActions` calls that `GizmoAtMouse` path at `0x22E7134` and stores its
  result in editor state;
- `HandleMouseActions` reaches `SelectFloor` through three distinct callsites:
  `0x22E564C`, `0x22E6AA0` and `0x22E7838`. All three pass the same second bool
  argument value `true`;
- inside `ObjectsAtMouse`, the exact RaycastAll callsites remain `0x22E9400` and
  `0x22E9454`. They derive layer masks from two separate editor integer fields
  (`self+0x824` and `self+0x820`) and merge the returned hit arrays afterward.

This changes the interpretation of the old counters. One logical pointer-selection path can
execute a direct `ObjectsAtMouse` and then another `ObjectsAtMouse` inside
`SmartObjectSelect`; these are sequential top-level calls, not recursive invocations of the
same active wrapper.

R31 counts every completed outer `ObjectsAtMouse` call in a pointer transaction, but its
`objectsLastCount` and `ray1/ray2` fields are overwritten by each successive call. Therefore
a transaction with `objCalls > 1` and a final zero count does **not** prove that every query
in that transaction was empty. The offline analyzer deliberately reports that case as
`WORLD_QUERY_AMBIGUOUS` rather than manufacturing a zero-hit conclusion. A positive final
snapshot remains valid positive evidence.

This exact nested-at-the-state-machine-level structure also explains why raw global
Objects/Raycast totals cannot be interpreted as a fixed call ratio without reconstructing
which `HandleMouseActions` branch ran.

## Structural failure map

### 1. Tap is classified as a drag or camera operation

Priority: **HIGH**

Why it fits the architecture:

- exact v2.4 exposes `mousePosition0`, `cameraPositionAtDragStart`, `DragCamera`,
  `DragTilesStart`, and `DragTiles`;
- public HandleMouseActions patches show selection and angle-dragging share the same state
  machine;
- Android touch coordinates can move by a few physical pixels even during a human “tap”;
- a threshold designed around desktop mouse motion can behave differently with phone density,
  frame pacing or touch emulation.

Expected sibling bugs:

- camera nudges when tapping;
- tile angle/drag mode starts unexpectedly;
- long press behaves differently from quick tap;
- failure rate changes with screen density or touch sampling rate.

Do not patch thresholds until the exact v2.4 branch and units are recovered.

### 2. Screen/surface/camera coordinate spaces diverge

Priority: **HIGH**

Why it fits:

- world picking is camera-relative;
- public editor code uses `camera.ScreenToWorldPoint(Input.mousePosition)`;
- the Android compatibility layer changes window/inset behavior after Unity startup;
- ordinary UI can still work if EventSystem/UI canvas coordinates are correct while the
  editor camera pixel rect or world conversion is not.

Expected sibling bugs:

- hit offset changes across the screen;
- center works better than edges or vice versa;
- portrait/cutout/navigation mode changes behavior;
- camera zoom changes hit error;
- gizmo hover/drag is offset similarly.

R30’s original raycast origins are useful evidence for this class, but a proper audit also
needs the screen position and camera pixel rect from the same click transaction.

### 3. Conditional `ObjectsAtMouse` path skips physics work

Priority: **HIGH**

Why it fits:

- exact binary has known RaycastAll callsites;
- real-device totals prove they are not executed in a simple fixed 2-per-call pattern;
- collider generation, floor candidate pruning, camera/view bounds or mode checks can
  short-circuit before those calls.

Expected sibling bugs:

- some tiles/zoom levels/regions work while others do not;
- selection success depends on visible-floor density;
- first click after camera movement differs from later clicks.

The exact branch that explains 60 Objects calls / 60 Raycast calls / 30 first-sync calls must
be recovered before another world-hit mutation.

### 4. Raycast succeeds but object filtering/cycling rejects the floor

Priority: **HIGH**

Why it fits:

- `previouslyFoundObjects`, `foundObjects`, and
  `selectedObjectIndexOfBunch` are exact v2.4 fields;
- floor, decoration and gizmo objects can overlap;
- selection has explicit object-type ownership and multi-selection state;
- repeated clicks can intentionally cycle overlapping objects.

Expected sibling bugs:

- repeated taps select a floor only occasionally;
- dense decorations make floor selection worse;
- selection order changes after camera movement;
- tapping the same location repeatedly cycles unexpected objects.

This class can produce the user-visible symptom even when Physics2D is completely correct.

### 5. Pointer-down ownership and pointer-up state span different frames

Priority: **HIGH / MEDIUM**

The editor’s pointer owner can be Floor, Decoration, Gizmo or None. Public mods also show
that some selection meaning is only known after the game has processed a click. Android
legacy mouse synthesis can place touch, mouse-down, held and mouse-up observations on
different update boundaries than a desktop mouse.

Expected sibling bugs:

- down begins on a floor but up resolves as None or drag;
- very short and slightly longer taps behave differently;
- selection succeeds at frame-rate-dependent rates;
- a second tap works more often than the first.

A future trace must correlate one pointer transaction across frames instead of reporting
unrelated global counters.

### 6. Editor UI/focus gates suppress world selection

Priority: **MEDIUM**

The editor contains popups, inspector controls, input fields, find panels, hover state and
EventSystem selection. `userIsEditingAnInputField` and `hoveringFindFloorPanel` are
examples of gates that can legitimately suppress world input.

The earlier broad UI-hit probe did not support “all taps are blocked by EventSystem”, so this
is narrower: stale focus, a full-screen invisible UI raycast target, or a specific editor
gate could still win only inside `HandleMouseActions`.

Expected sibling bugs:

- tapping after closing a popup leaves the editor unresponsive;
- keyboard/IME use changes selection behavior;
- inspector open/closed state changes hit reliability.

### 7. Forced `isUnityEditor=true` creates a mixed platform state

Priority: **MEDIUM / OPEN**

The pinned upstream forces `ADOBase.isUnityEditor` globally to expose load-level behavior.
The Android process is not the Unity Editor, so any unrelated game code branching on this
property can receive a false platform model.

This is not yet tied to `HandleMouseActions`; no repair should simply turn it off because it
may be needed by upstream functionality. Audit all exact v2.4 callsites first.

### 8. Hover-only assumptions do not map cleanly to touch

Priority: **MEDIUM for gizmos/decorations, LOW for plain floor selection**

Current source-derived docs show editor gizmos using pointer/mouse enter/exit hover state.
Touch has no persistent hover equivalent. On phones this can break:

- gizmo ownership;
- decoration scaling/rotation handles;
- cursor-edge guards;
- drag start state.

It is likely a separate Android editor compatibility class even if not the floor bug.

### 9. IME / resize / focus lifecycle invalidates cached coordinates

Priority: **MEDIUM for editor-wide robustness**

The compatibility layer uses `SOFT_INPUT_ADJUST_RESIZE`. Text editing can resize the Unity
surface, change focus, then restore it. Any editor state caching screen-space positions across
that transition can become stale.

Expected sibling bugs:

- after editing a text field, world picking is offset or dead until another layout event;
- camera drag origin jumps;
- popups/input fields leave stale focus.

### 10. Physics synchronization

Priority: **LOW / DEVICE-FALSIFIED for the current symptom**

Keep as historical evidence only.

## Other Android-editor bug classes worth auditing

Even if unrelated to the current tile bug, the same architecture can plausibly cause:

- accidental camera panning from touch jitter;
- decoration/gizmo drag beginning on tap;
- hover-dependent handles that cannot be reliably acquired by touch;
- multi-touch producing legacy mouse or modifier-like state the desktop editor never expected;
- right/middle mouse editor features having no mobile semantic;
- keyboard-shortcut-only editor operations with no touch replacement;
- IME focus swallowing editor input after a text field closes;
- cutout/navigation-bar changes shifting world-space picking;
- lifecycle resume restoring the Unity surface before editor camera state is coherent;
- selection cycling becoming unstable when several floor/deco/gizmo colliders overlap;
- full-screen transparent UI elements blocking only some input phases;
- save/open flows depending on desktop-path semantics rather than Android SAF;
- editor state saved/restored across play-preview with stale selected-object references.

These are audit targets, not claims that all are currently broken.

## Calibration is a separate failure class

The repeating calibration defect should not be mixed into the editor-picking analysis.

Exact v2.4 analysis found that `scrConductor.SaveCurrentPreset` updates the in-memory preset
collection without scheduling the normal Persistence save path. R22 hooks the exact
zero-argument SaveCurrentPreset, calls the original first, then schedules the game-owned
`Persistence.Save`.

The phone has proved the R22 hook installs, but the latest report had not exercised the
SaveCurrentPreset boundary yet. Keep its device-verification status separate.

## Current r30/r31 purpose

Runtime `ba5d72262ba8dff4c29e25f5e5abb4db8d5953dc` (r30) is intentionally
observational for tile picking. It restored the original raycast behavior and records the
original RaycastAll counts, final ObjectsAtMouse count, origins, distances and layer masks.
It is not a repair.

R31 is the next observation layer produced from this structural audit. It does **not** add
another tile mutation. It keeps the r30 original-query path and adds exact-ABI-guarded,
pass-through hooks for:

- `HandleMouseActions()`;
- `SelectFloor(scrFloor, bool)`;
- `SmartObjectSelect(bool)`;
- `GizmoAtMouse()`;
- `DragCamera(Vector3)`;
- `DragTilesStart()`;
- `DragTiles(Vector3)`.

The existing r30 ObjectsAtMouse/RaycastAll wrappers feed the same pointer transaction. Only
the latest eight transactions are retained in memory; there is no per-frame log stream.
A transaction starts on legacy left-mouse down and remains open through mouse-up plus one
additional HandleMouseActions pass so delayed selection decisions are not lost at a frame
boundary.

The first r31 phase intentionally avoids direct reads/writes of editor internals such as
`pointerDownObjectType`, `previouslyFoundObjects`, `selectedObjectIndexOfBunch` or
`mousePosition0`. Those fields are valuable, but method-boundary evidence is safer and
already separates several structural branches without relying on uncertain field ABI.

R31 must not be promoted as a repair.

## New debugging rule

Do not create another speculative mutation merely because one r30/r31 value is surprising.

Before the next behavior change:

1. use the correlated transaction to determine whether the failed tap enters world-hit,
   SmartObjectSelect/gizmo, SelectFloor, tile-drag, or camera-drag paths;
2. explain the conditional ObjectsAtMouse/Raycast relationship rather than assuming a fixed
   number of queries per call;
3. compare pointer displacement against accidental drag/camera activity;
4. compare screen-space tap coordinates with the world-query origins from the same
   transaction;
5. only if these boundaries remain ambiguous, add exact-ABI reads for the smallest necessary
   editor state fields;
6. only then choose the smallest repair boundary.

## R31 transaction fields

Each `editorTraceR31TxnN` record is one bounded pointer transaction and currently contains:

- sequence/state;
- start/end `Input.mousePosition`;
- Unity `Screen.width/height`;
- starting touch count;
- HandleMouseActions/held-frame counts;
- release observation and one post-release Handle frame;
- maximum pointer displacement;
- ObjectsAtMouse call count and last final object count;
- exact r30 RaycastAll counts/origins/layer masks;
- SmartObjectSelect call/non-null result counts;
- GizmoAtMouse call/non-null result counts;
- SelectFloor call/non-null/cameraJump values;
- DragCamera / DragTilesStart / DragTiles call counts.

This means one report can distinguish several broad cases without another mutation:

- no transaction at all -> legacy pointer transaction boundary failed;
- transaction + drag calls -> tap/drag arbitration becomes a leading cause;
- transaction + raycast/object hits + no SelectFloor -> post-hit selection arbitration becomes
  a leading cause;
- transaction + no world-hit calls -> an earlier HandleMouseActions/UI/mode branch won;
- SelectFloor called but visible selection still fails -> the defect is downstream of the
  selection decision.

The trace is bounded, read-only and tied to a click sequence. Global totals remain secondary
evidence.

### Offline r31 report analyzer

`tools/analyze_v240_r31_report.py` parses the copied compatibility report without changing the
runtime or game state:

```bash
python3 tools/analyze_v240_r31_report.py report.txt
cat report.txt | python3 tools/analyze_v240_r31_report.py -
cat report.txt | python3 tools/analyze_v240_r31_report.py --json
```

The analyzer first verifies the r31 runtime contract: revision, read-only policy, input ABI,
seven-method ABI/install masks, self-fuse readiness and recovery state. It refuses to present
a normal analysis status when those gates are incomplete.

Per transaction, the route names are evidence boundaries rather than root-cause claims:

- `SELECT_FLOOR_REACHED`: the exact SelectFloor boundary was entered;
- `TILE_DRAG_REACHED` / `CAMERA_DRAG_REACHED`: the pointer entered an explicit drag path;
- `WORLD_HIT_WITHOUT_SELECT`: a positive world hit was observed but SelectFloor was not;
- `WORLD_QUERY_NO_HIT`: a query ran and zero-hit evidence is complete enough to say so;
- `WORLD_QUERY_AMBIGUOUS`: query counters contain incomplete/sentinel evidence;
- `NON_FLOOR_EDITOR_PATH`: SmartObjectSelect/GizmoAtMouse ran without floor selection;
- `PRE_WORLD_QUERY_PATH`: no world-query/select/drag boundary was observed;
- `ACTIVE_TRANSACTION` / `SUPERSEDED_TRANSACTION`: the transaction is incomplete;
- `MIXED_EXPLICIT_PATH`: multiple explicit terminal paths ran, so the analyzer deliberately
  does not choose one.

The analyzer sorts the eight ring-buffer slots by transaction sequence, preserves malformed
records as warnings, and exposes the same result as JSON for future automated comparison.
It must not be used to infer that a route itself is defective; the route only identifies the
smallest observed boundary for the next evidence or repair decision.

## Current working model

The strongest current model is not “Physics2D is broken on this phone”.

It is:

> a desktop-style editor input/selection state machine is running on an Android touch/window
> environment, and the failure is most likely at a boundary where one pointer transaction is
> classified, transformed into world coordinates, or arbitrated among overlapping editor
> objects.

That model is broad enough to explain the phone/tablet difference without pretending the
root branch has already been identified. It also predicts multiple related Android-editor
bugs that should be addressed as one compatibility layer once the exact failing branch is
known.
