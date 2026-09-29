# ADOFAI 2.4.0 Custom Android root-cause record

Status: **DEVICE_RUNTIME_UNVERIFIED**

This record pins the production reasoning for the phone-only editor tile-selection and
repeating startup-calibration defects. CI/build success proves only the static/package
contract. A real phone still has to validate the final runtime. The healthy tablet is a
control and must not be updated merely for comparison.

## Authoritative game identity

Authoritative original APK:

- V2.4.0 Custom.apk
- size: 370,092,054 bytes
- SHA-256: 630f519ae1ab3391aad95da90ebc296f4f0f8ae4ea41024ace7349d93926ef30

The analyzed patched APK contains the following original critical entries byte-identical
to the authoritative source:

- lib/arm64-v8a/libil2cpp.so:
  c86d7ff549eeef7ecef2c8471019f771e609b3ce7cccf3f775625531b43f3494
- lib/arm64-v8a/libunity.so:
  b18718a2452441d05d9a6eb39bd5cb8bb38603ac0fc740fcb04348bb8f36dfd0
- lib/arm64-v8a/libmain.so:
  8588e1be5a574f2a276696651713e39483904b245c56f91aec970c3170b710bc
- assets/bin/Data/Managed/Metadata/global-metadata.dat:
  26b6c4c711eb43815092925cad2fad99c9de12ba9665d880b7f3e6792823d6f9

The original player identifies as **Unity 2021.3.10f1**.

## Editor tile selection

### Original v2.4 path

Exact metadata/disassembly establishes:

- scnEditor.HandleMouseActions: RVA 0x22E3EB0
- scnEditor.ObjectsAtMouse: token 0x060006F2, RVA 0x22E8DF4
- scnEditor.SelectFloor: RVA 0x22E7DD0
- FloorMesh.GenerateCollider: RVA 0x10835F8
- scrFloor.GenerateCollider: RVA 0x0C61EFC
- Physics2D.RaycastAll(Vector2, Vector2, float, int):
  token 0x06000023, RVA 0x1B9FF44

Inside ObjectsAtMouse itself:

1. callsite 0x22E91BC calls FloorMesh.GenerateCollider;
2. 0x22E91D0 enables the generated behaviour/collider;
3. 0x22E9268 calls scrFloor.GenerateCollider;
4. 0x22E9400 calls the exact four-argument Physics2D.RaycastAll;
5. 0x22E9454 calls that same overload again.

There is no collider-state mutation between the two raycasts. The game therefore creates
or enables transient floor collider state and queries the Physics2D world immediately in
the same ObjectsAtMouse call.

Device logs already proved Android touch-to-legacy-mouse down edges are present while
SelectFloor is reached only rarely. That rules out the earlier missing-GetMouseButtonDown
model and places the defect after input delivery, in the world-hit path.

### Production repair

The active r28 policy keeps the original selection and coordinates intact. It scopes an
exact Physics2D.RaycastAll(Vector2, Vector2, float, int) hook to execution inside
scnEditor.ObjectsAtMouse and calls Unity's own Physics2D.SyncTransforms() once before
the first original raycast. The original origin, direction, distance, layer mask, result
array, and SelectFloor behavior are unchanged. The second raycast observes the same
synchronized world because no collider state changes between the two calls.

Input.touchCount is telemetry only, not a correctness gate. This matters because legacy
mouse emulation and touch lifetime can cross frame boundaries differently on Android.

The tablet being healthy is compatible with this root cause: the original code contains
a same-frame synchronization hazard whose observed stale/current physics state can depend
on runtime/frame timing. The repair removes that dependency at the exact query boundary;
it does not assume a device-specific coordinate scale or force a selection.

## Repeating startup calibration

### Startup decision chain

Exact v2.4 control flow:

1. ADOStartup.Startup calls Persistence.Load at callsite 0x1F1C820.
2. The same startup later calls LoadCalibration at 0x1F1C880.
3. LoadCalibration calls CalibrationPreset.LoadDefaults, then
   scrConductor.UpdateCurrentAudioOutput.
4. UpdateCurrentAudioOutput calls
   GetSuitablePresetForCurrentAudioOutput(false) and copies its full 32-byte
   CalibrationPreset into scrConductor.currentPreset.
5. The fallback preset path writes confident=false at 0x2185388.
6. scnSplash.GoToMenu reads currentPreset.confident
   (scrConductor statics + 0x28): false routes to ADOBase.GoToCalibration;
   true routes to GoToLevelSelect.

CalibrationPreset.confident is byte offset +24 in the 32-byte value type.

### Why the old confidence theory was wrong

Persistence.Load explicitly writes confident=true at 0x1169674 before it calls
CalibrationPreset.FromDict at callsite 0x11696B8. Therefore the fact that
CalibrationPreset.ToDict does not serialize confident is intentional and is not the
repeat-calibration root cause.

### Missing persistence boundary

scrCalibrationPlanet.PostSong is the only direct caller of
scrConductor.SaveCurrentPreset; its callsite is 0x0B8A3F0 and the target is
RVA 0x218576C.

The original SaveCurrentPreset updates/replaces/appends scrConductor.userPresets in
memory but does not call Persistence.Save or Persistence.WriteSaveToDisk.

The game's own disk path is:

Persistence.Save (RVA 0x115F748)
→ SaveCo(0.5s)
→ Persistence.WriteSaveToDisk (RVA 0x1169E64)
→ CalibrationPreset.ToDict
→ PlayerPrefsJson.SetList
→ PlayerPrefsJson.SaveAllFiles.

Without a subsequent unrelated save before process exit, a calibration can therefore
finish successfully while the matching confident user preset exists only in memory.
On the next launch Persistence.Load cannot restore that preset; suitable-preset
selection can fall back to the unconfident preset, and GoToMenu sends the user to
calibration again.

The tablet can legitimately avoid this branch if it already has a matching persisted
user preset. No device-health or bad-phone-setting assumption is required.

### Production repair

The active r22 hook is the exact zero-argument static
scrConductor.SaveCurrentPreset. It:

1. calls the original SaveCurrentPreset first;
2. calls the game's own zero-argument static Persistence.Save;
3. changes no offset, output identity, preset fields, or confidence byte.

It is ABI-guarded, fail-open, and self-fused. Raw PlayerPrefs keys and
SetInputOffset(0) are not part of the active repair.

## Safety and proof boundary

Production invariants:

- original libil2cpp.so, libunity.so, libmain.so, and global-metadata.dat remain
  byte-identical;
- no floor coordinate scaling and no forced SelectFloor;
- no calibration-value overwrite and no raw PlayerPrefs mutation;
- exact managed ABI checks precede native hooks;
- mutation hooks use fail-open/self-fuse markers;
- static tests, native arm64 build, DEX build, packaging, and Release publishing are CI
  proof only.

Until the affected phone passes normal tile selection and two-launch calibration
persistence, the correct status remains **DEVICE_RUNTIME_UNVERIFIED**.
