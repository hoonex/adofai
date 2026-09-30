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

The broader Android editor failure map and evidence ranking are maintained in
[`V240_EDITOR_STRUCTURAL_AUDIT.md`](V240_EDITOR_STRUCTURAL_AUDIT.md). Do not add another
tile mutation without reconciling it with that audit.

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

### Device result and current observation boundary

R28/R29 tested the same-frame Physics2D synchronization hypothesis without changing
selection or raycast arguments. On the affected phone, runtime 3701b119 loaded healthy
and proved all of the following on-device:

- the exact ObjectsAtMouse and four-argument RaycastAll hooks installed;
- the exact ABI guards passed;
- il2cpp_resolve_icall was obtained through the already-loaded libil2cpp owner;
- Physics2D.SyncTransforms resolved and executed;
- ObjectsAtMouse and RaycastAll were actually called during touch interaction.

Tile selection still failed. Therefore the SyncTransforms hypothesis is **device-falsified**
and is not an active repair.

R30 keeps the two already-proven hook sites as read-only observation boundaries and restores
the original Physics2D query behavior. It does not call SyncTransforms, change origin,
direction, distance, layer mask, return arrays, or SelectFloor. It records bounded counts
for each of the two original RaycastHit2D[] results and the final GameObject[] returned by
ObjectsAtMouse, together with the original query arguments.

Exact original disassembly also establishes that ObjectsAtMouse combines the two RaycastAll
arrays before filtering the combined hits into its final GameObject[] result. The next
device result can therefore distinguish a Physics2D query miss from post-raycast filtering
without another speculative mutation.

Input.touchCount remains telemetry only, not a correctness gate. The healthy tablet remains
the control and is not modified for comparison.

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

## Runtime delivery and graceful-exit safety

The cache channel requires bootstrap version 3 or newer. The current one-click patcher
embeds bootstrap v3, while older pre-updater patched game APKs cannot consume this channel
and must not be treated as current runtime evidence.

Bootstrap v3 deliberately creates boot.pending before native loading and normally clears
it after a 10-second health window. Audit found one false-positive case: two ordinary user
exits inside that 10-second window could be counted as two interrupted boots and quarantine
an otherwise healthy runtime.

The production repair keeps the 10-second crash-detection deadline, but also treats Android's
graceful UnityPlayerActivity stop/destroy lifecycle as positive health evidence:

- new patcher builds handle this directly inside V240RuntimeUpdater;
- the hot RuntimeEntry also invokes bootstrap v3's existing private markHealthy(File)
  fail-open through parent-first reflection, so already-installed bootstrap-v3 devices gain
  the protection without replacing their parent DEX;
- the hot shim never deletes boot.pending itself and never duplicates rollback counters;
  the parent updater remains the sole owner of health-marker and boot-failure state.

A native abort or process death before Android delivers a graceful lifecycle transition still
leaves boot.pending intact, so genuine startup-failure rollback remains active.

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
