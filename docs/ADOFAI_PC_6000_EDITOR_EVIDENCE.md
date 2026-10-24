# PC Unity 6000 editor reference — static evidence (2026-10-25)

This document records **read-only** structural findings from a user-supplied,
third-party-distributed Windows ADOFAI ZIP. It does not add game binaries, scenes,
assets, executable installers, or third-party code to the repository.

## Artifact identity and scope

- Archive name: `A-Dance-Of-Fire-And-Ice-AnkerGames.zip`
- Archive size: `1,348,189,413` bytes; 1,415 ZIP records
- Uncompressed entry-size sum: `3,750,867,192` bytes
- ZIP central directory read using HTTP byte ranges, not full extraction
- Selected items extracted read-only and validated against their ZIP CRC32
- Windows build entry dates cluster around 2026-07-27; that **does not prove**
  the ADOFAI release version matches the Android package.
- Package provenance is unofficial. Never run included `.bat`, `.exe`, or
  redistributable binaries as part of analysis.

| Observed entry | Bytes | SHA-256 |
| --- | ---: | --- |
| `Assembly-CSharp.dll` | 2,785,280 | `0d1524dabf0f8c67aa58ba1992d6ed858522b9094490ebbb00e5e8aeff7f40cd` |
| `globalgamemanagers` | 1,011,540 | `2f224725b1041d83661589ee0c98e74d64fa618266940a593d8c5c05dab802a4` |
| `level5` | 602,692 | `45cffa740bd852b8ad9821eba5984f8c93b9016123a7ca1b0c645e36ef8d4800` |

`globalgamemanagers` explicitly identifies **Unity 6000.3.10f1**, matching
the Unity engine version previously observed for Android ADOFAI 3.3.1. This
alone does not establish same game build, scene compatibility, or ABI.

## Serialized editor scene exists in PC build

The PC player scene list in `globalgamemanagers` maps:

| Scene index / serialized file | Scene |
| --- | --- |
| 4 / `level4` | `Assets/scenes/scnGame.unity` |
| **5 / `level5`** | **`Assets/scenes/scnEditor.unity`** |
| 6 / `level6` | `Assets/scenes/scnCLS.unity` |

Parsing `level5` with a Unity serialized-object reader produced:

- 867 `GameObject` records, 1,220 `MonoBehaviour` records;
- an `editorCollider` GameObject on **layer 7** with
  `Transform` and `BoxCollider2D` components;
- a `levelEditorScene` GameObject with a Canvas/MonoBehaviours;
- a `MainTile` GameObject and other editor UI objects.

This **proves the current PC build ships the editor scene and object graph**.
It does **not** prove that the serialized PC scene can be inserted into the
Android 3.3.1 asset-split set. Cross-platform asset dependencies,
script identities, resources and the native game ABI must be checked.

The previously examined Android 3.3.1 package did not ship a serialized
`scnEditor` scene even though class/method metadata remains. That absence is
the key obstacle to directly restoring the desktop-style editor.

## Actual C# implementation boundaries

The PC `Assembly-CSharp.dll` is a managed .NET/Mono assembly (not an ARM64
IL2CPP executable). Static metadata and method bodies contain:

- `scnEditor.HandleMouseActions`
- `scnEditor.ObjectsAtMouse`
- `scnEditor.SelectFloor`
- `scnEditor.SmartObjectSelect`
- `scnEditor.GizmoAtMouse`
- `FloorMesh.GenerateCollider`
- `scrFloor.GenerateCollider`

Static decompilation of `scnEditor.ObjectsAtMouse` confirms the following
PC-version selection path:

1. When the frame changes, sample `Input.mousePosition` and convert through
   `camera.ScreenToWorldPoint(...).xy()`.
2. Limit floor collider work to floors within
   `scrFloor.LongDimensions.magnitude` of that world position.
3. Generate and enable `FloorMesh.polygonCollider` where applicable; for
   sprite-rendered floors, instantiate an angle-specific
   `scrFloor.GenerateCollider` prefab.
4. Query `Physics2D.RaycastAll` twice with the same world point, zero
   direction and distance, but separate **`foregroundMask`** and
   **`floorLayerMask`** fields.
5. Merge hits, map temporary collider parents back to their floor objects,
   sort candidates and update `foundObjects`.
6. Disable temporary colliders and destroy temporary instances.

`scrFloor.GenerateCollider` selects one of editor-owned Collider15/30/45/
60/75/90/105/120/135/165/180 prefabs based on floor angle. The actual
layers of each instantiated prefab must be inspected independently;
**the scene's layer-7 `editorCollider` is not sufficient proof of every
floor prefab's layer**.

`GizmoAtMouse` separately queries `handlesLayerMask`.
`SmartObjectSelect` calls `ObjectsAtMouse` again to support cycling through
overlaps. PC `ObjectsAtMouse` also has a frame guard, so multiple calls
within one frame do not necessarily imply multiple physics queries.

These are facts about the **PC 6000.3.10f1** managed implementation. They must
not be substituted for exact v2.4 Android ARM64 instruction evidence.

## Impact on the Android plan

- **Important positive:** the modern PC build provides both a current editor
  scene and inspectable managed implementation of the selection logic.
- **Not a completed port:** Unity runtime version equality is not sufficient
  to load desktop `level5` into Android, nor to use Windows Mono DLLs in
  Android IL2CPP. No patched APK or physical-device success follows from this.
- **Current safe vertical slice:** the Android-native 3.3.1 editor shell in this
  repository avoids legacy physics picking and can offer phone-native floor
  selection, but it is currently a schematic view rather than a complete
  reproduction of the PC editor scene.
- **Next compatibility gate:** compare PC scene dependencies/MonoScript
  references against the exact Android 3.3.1 asset-split/metadata set without
  changing the original installed game. If full scene integration proves
  impractical, use the managed implementation as a behavioral specification
  for the Android-native editor rather than transplanting proprietary binaries.
- **Separate v2.4 defect:** the phone's r31 zero-hit evidence still requires
  exact runtime collider layer/position/camera eligibility checks. The PC
  reference cannot retroactively prove that cause.

Do not publish extracted proprietary game assets or executable binaries to
this repository. Build/CI success and actual device selection success remain
separate gates.
