# ADOFAI 3.3.1 integrated Custom Android delivery boundary

The intended installable product is **ADOFAI itself with an editor integrated
into the game**, not a second standalone editor. The separate Companion Editor
APK is only a compatibility/test fallback; it is **not** the final game.

## Why there is a builder APK

Official ADOFAI Android 3.3.1 is delivered as a multi-APK split set containing
proprietary Unity runtime, assets and licensed music. This repository cannot
produce or distribute an actual modified **full game** from source alone:
the exact verified user's installed original APK split set is a required input.
The unofficial Windows PC archive is not an Android split set, and Unity
version equality does not make the managed PC DLL or PC serialized assets an
Android game package. **Some unofficial 3.3.1 downloads are monolithic APKs**;
these must never be represented as the known official multi-split install
without actual archive and signing evidence.

The build artifact `ADOFAI-3.3.1-Game-Patcher.apk` is therefore a **local game
builder and system installer**, not itself the game. It embeds only our compiled
editor payload and applies it to the exact ADOFAI 3.3.1 / 300382 install on the
user's device.

## What the on-device builder actually does

1. Inspect the installed package `com.fizzd.connectedworlds`,
   verify version `3.3.1` / `300382`, and locate its base plus any installed
   splits. Monolithic APKs with arm64 Unity libraries embedded in the base
   are supported as a distinct source layout.
   Version or ABI drift fails closed.
2. Copy and patch the app-private work copies of the base APK and, when
   separate, the arm64 split. A second DEX adds `MobileEditorShell` and
   `MobileTileMapView`; the native `libOctober.so` profile adds
   runtime bindings. For **monolithic** installs both modifications occur
   in **the same one APK**. Other existing splits remain logically unchanged.
3. Reject collisions with pre-existing editor DEX or native libraries.
   Re-sign every split with a single persistent AndroidKeyStore identity.
   Confirm critical Unity native/metadata entries remain byte-identical.
4. Persist a report with names, sizes, hashes and signer fingerprint.
   Validate every APK hash and the device signing identity before proceeding
   with installation, including after app restart.
5. Require an explicit acknowledgment of game data backup; only then offer
   the **Android system** remove-original prompt if the Play-signed original is
   present. The app never silently uninstalls, clears data or overwrites it.
6. When the differently signed original is gone, request Android package
   install permission if necessary and send **the complete set of modified
   split APKs** to the system PackageInstaller for user confirmation.

## Genuinely verified and unverified

- Editor Java source contract tests and independent Android APK compile/CI
  establish only that the source **builds**.
- The actual installed-game transformation needs **a device with the exact
  input game package** (official split set or a verified monolithic APK), free
  space, a working device keystore, and an Android PackageInstaller round-trip.
  **Source-only CI cannot prove successful transformation or installation.**
- A newly discovered user-provided Drive APK
  `A_Dance_Of_Fire_And_Ice_3.3.1_1787072462_latestmodapks.com.apk` was
  registered in Drive at 2,683,468,431 bytes on 2026-10-28. Filename and
  metadata are *not proof* of actual package identity, manifest version,
  signing certificate, archive integrity or safety. Its provider-imposed
  256MiB raw fetch limit and unavailable connected Windows tunnel currently
  prevent reading its actual APK entries. Never copy its game binaries into
  this public repository or distribute a modified full APK from CI.
- The original Unity `scnEditor` scene is **not bundled into Android 3.3.1**.
  The current game editor is an Android-native overlay with Chart, Settings,
  Events, Raw, and schematic Tiles selection. The current implementation is
  not feature-equivalent to the complete Windows PC editor (effects, precise
  scene visuals, gizmos and animation editing remain to be ported).
- A physical-phone acceptance run must confirm: open chart, tap selects exact
  floor, drag/pinch, add/edit/delete events, save and reopen byte-preserving
  chart structure, native Preview/current gameplay, cancel/failure recovery,
  system signature/install behavior and original app data backup.

Never publish the unofficial source archive's original game binaries as
repository assets or GitHub Actions artifacts.
