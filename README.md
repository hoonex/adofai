# ADOFAI Loader Installer

Personal Android 14+ ARM64 installer for the officially purchased ADOFAI 3.3.1 (300382). This branch is independent of the existing game editor project. It contains the installer source and its APK only, not the game APK, game assets, or upstream loader binary.

## Download

- [Current installer](dist/ADOFAI-Loader-Installer-v1.0.5.apk)
- [Update metadata](dist/update.json)

## 1.0.5

- Korean native loader ko.2: system ZIP picker via Android ACTION_OPEN_DOCUMENT, local or cloud document providers, scoped storage copy with byte progress, then upstream automatic ZIP extraction and level selection.
- Rebuild from the original backup once to embed the Java bridge DEX. Existing patch results require repatching. No storage-wide access permission needed.
- Native CI and installer builds, DEX inspection and signing checks passed; real Android file-picker and map playback still need device validation.

## 1.0.4

- Korean UI defaults to Korean (76 translated keys, Hangul glyph ranges) with English and Chinese available in loader settings. Existing legacy zh settings migrate to Korean.
- Korean native build is pinned to upstream v1.1.10.1; the installer verifies its SHA-256 and offers the official upstream separately. Future upstream releases require a new Korean build and translation coverage check.
- Build and remote hash validation passed; device UI remains unverified.

## 1.0.3

- Follow the upstream author installation recipe for locally purchased ADOFAI 3.3.1 only. New input backups require a Play Store installation. Existing local original backups are version-checked. No game APK is distributed.
- Scan all DEX files and verify all required targets exist before marking a patch complete. Earlier patch results must be rebuilt from the original backup.
- Synthetic DEX smoke tests passed (six cases); physical-device launch is not yet verified.

## 1.0.2

- Real byte progress for downloads, APK copies, export and install staging; indeterminate progress plus elapsed time for DEX/signing steps.
- Detect mismatched game signing certificates before install; no automatic uninstall. Added app settings and original APK restore actions. Game saves are not backed up by APK export.
- Keep Android confirmation pending until installation callback reports result. Fix literal newline and status bar contrast.
- Device execution remains unverified.

## 1.0.1

- Declare ACCESS_NETWORK_STATE for Android 14+ JobScheduler requirements.
- Prevent scheduler or notification setup failure from blocking UI startup.
- Check installer updates from this branch and upstream loader releases separately. Download and verify SHA-256, package name, version and the installed signing certificate before offering installer updates. Android requires installation confirmation.
- Updates are checked approximately hourly; OS background restrictions can delay checks.
- Do not uninstall the official game without backing up save data. This installer is for personal use with a legitimately purchased copy; do not redistribute modified game files.

Build: Android Gradle Plugin 8.6.1, JDK 17, SDK 35, Gradle 8.9, :app:assembleDebug. Reuse the original private signing identity for a compatible update; never commit signing keys. For subsequent versions bump versionCode, publish the signed versioned APK, and update dist/update.json atomically in the same commit.

Verification: Gradle build and apksigner verification passed and installer 1.0.0 and 1.0.1 certificate digests matched. Physical-device launch and in-game patch execution are not verified.
## 1.0.11

- Self-update from inside the installer: a single button fetches update.json, validates SHA-256, installed signing certificate and version, then opens the Android installer approval. No silent update bypass.
- Replace the old hourly persistent JobScheduler with foreground/resume/network-recovery checks, throttled to 4 hours; background recurring jobs are removed.
- Game-facing custom home now has red/cyan orbit graphics, access to the original game's PauseMenu SettingsMenu (mobile resolution / effects), and an opt-in custom-map minimal-visual-effects switch. Engine hit timing and chart data are not modified.
- Stage media pruning remains opt-in/experimental: exact-version inventory validation and preserved original backup. Android device regression is still pending.


## 1.0.12

- Custom home: centered compact ADOFAI-style red/cyan orbit layout, two landscape panels, recent maps independently touch-scroll in bounded-height list, UTF-8-safe filename shortening with full-path tooltip, and accessible Settings/Storage controls even with many recently played maps. Portrait remains one-column.
- Added a mutually exclusive opt-in experimental 'Tiles + Music' visual mode distinct from the original minimum-effects switch. It suppresses a conservative subset of known visual-only custom stage events and background GameObjects, captures original enabled/active state for restoration and preserves input/rhythm/music/tile/planet logic. Not guaranteed to remove every decoration in arbitrary maps; no runtime smoke test on a connected Android device yet.
- Original graphics/resolution SettingsMenu access remains intact. Correct original calibration API overload selection based on Unity 6000.3.10f1 metadata. Updated UI design evidence guide at docs/ADOFAI_UI_DESIGN_GUIDE.md.
- Installs into a locally purchased Android 3.3.1 copy only; does not include paid original game binaries. Existing original backup and opt-in installed-stage resource pruning continue to be supported.

## 1.0.13

- Safety follow-up to 1.0.12: stronger scoped visual-state restoration and exact Unity 6000 Type/bool overload resolution validated against local IL2CPP metadata. Stage and hit mechanics are unmodified. Device runtime test pending.

## v1.0.14 — Fast-input and Tiles+Music capability fixes

- Fixed a reproduced mobile fast-tap loss path: dismissing the overlay formerly blocked ALL gameplay input until Unity reported touchCount==0. Now only overlay-owned finger IDs are quarantined, and other live/recycled fingers pass through without timing changes. Original 68-byte Touch values remain intact. The non-input settings/event poll is bounded to one call per 8ms; actual touch callbacks are not rate limited. Host tests cover 2,500 overlapping rapid bursts, 32-pointer boundaries, mouse transition and repeated touchCount queries.
- Fixed the Tiles+Music unsupported toggle in ADOFAI 3.3.1: ffxPlusBase.runOnHit is a virtual get_runOnHit() property, not a field. Exact subclass getter and GameObject.GetComponent(Type) signatures are validated. Unknown versions fail safely with the diagnostics shown in Settings. The stronger visual mode also includes original VisualEffects.Minimum internally; chart hit/timing/audio remain untouched, and only known decorative components/backgrounds are targeted.
- Simplified installer home: the only default workflow is **1. Official-stage asset removal + patch** and **2. Install patched game**. Other patch modes, original restore, backup and cache management are under a collapsible advanced section; no recovery features removed.
- Stage pruning remains an exact, SHA-256/ZIP-inventory-verified transformation of the user's own installed 3.3.1 assets, with original APK backup intact. The installer does not contain the paid original game.
- Verification: Android NDK AArch64 full link, Gradle 8.9/SDK35 assembleDebug, APK v2 signing/zipalign/CRC, embedded native SHA and stage prune plan, native input host regression. No live Android device was connected: runtime tap handling and custom noFX scene/render restoration are not claimed verified on device.