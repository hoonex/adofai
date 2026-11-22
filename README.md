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

