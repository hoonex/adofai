# ADOFAI Loader Installer

Personal Android 14+ ARM64 installer for the officially purchased ADOFAI 3.3.1 (300382). This branch is independent of the existing game editor project. It contains the installer source and its APK only, not the game APK, game assets, or upstream loader binary.

## Download

- [Current installer](dist/ADOFAI-Loader-Installer-v1.0.3.apk)
- [Update metadata](dist/update.json)

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