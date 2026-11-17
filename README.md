# ADOFAI Loader Installer

Personal Android 14+ ARM64 installer for the officially purchased ADOFAI 3.3.1 (300382). This branch is independent of the existing game editor project. It contains the installer source and its APK only, not the game APK, game assets, or upstream loader binary.

## Download

- [Current installer](dist/ADOFAI-Loader-Installer-v1.0.1.apk)
- [Update metadata](dist/update.json)

## 1.0.1

- Declare ACCESS_NETWORK_STATE for Android 14+ JobScheduler requirements.
- Prevent scheduler or notification setup failure from blocking UI startup.
- Check installer updates from this branch and upstream loader releases separately. Download and verify SHA-256, package name, version and the installed signing certificate before offering installer updates. Android requires installation confirmation.
- Updates are checked approximately hourly; OS background restrictions can delay checks.
- Do not uninstall the official game without backing up save data. This app does not bypass purchase checks.

Build: Android Gradle Plugin 8.6.1, JDK 17, SDK 35, Gradle 8.9, :app:assembleDebug. Reuse the original private signing identity for a compatible update; never commit signing keys. For subsequent versions bump versionCode, publish the signed versioned APK, and update dist/update.json atomically in the same commit.

Verification: Gradle build and apksigner verification passed and installer 1.0.0 and 1.0.1 certificate digests matched. Physical-device launch and in-game patch execution are not verified.