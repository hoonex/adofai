package dev.hoonex.adofai.gamepatcher;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.zip.ZipFile;

final class InstalledGame {
    static final String PACKAGE_NAME = "com.fizzd.connectedworlds";
    static final String EXPECTED_VERSION_NAME = "3.3.1";
    static final long EXPECTED_VERSION_CODE = 300382L;

    final String versionName;
    final long versionCode;
    final File baseApk;
    final File arm64Apk;
    final List<File> allApks;
    final long totalBytes;

    private InstalledGame(String versionName, long versionCode, File baseApk,
                          File arm64Apk, List<File> allApks, long totalBytes) {
        this.versionName = versionName;
        this.versionCode = versionCode;
        this.baseApk = baseApk;
        this.arm64Apk = arm64Apk;
        this.allApks = Collections.unmodifiableList(allApks);
        this.totalBytes = totalBytes;
    }

    static InstalledGame inspect(Context context) throws Exception {
        PackageManager pm = context.getPackageManager();
        PackageInfo info = pm.getPackageInfo(PACKAGE_NAME, 0);
        long versionCode = Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode;
        String versionName = info.versionName == null ? "" : info.versionName;
        if (!EXPECTED_VERSION_NAME.equals(versionName) || versionCode != EXPECTED_VERSION_CODE) {
            throw new IllegalStateException(
                "검증되지 않은 ADOFAI 버전: " + versionName + " (" + versionCode + ")\n" +
                "이 패처는 " + EXPECTED_VERSION_NAME + " / " + EXPECTED_VERSION_CODE + " 전용입니다."
            );
        }

        ApplicationInfo app = info.applicationInfo;
        if (app == null || app.sourceDir == null) {
            throw new IllegalStateException("ADOFAI base APK 경로를 찾지 못했습니다.");
        }

        File base = new File(app.sourceDir);
        if (!base.isFile() || !base.canRead()) {
            throw new IllegalStateException("ADOFAI base APK를 읽을 수 없습니다: " + base);
        }

        List<File> apks = new ArrayList<File>();
        apks.add(base);
        if (app.splitSourceDirs != null) {
            for (String split : app.splitSourceDirs) {
                File f = new File(split);
                if (!f.isFile() || !f.canRead()) {
                    throw new IllegalStateException("설치 split을 읽을 수 없습니다: " + f);
                }
                apks.add(f);
            }
        }
        Collections.sort(apks, new Comparator<File>() {
            @Override public int compare(File a, File b) {
                if (a.equals(base)) return -1;
                if (b.equals(base)) return 1;
                return a.getName().compareTo(b.getName());
            }
        });

        File arm64 = null;
        long total = 0L;
        for (File apk : apks) {
            total += apk.length();
            String lower = apk.getName().toLowerCase();
            if (lower.contains("arm64") && hasEntry(apk, "lib/arm64-v8a/libil2cpp.so")) {
                arm64 = apk;
            }
        }
        if (arm64 == null) {
            for (File apk : apks) {
                if (hasEntry(apk, "lib/arm64-v8a/libil2cpp.so")) {
                    arm64 = apk;
                    break;
                }
            }
        }
        if (arm64 == null) {
            throw new IllegalStateException("arm64 libil2cpp split을 찾지 못했습니다.");
        }
        if (!hasEntry(base, "classes.dex")) {
            throw new IllegalStateException("base.apk에 classes.dex가 없습니다.");
        }

        return new InstalledGame(versionName, versionCode, base, arm64, apks, total);
    }

    static boolean isInstalled(Context context) {
        try {
            context.getPackageManager().getPackageInfo(PACKAGE_NAME, 0);
            return true;
        } catch (PackageManager.NameNotFoundException notInstalled) {
            return false;
        }
    }

    private static boolean hasEntry(File apk, String name) {
        try (ZipFile zip = new ZipFile(apk)) {
            return zip.getEntry(name) != null;
        } catch (Throwable ignored) {
            return false;
        }
    }

    String describe() {
        StringBuilder out = new StringBuilder();
        out.append("ADOFAI ").append(versionName).append(" (").append(versionCode).append(")\n");
        out.append("split ").append(allApks.size()).append("개 / ")
            .append(formatBytes(totalBytes)).append("\n");
        out.append("base: ").append(baseApk.getName()).append("\n");
        out.append("arm64: ").append(arm64Apk.getName());
        return out.toString();
    }

    static String formatBytes(long value) {
        double gb = value / (1024.0 * 1024.0 * 1024.0);
        if (gb >= 1.0) return String.format(java.util.Locale.US, "%.2f GB", gb);
        double mb = value / (1024.0 * 1024.0);
        return String.format(java.util.Locale.US, "%.1f MB", mb);
    }
}
