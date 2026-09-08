package dev.hoonex.adofai.gamepatcher;

import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.content.pm.PackageManager;
import android.os.Build;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

final class PreparedInstaller {
    static int install(Context context, PatchPipeline.PreparedSet prepared) throws Exception {
        if (prepared == null || prepared.apks.isEmpty()) {
            throw new IllegalStateException("준비된 패치 APK 세트가 없습니다.");
        }
        if (InstalledGame.isInstalled(context)) {
            throw new IllegalStateException("Play판 ADOFAI가 아직 설치되어 있습니다. 서명이 달라 먼저 시스템 삭제 확인이 필요합니다.");
        }
        if (Build.VERSION.SDK_INT >= 26 && !context.getPackageManager().canRequestPackageInstalls()) {
            throw new SecurityException("UNKNOWN_SOURCES_PERMISSION_REQUIRED");
        }

        long total = 0L;
        for (File apk : prepared.apks) total += apk.length();

        PackageInstaller installer = context.getPackageManager().getPackageInstaller();
        PackageInstaller.SessionParams params = new PackageInstaller.SessionParams(
            PackageInstaller.SessionParams.MODE_FULL_INSTALL
        );
        params.setAppPackageName(InstalledGame.PACKAGE_NAME);
        params.setSize(total);
        if (Build.VERSION.SDK_INT >= 26) params.setInstallReason(PackageManager.INSTALL_REASON_USER);

        int sessionId = installer.createSession(params);
        boolean committed = false;
        try (PackageInstaller.Session session = installer.openSession(sessionId)) {
            byte[] buffer = new byte[1024 * 1024];
            for (File apk : prepared.apks) {
                try (InputStream in = new BufferedInputStream(new FileInputStream(apk), buffer.length);
                     OutputStream out = session.openWrite(apk.getName(), 0L, apk.length())) {
                    int count;
                    while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                    session.fsync(out);
                }
            }

            Intent callback = new Intent(context, InstallResultReceiver.class)
                .setAction(InstallResultReceiver.ACTION_INSTALL_RESULT);
            int flags = PendingIntent.FLAG_UPDATE_CURRENT;
            if (Build.VERSION.SDK_INT >= 31) flags |= PendingIntent.FLAG_MUTABLE;
            PendingIntent pending = PendingIntent.getBroadcast(context, sessionId, callback, flags);
            session.commit(pending.getIntentSender());
            committed = true;
        } finally {
            if (!committed) installer.abandonSession(sessionId);
        }
        return sessionId;
    }

    private PreparedInstaller() {}
}
