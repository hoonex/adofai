package com.unity3d.player;

import android.app.Activity;
import android.content.ClipData;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.os.Build;
import android.util.Log;

import java.io.File;

import dev.hoonex.adofai.companion.OfficialChartProvider;

/**
 * Non-root bridge from the standalone companion editor to the unmodified
 * Google Play ADOFAI 3.3.1 process.
 *
 * The historical mobile editor accepted a URL to a ZIP bundle.  We reproduce
 * that input shape locally: package the current bundle, serve it from a
 * loopback-only HTTP endpoint, and launch the official Unity activity with the
 * ZIP URL on all reasonable Android/Unity handoff surfaces at once.  A scoped
 * content:// URI for main.adofai is also attached as a fallback.
 */
public final class OfficialGameBridge {
    private static final String TAG = "ADOFAI.OfficialBridge";
    private static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";
    private static final String TARGET_ACTIVITY = "com.unity3d.player.UnityPlayerActivity";
    private static final String EXPECTED_VERSION_NAME = "3.3.1";
    private static final long EXPECTED_VERSION_CODE = 300382L;

    private static volatile String lastStatus = "공식 ADOFAI handoff를 아직 시도하지 않았습니다";

    private OfficialGameBridge() {}

    public static boolean open(String localPath) {
        Activity owner = FileSelector.context;
        if (owner == null || owner.isFinishing() || localPath == null || localPath.length() == 0) {
            lastStatus = "공식 ADOFAI를 열 수 없습니다: 편집기 Activity 또는 차트가 없습니다";
            return false;
        }
        if (!FileSelector.syncSavedPath(localPath)) {
            lastStatus = "공식 ADOFAI로 넘기기 전에 현재 bundle 저장에 실패했습니다";
            return false;
        }

        try {
            assertExactOfficialBuild(owner);

            File chart = new File(localPath);
            File bundle = BundleWorkspace.packageBundle(owner, localPath);
            String bundleUrl = LoopbackZipServer.publish(bundle);
            Uri bundleUri = Uri.parse(bundleUrl);

            Uri chartUri = OfficialChartProvider.publish(owner, chart);
            owner.grantUriPermission(TARGET_PACKAGE, chartUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);

            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setComponent(new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY));
            intent.setDataAndType(bundleUri, "application/zip");
            intent.setClipData(ClipData.newRawUri("ADOFAI main.adofai", chartUri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_SINGLE_TOP);

            // Reproduce the old Open From URL shape and simultaneously provide
            // the extracted chart URI. Unknown extras are harmless to Android;
            // the unmodified official build may ignore any surface it does not consume.
            intent.putExtra(Intent.EXTRA_TEXT, bundleUrl);
            intent.putExtra(Intent.EXTRA_STREAM, chartUri);
            intent.putExtra("url", bundleUrl);
            intent.putExtra("URL", bundleUrl);
            intent.putExtra("levelUrl", bundleUrl);
            intent.putExtra("zipUrl", bundleUrl);
            intent.putExtra("openFromUrl", bundleUrl);

            String encodedChart = chartUri.toString();
            intent.putExtra("path", encodedChart);
            intent.putExtra("filePath", encodedChart);
            intent.putExtra("levelPath", encodedChart);
            intent.putExtra("adofai", encodedChart);

            owner.startActivity(intent);
            lastStatus = "공식 ADOFAI 3.3.1에 ZIP URL bundle을 전달했습니다 (" + bundleUrl + ")";
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Official ZIP-URL handoff failed", error);
            lastStatus = "공식 ADOFAI handoff 실패: " + error.getClass().getSimpleName() + safeMessage(error);
            return false;
        }
    }

    public static String getLastStatus() {
        return lastStatus;
    }

    private static void assertExactOfficialBuild(Activity owner) throws Exception {
        PackageInfo info = owner.getPackageManager().getPackageInfo(TARGET_PACKAGE, 0);
        String versionName = info.versionName == null ? "unknown" : info.versionName;
        long versionCode = Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode;
        if (!EXPECTED_VERSION_NAME.equals(versionName) || versionCode != EXPECTED_VERSION_CODE) {
            throw new IllegalStateException("expected official " + EXPECTED_VERSION_NAME + "/" + EXPECTED_VERSION_CODE
                    + ", got " + versionName + "/" + versionCode);
        }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.length() == 0 ? "" : ": " + message;
    }
}
