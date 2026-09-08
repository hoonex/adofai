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
            lastStatus = "공식 ADOFAI로 넘기기 전에 저장 문서 동기화에 실패했습니다";
            return false;
        }

        try {
            assertExactOfficialBuild(owner);
            Uri uri = OfficialChartProvider.publish(owner, new File(localPath));
            owner.grantUriPermission(TARGET_PACKAGE, uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);

            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setComponent(new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY));
            intent.setDataAndType(uri, "application/json");
            intent.setClipData(ClipData.newRawUri("ADOFAI chart", uri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_SINGLE_TOP);

            // Supply the same chart through the common Android/Unity handoff surfaces
            // in one launch. The official build is not modified and may legally ignore
            // fields it does not consume.
            intent.putExtra(Intent.EXTRA_STREAM, uri);
            String encoded = uri.toString();
            intent.putExtra("url", encoded);
            intent.putExtra("path", encoded);
            intent.putExtra("filePath", encoded);
            intent.putExtra("levelPath", encoded);
            intent.putExtra("adofai", encoded);

            owner.startActivity(intent);
            lastStatus = "공식 ADOFAI 3.3.1에 차트를 전달했습니다. 공식 앱이 외부 차트 URI를 소비하지 않으면 메인 화면만 열릴 수 있습니다";
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Explicit official-game handoff failed", error);
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
