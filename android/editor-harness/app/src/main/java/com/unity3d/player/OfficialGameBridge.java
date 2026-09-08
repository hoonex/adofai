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
 * The canonical handoff packages the current bundle, serves it from a
 * loopback-only HTTP endpoint, and launches the official Unity activity with
 * the ZIP URL. A scoped content:// URI for main.adofai is attached as fallback.
 *
 * For URL-imported bundles an isolated HTTPS probe is also available. It sends
 * only the original HTTPS ZIP URL (no content URI fallback), allowing a device
 * test to distinguish loopback cleartext/network-policy failure from failure to
 * consume a historical Open-From-URL-shaped external URL.
 */
public final class OfficialGameBridge {
    private static final String TAG = "ADOFAI.OfficialBridge";
    private static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";
    private static final String TARGET_ACTIVITY = "com.unity3d.player.UnityPlayerActivity";
    private static final String EXPECTED_VERSION_NAME = "3.3.1";
    private static final long EXPECTED_VERSION_CODE = 300382L;
    private static final long MIN_RETURN_DIAGNOSTIC_DELAY_MS = 300L;

    private static volatile String lastStatus = "공식 ADOFAI handoff를 아직 시도하지 않았습니다";
    private static volatile String lastBundleUrl;
    private static volatile long lastHandoffAtMs;
    private static volatile boolean pendingReturnDiagnostic;

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

            Intent intent = buildUrlIntent(bundleUri, bundleUrl);
            intent.setClipData(ClipData.newRawUri("ADOFAI main.adofai", chartUri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.putExtra(Intent.EXTRA_STREAM, chartUri);

            String encodedChart = chartUri.toString();
            intent.putExtra("path", encodedChart);
            intent.putExtra("filePath", encodedChart);
            intent.putExtra("levelPath", encodedChart);
            intent.putExtra("adofai", encodedChart);

            lastBundleUrl = bundleUrl;
            lastHandoffAtMs = System.currentTimeMillis();
            pendingReturnDiagnostic = true;
            try {
                owner.startActivity(intent);
            } catch (Throwable launchError) {
                pendingReturnDiagnostic = false;
                throw launchError;
            }
            lastStatus = "공식 ADOFAI 3.3.1에 localhost ZIP URL bundle을 전달했습니다 (" + bundleUrl + ")";
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Official ZIP-URL handoff failed", error);
            lastStatus = "공식 ADOFAI handoff 실패: " + error.getClass().getSimpleName() + safeMessage(error);
            return false;
        }
    }

    /**
     * Sends only the original HTTPS URL used to import this bundle. This is a
     * diagnostic path: it intentionally does not send the edited local bundle
     * or the content:// fallback, so a positive result is attributable to the
     * external URL-shaped entry surface.
     */
    public static boolean openOriginalHttps(String localPath) {
        Activity owner = FileSelector.context;
        if (owner == null || owner.isFinishing() || localPath == null || localPath.length() == 0) {
            lastStatus = "원본 HTTPS 테스트를 실행할 수 없습니다: 편집기 Activity 또는 차트가 없습니다";
            return false;
        }

        String sourceUrl = BundleWorkspace.sourceHttpsUrlForChart(localPath);
        if (sourceUrl == null) {
            lastStatus = "원본 HTTPS 테스트는 HTTPS ZIP URL로 가져온 bundle에서만 사용할 수 있습니다";
            return false;
        }

        try {
            assertExactOfficialBuild(owner);
            pendingReturnDiagnostic = false;
            Intent intent = buildUrlIntent(Uri.parse(sourceUrl), sourceUrl);
            owner.startActivity(intent);
            lastStatus = "원본 HTTPS ZIP URL만 공식 ADOFAI 3.3.1에 전달했습니다. 이 테스트는 로컬 편집 내용을 포함하지 않습니다: " + sourceUrl;
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Official original-HTTPS handoff failed", error);
            lastStatus = "원본 HTTPS handoff 실패: " + error.getClass().getSimpleName() + safeMessage(error);
            return false;
        }
    }

    private static Intent buildUrlIntent(Uri uri, String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setComponent(new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY));
        intent.setDataAndType(uri, "application/zip");
        intent.addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        intent.putExtra(Intent.EXTRA_TEXT, url);
        intent.putExtra("url", url);
        intent.putExtra("URL", url);
        intent.putExtra("levelUrl", url);
        intent.putExtra("zipUrl", url);
        intent.putExtra("openFromUrl", url);
        return intent;
    }

    public static String getLastStatus() {
        return lastStatus;
    }

    /**
     * Called when the Companion activity resumes after the official game was
     * launched. The diagnostic is consumed once so ordinary picker resumes do
     * not repeatedly overwrite the editor status.
     */
    public static synchronized String consumeReturnDiagnostic() {
        if (!pendingReturnDiagnostic || lastBundleUrl == null) return null;
        if (System.currentTimeMillis() - lastHandoffAtMs < MIN_RETURN_DIAGNOSTIC_DELAY_MS) return null;
        pendingReturnDiagnostic = false;
        String diagnostic = LoopbackZipServer.diagnosticFor(lastBundleUrl);
        return diagnostic == null ? "ZIP URL 진단 정보를 찾지 못했습니다." : diagnostic;
    }

    /** Non-consuming snapshot for debugging/tests. */
    public static String getLastProbeStatus() {
        String url = lastBundleUrl;
        return url == null ? null : LoopbackZipServer.diagnosticFor(url);
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
