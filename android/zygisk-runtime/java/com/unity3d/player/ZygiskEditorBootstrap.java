package com.unity3d.player;

import android.app.Activity;
import android.content.pm.PackageInfo;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.lang.reflect.Field;

/** Starts the injected editor only after the supported official Unity Activity is alive. */
public final class ZygiskEditorBootstrap {
    private static final String TAG = "ADOFAI.ZygiskBootstrap";
    private static final String PACKAGE = "com.fizzd.connectedworlds";
    private static final String VERSION_NAME = "3.3.1";
    private static final long VERSION_CODE = 300382L;
    private static final int MAX_ATTEMPTS = 300;
    private static final long RETRY_MS = 100L;
    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private static boolean started;
    private static int attempts;

    private ZygiskEditorBootstrap() {}

    public static synchronized void start() {
        if (started) return;
        started = true;
        attempts = 0;
        MAIN.post(TRY_INSTALL);
    }

    private static final Runnable TRY_INSTALL = new Runnable() {
        @Override public void run() {
            Activity activity = resolveUnityActivity();
            if (activity == null || activity.isFinishing()) {
                if (++attempts < MAX_ATTEMPTS) {
                    MAIN.postDelayed(this, RETRY_MS);
                } else {
                    Log.e(TAG, "Timed out waiting for UnityPlayer.currentActivity");
                }
                return;
            }
            if (!isSupportedBuild(activity)) {
                Log.e(TAG, "Unsupported ADOFAI build; editor injection aborted fail-closed");
                return;
            }
            FileSelector.context = activity;
            MobileEditorShell.installLauncher();
            Log.i(TAG, "Editor launcher requested inside official Play-signed ADOFAI 3.3.1");
        }
    };

    private static Activity resolveUnityActivity() {
        try {
            Class<?> unityPlayer = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = unityPlayer.getField("currentActivity");
            Object value = field.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable error) {
            return null;
        }
    }

    private static boolean isSupportedBuild(Activity activity) {
        try {
            PackageInfo info = activity.getPackageManager().getPackageInfo(PACKAGE, 0);
            long code = Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode;
            boolean ok = VERSION_NAME.equals(info.versionName) && code == VERSION_CODE;
            if (!ok) {
                Log.e(TAG, "Expected ADOFAI " + VERSION_NAME + "/" + VERSION_CODE
                        + " but found " + info.versionName + "/" + code);
            }
            return ok;
        } catch (Throwable error) {
            Log.e(TAG, "Could not verify installed ADOFAI build", error);
            return false;
        }
    }
}
