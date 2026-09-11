package com.unity3d.player;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/** Entry point injected into the historical 2.4 APK. No root/Zygisk dependency. */
public final class V240Bootstrap {
    private static final String TAG = "ADOFAI.V240Bootstrap";
    private static boolean nativeStarted;

    private V240Bootstrap() {}

    public static synchronized void init() {
        // The native library only needs to load once per process, but this method is
        // injected into UnityPlayerActivity.onCreate and therefore also runs after an
        // Activity recreation. Always re-bind the mobile compatibility layer to the
        // current Activity.
        if (!nativeStarted) {
            nativeStarted = true;
            try {
                System.loadLibrary("v240fix");
                Log.i(TAG, "v240fix native runtime loaded");
            } catch (Throwable error) {
                Log.e(TAG, "v240fix native runtime failed to load", error);
            }
        }
        installMobileRuntimeWhenActivityIsReady();
    }

    private static void installMobileRuntimeWhenActivityIsReady() {
        final Handler main = new Handler(Looper.getMainLooper());
        main.post(new Runnable() {
            int attempts;

            @Override public void run() {
                attempts++;
                try {
                    V240WindowCompat.apply();
                    V240SettingsOverlay.install();
                    V240SettingsOverlay.refresh();
                    if (V240SettingsOverlay.isInstalled()) return;
                } catch (Throwable error) {
                    Log.w(TAG, "mobile runtime install attempt failed", error);
                }
                // init() can be injected at the first onCreate instruction. UnityPlayer.currentActivity
                // may not exist yet, so retry only until the overlay/runtime is actually installed.
                if (attempts < 24) main.postDelayed(this, 250L);
            }
        });
    }
}
