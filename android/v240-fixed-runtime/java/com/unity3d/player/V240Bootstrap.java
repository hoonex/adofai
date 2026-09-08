package com.unity3d.player;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/** Entry point injected into the historical 2.4 APK. No root/Zygisk dependency. */
public final class V240Bootstrap {
    private static final String TAG = "ADOFAI.V240Bootstrap";
    private static boolean started;

    private V240Bootstrap() {}

    public static synchronized void init() {
        if (started) return;
        started = true;
        try {
            System.loadLibrary("v240fix");
            Log.i(TAG, "v240fix native runtime loaded");
        } catch (Throwable error) {
            Log.e(TAG, "v240fix native runtime failed to load", error);
        }
        installOverlayWhenActivityIsReady();
    }

    private static void installOverlayWhenActivityIsReady() {
        final Handler main = new Handler(Looper.getMainLooper());
        main.post(new Runnable() {
            int attempts;

            @Override public void run() {
                attempts++;
                try {
                    V240SettingsOverlay.install();
                } catch (Throwable error) {
                    Log.w(TAG, "mobile settings overlay install attempt failed", error);
                }
                // init() can be injected at the first onCreate instruction. UnityPlayer.currentActivity
                // may not exist yet, so keep retrying during the short startup window. install() is
                // idempotent and refuses to add a duplicate button once it succeeds.
                if (attempts < 24) main.postDelayed(this, 250L);
            }
        });
    }
}
