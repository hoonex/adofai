package com.unity3d.player;

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
            // Settings remains available so native compatibility failures are diagnosable
            // without preventing the historical game/editor from launching.
            Log.e(TAG, "v240fix native runtime failed to load", error);
        }
        try {
            V240SettingsOverlay.install();
        } catch (Throwable error) {
            Log.e(TAG, "mobile settings overlay install failed", error);
        }
    }
}
