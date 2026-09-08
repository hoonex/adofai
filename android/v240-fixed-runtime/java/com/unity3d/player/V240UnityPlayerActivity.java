package com.unity3d.player;

import android.os.Bundle;
import android.util.Log;

/**
 * Drop-in launcher activity for the historical 2.4 APK.
 * The manifest patch only changes the existing launcher activity class name, so all
 * original intent filters and Unity activity attributes stay intact.
 */
public final class V240UnityPlayerActivity extends UnityPlayerActivity {
    private static final String TAG = "ADOFAI.V240Activity";
    private static volatile boolean nativeLoaded;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ensureRuntime();
        V240SettingsOverlay.install();
    }

    @Override protected void onResume() {
        super.onResume();
        ensureRuntime();
        V240SettingsOverlay.install();
    }

    private static synchronized void ensureRuntime() {
        if (nativeLoaded) return;
        try {
            System.loadLibrary("v240fix");
            nativeLoaded = true;
            Log.i(TAG, "2.4 fixed native runtime loaded");
        } catch (Throwable error) {
            Log.e(TAG, "2.4 fixed native runtime failed to load", error);
        }
    }
}
