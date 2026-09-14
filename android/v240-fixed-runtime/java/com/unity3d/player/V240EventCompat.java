package com.unity3d.player;

import android.util.Log;

/** Native capability gate for post-v2.4 event execution backports. */
final class V240EventCompat {
    private static final String TAG = "ADOFAI.V240EventCompat";

    private V240EventCompat() {}

    static void initialize() {
        try {
            nativeRegister();
        } catch (Throwable error) {
            // Preserve-only opaque event handling remains the authoritative fallback.
            Log.w(TAG, "event compatibility runtime registration unavailable", error);
        }
        try {
            // Evidence-only TileDimensions observation is registered independently so a
            // diagnostic failure can never disable the proven SetFrameRate path above.
            nativeRegisterTileDimensionsSnapshot();
        } catch (Throwable error) {
            Log.w(TAG, "TileDimensions read-only snapshot registration unavailable", error);
        }
    }

    static boolean isSetFrameRateBackportReady() {
        try {
            return nativeIsSetFrameRateBackportReady();
        } catch (Throwable error) {
            Log.w(TAG, "SetFrameRate runtime capability unavailable", error);
            return false;
        }
    }

    private static native void nativeRegister();
    private static native void nativeRegisterTileDimensionsSnapshot();
    private static native boolean nativeIsSetFrameRateBackportReady();
}
