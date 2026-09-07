package com.unity3d.player;

import android.util.Log;

/**
 * Zero-register smali bootstrap target for the injected native editor runtime.
 *
 * UnityPlayerActivity can invoke init() with no arguments, so the activity patch
 * does not need to reserve a scratch v-register just to call System.loadLibrary.
 * This class lives in the injected secondary DEX and keeps the native load
 * fail-closed: the game continues if the optional editor library cannot load.
 */
public final class MobileEditorBootstrap {
    private static final String TAG = "ADOFAI.EditorBootstrap";
    private static boolean loaded;

    private MobileEditorBootstrap() {}

    public static synchronized void init() {
        if (loaded) return;
        try {
            System.loadLibrary("October");
            loaded = true;
            Log.i(TAG, "libOctober loaded through secondary-DEX bootstrap");
        } catch (Throwable error) {
            Log.e(TAG, "libOctober bootstrap failed", error);
        }
    }
}
