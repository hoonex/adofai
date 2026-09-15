package dev.hoonex.adofai.v240.dynamic;

import android.content.Context;
import android.util.Log;

/**
 * Unique-name entrypoint loaded from app-private code_cache by the stable bootstrap.
 * Keep this package out of the embedded runtime so DexClassLoader can replace it on
 * future channel updates without reinstalling the APK.
 */
public final class RuntimeEntry {
    private static final String TAG = "ADOFAI.V240Dynamic";

    private RuntimeEntry() {}

    public static void install(Context context) {
        Context app = context == null ? null : context.getApplicationContext();
        Log.i(TAG, "dynamic runtime entry loaded; app=" + (app == null ? "null" : app.getPackageName()));
        // Recovery channel v3 deliberately installs no gameplay/event/UI hooks.
        // Its native payload waits for BNM, then performs metadata-only ABI discovery
        // for the exact v2.4 hook surfaces. It never calls or mutates managed game state.
        // System.load remains owned by the stable bootstrap/updater before this DEX runs.
    }
}
