package com.unity3d.player;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/** Entry point injected into the historical 2.4 APK. No root/Zygisk dependency. */
public final class V240Bootstrap {
    private static final String TAG = "ADOFAI.V240Bootstrap";

    private V240Bootstrap() {}

    public static synchronized void init() {
        // Recovery bootstrap rule: never make an experimental native runtime a hard
        // dependency of UnityPlayerActivity.onCreate. V240RuntimeUpdater loads only a
        // hash-verified app-private cached candidate with crash-loop rollback. With no
        // healthy cached candidate the game stays in Java recovery mode instead of
        // repeatedly loading the embedded libv240fix.so and crashing the process.
        installMobileRuntimeWhenActivityIsReady();
    }

    private static void installMobileRuntimeWhenActivityIsReady() {
        final Handler main = new Handler(Looper.getMainLooper());
        final Runnable forceRebind = new Runnable() {
            @Override public void run() {
                try {
                    V240RuntimeUpdater.startIfReady();
                    // Activity recreation can leave the old overlay's process-global
                    // installed flag true before UnityPlayer.currentActivity points at
                    // the replacement Activity. Re-run idempotent binding after the
                    // transition window even if the normal retry loop ended early.
                    V240WindowCompat.apply();
                    V240SettingsOverlay.install();
                    V240CompatibilityReport.install();
                    V240SettingsOverlay.refresh();
                } catch (Throwable error) {
                    Log.w(TAG, "delayed mobile runtime rebind failed", error);
                }
            }
        };
        main.postDelayed(forceRebind, 500L);
        main.postDelayed(forceRebind, 1500L);
        main.post(new Runnable() {
            int attempts;

            @Override public void run() {
                attempts++;
                try {
                    V240RuntimeUpdater.startIfReady();
                    V240WindowCompat.apply();
                    V240SettingsOverlay.install();
                    V240CompatibilityReport.install();
                    V240SettingsOverlay.refresh();
                    if (V240SettingsOverlay.isInstalled()) return;
                } catch (Throwable error) {
                    Log.w(TAG, "mobile runtime install attempt failed", error);
                }
                // init() is injected at the first onCreate instruction. UnityPlayer.currentActivity
                // may not exist yet, so retry only until the overlay/runtime is actually installed.
                if (attempts < 24) main.postDelayed(this, 250L);
            }
        });
    }
}
