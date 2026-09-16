package dev.hoonex.adofai.v240.dynamic;

import android.app.Activity;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import java.lang.reflect.Field;

/** Hot-swappable entrypoint loaded from app-private code_cache by the stable bootstrap. */
public final class RuntimeEntry {
    private static final String TAG = "ADOFAI.V240Dynamic";
    private static final String LEGACY_GEAR_TAG = "adofai-v240-settings-button";

    private RuntimeEntry() {}

    private static native void nativeRegisterDynamicBridge(Class<?> bridgeClass);
    private static native void nativeReconcileDynamicRuntime();

    public static void install(Context context) {
        Context app = context == null ? null : context.getApplicationContext();
        Log.i(TAG, "dynamic runtime entry loaded; app=" +
                (app == null ? "null" : app.getPackageName()));

        // The native library is loaded by the stable bootstrap before this DEX. Register the
        // hot-swappable document importer, then reconcile hook installation regardless of whether
        // BNM or the DEX became ready first. No System.load is performed here.
        try {
            nativeRegisterDynamicBridge(DirectDocumentBridge.class);
            nativeReconcileDynamicRuntime();
            Log.i(TAG, "dynamic document bridge registered");
        } catch (Throwable error) {
            // Fail open: native SFB installation is gated on successful registration, while the
            // rest of the historical game remains usable if this optional runtime path is absent.
            Log.w(TAG, "dynamic document bridge registration failed", error);
        }

        scheduleLegacyGearRelocation();
    }

    private static void scheduleLegacyGearRelocation() {
        final Handler main = new Handler(Looper.getMainLooper());
        main.postDelayed(new Runnable() {
            @Override public void run() { relocateLegacyGear(); }
        }, 750L);
        main.postDelayed(new Runnable() {
            @Override public void run() { relocateLegacyGear(); }
        }, 1800L);
    }

    private static void relocateLegacyGear() {
        try {
            final Activity activity = currentActivity();
            if (activity == null || activity.isFinishing()) return;
            View decor = activity.getWindow().getDecorView();
            if (!(decor instanceof ViewGroup)) return;
            final View gear = decor.findViewWithTag(LEGACY_GEAR_TAG);
            if (gear == null) return;

            ViewGroup.LayoutParams raw = gear.getLayoutParams();
            if (!(raw instanceof FrameLayout.LayoutParams)) return;
            FrameLayout.LayoutParams lp = (FrameLayout.LayoutParams) raw;
            lp.width = dp(activity, 44);
            lp.height = dp(activity, 44);
            lp.gravity = Gravity.CENTER_VERTICAL | Gravity.END;
            lp.topMargin = 0;
            lp.bottomMargin = 0;
            lp.rightMargin = dp(activity, 12);
            gear.setLayoutParams(lp);
            gear.setContentDescription("ADOFAI 모바일 설정 (임시 버튼)");
            gear.requestLayout();
        } catch (Throwable error) {
            Log.w(TAG, "legacy settings button relocation failed", error);
        }
    }

    private static Activity currentActivity() {
        try {
            Class<?> unityPlayer = Class.forName("com.unity3d.player.UnityPlayer");
            Field current = unityPlayer.getField("currentActivity");
            Object value = current.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static int dp(Context context, int value) {
        DisplayMetrics metrics = context.getResources().getDisplayMetrics();
        float density = metrics == null ? 1f : Math.max(0.75f, metrics.density);
        return Math.max(1, Math.round(value * density));
    }
}
