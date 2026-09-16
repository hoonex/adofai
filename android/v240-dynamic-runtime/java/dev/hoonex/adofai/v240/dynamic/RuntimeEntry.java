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

/**
 * Unique-name entrypoint loaded from app-private code_cache by the stable bootstrap.
 * Keep this package out of the embedded runtime so DexClassLoader can replace it on
 * future channel updates without reinstalling the APK.
 */
public final class RuntimeEntry {
    private static final String TAG = "ADOFAI.V240Dynamic";
    private static final String LEGACY_GEAR_TAG = "adofai-v240-settings-button";

    private RuntimeEntry() {}

    public static void install(Context context) {
        Context app = context == null ? null : context.getApplicationContext();
        Log.i(TAG, "dynamic runtime entry loaded; app=" +
                (app == null ? "null" : app.getPackageName()));

        // Recovery channel v6 is deliberately activation-free on the native side.
        // The native payload waits for BNM, then performs metadata-only ABI discovery.
        // It installs no BasicHook, invokes no managed game method, and mutates no game
        // state. SFB ExtensionFilter[] remains diagnostic-only until its exact call ABI
        // and value layout are proven on the target v2.4 runtime.
        //
        // The embedded bootstrap still creates a temporary Android gear button. Move
        // that Java-owned entry point away from the editor's top toolbar. This is only
        // a fail-open UI relocation; it does not activate any native gameplay hook.
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
