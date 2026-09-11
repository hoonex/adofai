package com.unity3d.player;

import android.app.Activity;
import android.graphics.Rect;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Device-window compatibility policy for the historical Unity 2.4 activity. */
public final class V240WindowCompat {
    private static final String TAG = "ADOFAI.V240Window";
    private static View installedDecor;
    private static View.OnLayoutChangeListener layoutListener;

    private V240WindowCompat() {}

    public static void apply() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return;
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                applyOnUiThread(owner);
            }
        });
    }

    private static synchronized void applyOnUiThread(Activity owner) {
        try {
            Window window = owner.getWindow();
            if (window == null) return;

            // Old Unity layouts often assume a rectangular desktop-like viewport. Do not
            // render interactive editor controls underneath a physical notch/camera cutout.
            if (Build.VERSION.SDK_INT >= 28) {
                WindowManager.LayoutParams params = window.getAttributes();
                if (params.layoutInDisplayCutoutMode !=
                        WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER) {
                    params.layoutInDisplayCutoutMode =
                            WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER;
                    window.setAttributes(params);
                }
            }

            // Let the Unity surface resize when Android shows the IME so text fields and
            // numeric editor controls are not hidden behind the software keyboard.
            window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

            final View decor = window.getDecorView();
            if (decor == null) return;
            if (Build.VERSION.SDK_INT >= 29) installGestureGuard(decor);
        } catch (Throwable error) {
            Log.w(TAG, "window compatibility policy failed", error);
        }
    }

    private static void installGestureGuard(final View decor) {
        if (Build.VERSION.SDK_INT < 29) return;
        if (installedDecor != decor) {
            if (installedDecor != null && layoutListener != null) {
                try { installedDecor.removeOnLayoutChangeListener(layoutListener); }
                catch (Throwable ignored) {}
            }
            installedDecor = decor;
            layoutListener = new View.OnLayoutChangeListener() {
                @Override public void onLayoutChange(View v, int left, int top, int right, int bottom,
                                                     int oldLeft, int oldTop, int oldRight, int oldBottom) {
                    updateGestureExclusion(v);
                }
            };
            decor.addOnLayoutChangeListener(layoutListener);
        }
        updateGestureExclusion(decor);
    }

    private static void updateGestureExclusion(View decor) {
        if (Build.VERSION.SDK_INT < 29) return;
        int width = decor.getWidth();
        int height = decor.getHeight();
        if (width <= 0 || height <= 0) {
            decor.setSystemGestureExclusionRects(Collections.<Rect>emptyList());
            return;
        }

        // Protect only a narrow, central portion of each side. This reduces accidental
        // Android back-gesture capture while dragging editor UI, but deliberately leaves
        // the upper/lower edge available for the system back gesture.
        int edge = dp(decor, 18);
        edge = Math.max(1, Math.min(edge, Math.max(1, width / 12)));
        int bandTop = Math.round(height * 0.30f);
        int bandBottom = Math.round(height * 0.70f);
        if (bandBottom <= bandTop) return;

        List<Rect> rects = new ArrayList<Rect>(2);
        rects.add(new Rect(0, bandTop, edge, bandBottom));
        rects.add(new Rect(width - edge, bandTop, width, bandBottom));
        decor.setSystemGestureExclusionRects(rects);
    }

    private static int dp(View view, int value) {
        try {
            DisplayMetrics metrics = view.getResources().getDisplayMetrics();
            float density = metrics != null ? metrics.density : 1.0f;
            return Math.round(value * Math.max(0.75f, density));
        } catch (Throwable ignored) {
            return value;
        }
    }

    private static Activity currentActivity() {
        try {
            Class<?> player = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = player.getField("currentActivity");
            Object value = field.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable ignored) {
            return null;
        }
    }
}
