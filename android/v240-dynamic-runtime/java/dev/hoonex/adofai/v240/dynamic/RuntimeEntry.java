package dev.hoonex.adofai.v240.dynamic;

import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/** Hot-swappable entrypoint loaded from app-private code_cache by the stable bootstrap. */
public final class RuntimeEntry {
    private static final String TAG = "ADOFAI.V240Dynamic";
    private static final String LEGACY_GEAR_TAG = "adofai-v240-settings-button";
    private static final int WINDOW_GUARD_REVISION = 25;
    private static boolean windowLifecycleInstalled;
    private static View guardedViewportDecor;
    private static View.OnLayoutChangeListener viewportLayoutListener;
    private static boolean parentRuntimeHealthConfirmed;

    private RuntimeEntry() {}

    public static void install(Context context) {
        Context app = context == null ? null : context.getApplicationContext();
        Log.i(TAG, "dynamic runtime entry loaded; app=" +
                (app == null ? "null" : app.getPackageName()));

        // libv240fix.so is loaded by the stable parent APK class loader before this child DEX.
        // Do not declare child-loader native methods here. Temporarily publish this DexClassLoader
        // as the thread context loader, then reach the stable parent V240CompatibilityReport via
        // normal parent-first delegation. Native resolves DirectDocumentBridge by name through the
        // context loader and reconciles SFB installation.
        installWindowLifecycleGuard(app);
        registerDynamicBridgeViaParent();
        scheduleNativeReconciliation();
        scheduleLegacyWindowNormalization();
        scheduleLegacyGearRelocation();
    }

    private static void registerDynamicBridgeViaParent() {
        final Thread thread = Thread.currentThread();
        final ClassLoader previous = thread.getContextClassLoader();
        final ClassLoader dynamic = RuntimeEntry.class.getClassLoader();
        try {
            thread.setContextClassLoader(dynamic);
            Class<?> report = Class.forName("com.unity3d.player.V240CompatibilityReport", true,
                    dynamic);
            Method nativeReport = report.getDeclaredMethod("nativeGetCompatibilityReport");
            nativeReport.setAccessible(true);
            Object value = nativeReport.invoke(null);
            Log.i(TAG, "dynamic document bridge parent reconcile=" +
                    (value instanceof String ? "ok" : "empty"));
        } catch (Throwable error) {
            // Fail open. Native SFB installation remains gated on successful bridge resolution.
            Log.w(TAG, "dynamic document bridge parent registration failed", error);
        } finally {
            try {
                thread.setContextClassLoader(previous);
            } catch (Throwable ignored) {
            }
        }
    }


    /**
     * Re-run parent-native reconciliation after startup state settles.
     * R20 leaves its one-shot unused while Persistence.generalPrefs is unavailable,
     * so these bounded retries can complete the durable repair without user action.
     */
    private static void scheduleNativeReconciliation() {
        final Handler main = new Handler(Looper.getMainLooper());
        final long[] delays = new long[] { 1200L, 3500L, 6500L };
        for (final long delay : delays) {
            main.postDelayed(new Runnable() {
                @Override public void run() { registerDynamicBridgeViaParent(); }
            }, delay);
        }
    }


    private static synchronized void installWindowLifecycleGuard(Context appContext) {
        if (windowLifecycleInstalled || !(appContext instanceof Application)) return;
        final Application app = (Application) appContext;
        app.registerActivityLifecycleCallbacks(new Application.ActivityLifecycleCallbacks() {
            @Override public void onActivityCreated(Activity activity, Bundle state) {
                if (isUnityActivity(activity)) {
                    normalizeLegacyWindowViewport(activity);
                    installViewportLayoutGuard(activity);
                }
            }

            @Override public void onActivityStarted(Activity activity) {}

            @Override public void onActivityResumed(Activity activity) {
                if (isUnityActivity(activity)) {
                    normalizeLegacyWindowViewport(activity);
                    installViewportLayoutGuard(activity);
                }
            }

            @Override public void onActivityPaused(Activity activity) {}

            @Override public void onActivityStopped(Activity activity) {
                markParentRuntimeHealthyOnGracefulStop(activity);
            }

            @Override public void onActivitySaveInstanceState(Activity activity, Bundle state) {}

            @Override public void onActivityDestroyed(Activity activity) {
                markParentRuntimeHealthyOnGracefulStop(activity);
                clearViewportLayoutGuard(activity);
            }
        });
        windowLifecycleInstalled = true;
        Activity activity = currentActivity();
        if (activity != null) installViewportLayoutGuard(activity);
        Log.i(TAG, "window viewport lifecycle guard r" + WINDOW_GUARD_REVISION + " installed");
    }

    private static boolean isUnityActivity(Activity activity) {
        if (activity == null || activity.isFinishing()) return false;
        Activity current = currentActivity();
        return current == null || current == activity;
    }

    /**
     * Bootstrap v3 originally kept boot.pending for a fixed 10 s foreground window.
     * A user could therefore exit normally twice during that window and be mistaken
     * for two startup crashes. Existing v3 installs cannot replace their parent DEX
     * through the hot channel, so the child runtime asks the parent's own markHealthy
     * method to close the pending boot when Unity receives a graceful stop/destroy.
     *
     * Reflection is deliberately fail-open and relies only on bootstrap-v3 internals.
     * Newer patcher builds also implement the lifecycle handling in the parent itself.
     */
    private static synchronized void markParentRuntimeHealthyOnGracefulStop(Activity activity) {
        if (parentRuntimeHealthConfirmed || !isUnityPlayerActivityClass(activity)) return;
        try {
            ClassLoader dynamic = RuntimeEntry.class.getClassLoader();
            Class<?> updater = Class.forName(
                    "com.unity3d.player.V240RuntimeUpdater", true, dynamic);
            Field loadedDirField = updater.getDeclaredField("loadedDir");
            loadedDirField.setAccessible(true);
            Object value = loadedDirField.get(null);
            if (!(value instanceof File)) return;

            File pending = new File((File) value, "boot.pending");
            Method markHealthy = updater.getDeclaredMethod("markHealthy", File.class);
            markHealthy.setAccessible(true);
            markHealthy.invoke(null, pending);
            if (!pending.exists()) {
                parentRuntimeHealthConfirmed = true;
                Log.i(TAG, "graceful stop confirmed parent runtime health");
            }
        } catch (Throwable error) {
            Log.w(TAG, "parent runtime graceful-stop health confirmation unavailable", error);
        }
    }

    private static boolean isUnityPlayerActivityClass(Activity activity) {
        return activity != null && hasUnityPlayerActivityType(activity.getClass());
    }

    private static boolean hasUnityPlayerActivityType(Class<?> type) {
        if (type == null) return false;
        if ("com.unity3d.player.UnityPlayerActivity".equals(type.getName())) return true;
        return hasUnityPlayerActivityType(type.getSuperclass());
    }

    private static synchronized void installViewportLayoutGuard(Activity activity) {
        if (!isUnityActivity(activity)) return;
        Window window = activity.getWindow();
        if (window == null) return;
        View decor = window.getDecorView();
        if (decor == null || guardedViewportDecor == decor) return;

        if (guardedViewportDecor != null && viewportLayoutListener != null) {
            try { guardedViewportDecor.removeOnLayoutChangeListener(viewportLayoutListener); }
            catch (Throwable ignored) {}
        }
        if (viewportLayoutListener == null) {
            viewportLayoutListener = new View.OnLayoutChangeListener() {
                @Override public void onLayoutChange(View v, int left, int top, int right, int bottom,
                                                     int oldLeft, int oldTop, int oldRight, int oldBottom) {
                    Activity current = currentActivity();
                    if (current != null && !current.isFinishing()) {
                        normalizeLegacyWindowViewport(current);
                    }
                }
            };
        }
        guardedViewportDecor = decor;
        decor.addOnLayoutChangeListener(viewportLayoutListener);
    }

    private static synchronized void clearViewportLayoutGuard(Activity activity) {
        if (guardedViewportDecor == null || viewportLayoutListener == null || activity == null) return;
        try {
            Window window = activity.getWindow();
            if (window != null && window.getDecorView() == guardedViewportDecor) {
                guardedViewportDecor.removeOnLayoutChangeListener(viewportLayoutListener);
                guardedViewportDecor = null;
            }
        } catch (Throwable ignored) {}
    }

    private static void scheduleLegacyWindowNormalization() {
        final Handler main = new Handler(Looper.getMainLooper());
        // V240Bootstrap's parent class reapplies its historical window policy during the
        // first ~1.5 s. Reassert SHORT_EDGES after those callbacks as well as immediately.
        main.post(new Runnable() {
            @Override public void run() { normalizeLegacyWindowViewport(); }
        });
        main.postDelayed(new Runnable() {
            @Override public void run() { normalizeLegacyWindowViewport(); }
        }, 1750L);
        main.postDelayed(new Runnable() {
            @Override public void run() { normalizeLegacyWindowViewport(); }
        }, 3000L);
        // The parent bootstrap retry loop is bounded to roughly six seconds. This final
        // assertion guarantees the hot runtime wins even if overlay installation was slow.
        main.postDelayed(new Runnable() {
            @Override public void run() { normalizeLegacyWindowViewport(); }
        }, 6500L);
    }

    private static void normalizeLegacyWindowViewport() {
        final Activity activity = currentActivity();
        if (activity == null || activity.isFinishing()) return;
        normalizeLegacyWindowViewport(activity);
    }

    private static void normalizeLegacyWindowViewport(Activity activity) {
        try {
            if (!isUnityActivity(activity)) return;
            final Window window = activity.getWindow();
            if (window == null) return;
            boolean changed = false;
            if (Build.VERSION.SDK_INT >= 28) {
                WindowManager.LayoutParams params = window.getAttributes();
                if (params.layoutInDisplayCutoutMode !=
                        WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES) {
                    params.layoutInDisplayCutoutMode =
                            WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
                    window.setAttributes(params);
                    changed = true;
                }
            }
            // Layout listeners call this method too. Only invalidate after an actual
            // policy transition; otherwise requestLayout() would feed the listener again.
            if (changed) {
                View decor = window.getDecorView();
                if (decor != null) {
                    decor.requestLayout();
                    decor.invalidate();
                }
            }
        } catch (Throwable error) {
            Log.w(TAG, "legacy full-width viewport normalization failed", error);
        }
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
