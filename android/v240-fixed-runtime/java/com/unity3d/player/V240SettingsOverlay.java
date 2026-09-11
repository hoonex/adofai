package com.unity3d.player;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.view.Display;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import java.lang.reflect.Field;

/** Mobile-only controls consumed directly by libv240fix.so. */
public final class V240SettingsOverlay {
    private static final String PREFS = "adofai-v240-mobile";
    private static final String TAG = "adofai-v240-settings-button";
    private static final float DEFAULT_UI_SCALE = 1.15f;
    private static final float DEFAULT_TOUCH_SCALE = 1.30f;
    private static final float DEFAULT_DRAG_SCALE = 1.05f;
    private static final int DEFAULT_FPS_MODE = 0;
    private static final int[] FPS_VALUES = {0, 60, 90, 120, 144, 165, 240};
    private static final String[] FPS_LABELS = {
            "화면 최대 Hz (권장)", "60 FPS", "90 FPS", "120 FPS", "144 FPS", "165 FPS", "240 FPS"
    };
    private static Activity activity;

    private V240SettingsOverlay() {}

    private static native void nativeApply(
            float uiScale,
            float touchScale,
            float dragScale,
            boolean touchAssist,
            int targetFps,
            boolean unlockFps,
            boolean lowLatency);

    public static void install() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return;
        activity = owner;
        pushNative(owner, owner.getSharedPreferences(PREFS, Context.MODE_PRIVATE));
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                View decor = owner.getWindow().getDecorView();
                if (!(decor instanceof ViewGroup)) return;
                ViewGroup root = (ViewGroup) decor;
                if (root.findViewWithTag(TAG) != null) return;

                Button gear = new Button(owner);
                gear.setTag(TAG);
                gear.setText("⚙");
                gear.setContentDescription("ADOFAI 모바일 에디터 설정");
                gear.setTextSize(18f);
                gear.setTextColor(Color.WHITE);
                gear.setAllCaps(false);
                gear.setPadding(0, 0, 0, 0);
                GradientDrawable bg = new GradientDrawable();
                bg.setColor(Color.argb(218, 28, 28, 34));
                bg.setCornerRadius(dp(24));
                gear.setBackground(bg);
                gear.setElevation(dp(5));
                gear.setOnClickListener(new View.OnClickListener() {
                    @Override public void onClick(View view) { show(owner); }
                });

                FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(dp(52), dp(52));
                lp.gravity = Gravity.TOP | Gravity.END;
                lp.topMargin = dp(10);
                lp.rightMargin = dp(10);
                root.addView(gear, lp);
            }
        });
    }

    private static void show(final Activity owner) {
        final SharedPreferences prefs = owner.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        LinearLayout root = new LinearLayout(owner);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(8), dp(18), dp(8));

        TextView note = new TextView(owner);
        note.setText("원본 게임 파일은 유지하고 런타임에서만 모바일 동작을 조정합니다. 높은 FPS는 발열과 배터리 사용량을 늘릴 수 있습니다.");
        note.setTextSize(13f);
        note.setPadding(0, 0, 0, dp(10));
        root.addView(note);

        final Slider uiScale = slider(root, "에디터 UI 크기", percent(prefs.getFloat("ui_scale", DEFAULT_UI_SCALE)), 80, 150);
        final Slider touchScale = slider(root, "UI 터치 판정 범위", percent(prefs.getFloat("touch_scale", DEFAULT_TOUCH_SCALE)), 100, 190);
        final Slider dragScale = slider(root, "에디터 드래그 민감도", percent(prefs.getFloat("drag_scale", DEFAULT_DRAG_SCALE)), 60, 180);
        final CheckBox touchAssist = check(root, "작은 UI 터치 보정", prefs.getBoolean("touch_assist", true));

        TextView fpsTitle = new TextView(owner);
        fpsTitle.setText("프레임 제한  ·  화면 최대 " + Math.round(maxDisplayRefreshRate(owner)) + " Hz");
        fpsTitle.setTextSize(14f);
        fpsTitle.setPadding(0, dp(8), 0, 0);
        root.addView(fpsTitle);

        final Spinner fpsMode = new Spinner(owner);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(owner, android.R.layout.simple_spinner_item, FPS_LABELS);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        fpsMode.setAdapter(adapter);
        fpsMode.setSelection(fpsSelection(prefs.getInt("fps_mode", DEFAULT_FPS_MODE)));
        root.addView(fpsMode);

        final CheckBox fpsUnlock = check(root, "고주사율 / FPS 제한 해제", prefs.getBoolean("fps_unlock", true));
        final CheckBox lowLatency = check(root, "저지연 모드 (VSync 비활성화)", prefs.getBoolean("low_latency", true));

        new AlertDialog.Builder(owner)
                .setTitle("ADOFAI 2.4 모바일 설정")
                .setView(root)
                .setPositiveButton("적용", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        int fpsIndex = Math.max(0, Math.min(FPS_VALUES.length - 1, fpsMode.getSelectedItemPosition()));
                        prefs.edit()
                                .putFloat("ui_scale", uiScale.value() / 100f)
                                .putFloat("touch_scale", touchScale.value() / 100f)
                                .putFloat("drag_scale", dragScale.value() / 100f)
                                .putBoolean("touch_assist", touchAssist.isChecked())
                                .putInt("fps_mode", FPS_VALUES[fpsIndex])
                                .putBoolean("fps_unlock", fpsUnlock.isChecked())
                                .putBoolean("low_latency", lowLatency.isChecked())
                                .apply();
                        pushNative(owner, prefs);
                    }
                })
                .setNeutralButton("기본값", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        prefs.edit().clear().apply();
                        pushNative(owner, prefs);
                    }
                })
                .setNegativeButton("취소", null)
                .show();
    }

    private static void pushNative(final Activity owner, SharedPreferences prefs) {
        try {
            final boolean unlock = prefs.getBoolean("fps_unlock", true);
            final int requestedFps = prefs.getInt("fps_mode", DEFAULT_FPS_MODE);
            final Display.Mode mode = unlock ? selectPreferredMode(owner, requestedFps) : null;
            final int targetFps = unlock ? effectiveTargetFps(owner, mode, requestedFps) : 0;
            if (unlock && mode != null) {
                owner.runOnUiThread(new Runnable() {
                    @Override public void run() { requestDisplayMode(owner, mode); }
                });
            }
            nativeApply(
                    prefs.getFloat("ui_scale", DEFAULT_UI_SCALE),
                    prefs.getFloat("touch_scale", DEFAULT_TOUCH_SCALE),
                    prefs.getFloat("drag_scale", DEFAULT_DRAG_SCALE),
                    prefs.getBoolean("touch_assist", true),
                    targetFps,
                    unlock,
                    prefs.getBoolean("low_latency", true));
        } catch (Throwable ignored) {
        }
    }

    private static Display.Mode selectPreferredMode(Activity owner, int requestedFps) {
        try {
            Display display = owner.getWindowManager().getDefaultDisplay();
            Display.Mode current = display.getMode();
            Display.Mode[] modes = display.getSupportedModes();
            if (current == null || modes == null || modes.length == 0) return null;

            Display.Mode best = current;
            float bestRate = 0f;
            int width = current.getPhysicalWidth();
            int height = current.getPhysicalHeight();
            float ceiling = requestedFps > 0 ? requestedFps + 1.0f : Float.MAX_VALUE;
            for (Display.Mode mode : modes) {
                if (mode.getPhysicalWidth() != width || mode.getPhysicalHeight() != height) continue;
                float rate = mode.getRefreshRate();
                if (rate <= ceiling && rate > bestRate) {
                    best = mode;
                    bestRate = rate;
                }
            }
            return best;
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static int effectiveTargetFps(Activity owner, Display.Mode mode, int requestedFps) {
        if (mode != null) return Math.max(30, Math.round(mode.getRefreshRate()));
        int max = Math.max(30, Math.round(maxDisplayRefreshRate(owner)));
        return requestedFps > 0 ? Math.min(requestedFps, max) : max;
    }

    private static float maxDisplayRefreshRate(Activity owner) {
        try {
            Display display = owner.getWindowManager().getDefaultDisplay();
            Display.Mode current = display.getMode();
            Display.Mode[] modes = display.getSupportedModes();
            float max = current != null ? current.getRefreshRate() : display.getRefreshRate();
            if (current == null || modes == null) return max;
            int width = current.getPhysicalWidth();
            int height = current.getPhysicalHeight();
            for (Display.Mode mode : modes) {
                if (mode.getPhysicalWidth() == width && mode.getPhysicalHeight() == height) {
                    max = Math.max(max, mode.getRefreshRate());
                }
            }
            return max;
        } catch (Throwable ignored) {
            return 60f;
        }
    }

    private static void requestDisplayMode(Activity owner, Display.Mode mode) {
        try {
            WindowManager.LayoutParams params = owner.getWindow().getAttributes();
            if (params.preferredDisplayModeId == mode.getModeId()) return;
            params.preferredDisplayModeId = mode.getModeId();
            owner.getWindow().setAttributes(params);
        } catch (Throwable ignored) {
        }
    }

    private static int fpsSelection(int value) {
        for (int i = 0; i < FPS_VALUES.length; ++i) {
            if (FPS_VALUES[i] == value) return i;
        }
        return 0;
    }

    private static final class Slider {
        final SeekBar seek;
        final int min;
        Slider(SeekBar seek, int min) { this.seek = seek; this.min = min; }
        int value() { return min + seek.getProgress(); }
    }

    private static Slider slider(LinearLayout root, final String label, int initial, final int min, int max) {
        final TextView title = new TextView(root.getContext());
        title.setText(label + "  " + initial + "%");
        title.setTextSize(14f);
        root.addView(title);

        SeekBar seek = new SeekBar(root.getContext());
        seek.setMax(max - min);
        seek.setProgress(Math.max(0, Math.min(max - min, initial - min)));
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                title.setText(label + "  " + (min + progress) + "%");
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        root.addView(seek);
        return new Slider(seek, min);
    }

    private static CheckBox check(LinearLayout root, String label, boolean checked) {
        CheckBox box = new CheckBox(root.getContext());
        box.setText(label);
        box.setChecked(checked);
        root.addView(box);
        return box;
    }

    private static int percent(float value) { return Math.round(value * 100f); }

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

    private static int dp(int value) {
        Context context = activity;
        if (context == null) return value;
        return Math.round(value * context.getResources().getDisplayMetrics().density);
    }
}
