package com.unity3d.player;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import java.lang.reflect.Field;

/** Small mobile-specific settings surface injected above the legacy Unity editor. */
public final class V240SettingsOverlay {
    private static final String PREFS = "adofai-v240-mobile";
    private static final String TAG = "adofai-v240-settings-button";
    private static Activity activity;

    private V240SettingsOverlay() {}

    public static void install() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return;
        activity = owner;
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                View decor = owner.getWindow().getDecorView();
                if (!(decor instanceof ViewGroup)) return;
                ViewGroup root = (ViewGroup) decor;
                if (root.findViewWithTag(TAG) != null) return;
                Button gear = new Button(owner);
                gear.setTag(TAG);
                gear.setText("⚙");
                gear.setTextSize(18f);
                gear.setTextColor(Color.WHITE);
                gear.setAllCaps(false);
                gear.setPadding(0, 0, 0, 0);
                GradientDrawable bg = new GradientDrawable();
                bg.setColor(Color.argb(210, 30, 30, 36));
                bg.setCornerRadius(dp(22));
                gear.setBackground(bg);
                gear.setOnClickListener(new View.OnClickListener() {
                    @Override public void onClick(View view) { show(owner); }
                });
                FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(dp(48), dp(48));
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

        final Slider uiScale = slider(root, "에디터 UI 크기", percent(prefs.getFloat("ui_scale", 1.0f)), 70, 140);
        final Slider touchScale = slider(root, "터치 영역 크기", percent(prefs.getFloat("touch_scale", 1.25f)), 80, 180);
        final Slider drag = slider(root, "드래그 민감도", percent(prefs.getFloat("drag_scale", 1.0f)), 50, 160);
        final Slider longPress = slider(root, "롱프레스 시간", prefs.getInt("long_press_ms", 380), 180, 800);
        final CheckBox touchAssist = check(root, "모바일 터치 보정", prefs.getBoolean("touch_assist", true));
        final CheckBox edgeGuard = check(root, "화면 가장자리 제스처 충돌 완화", prefs.getBoolean("edge_guard", true));
        final CheckBox safeArea = check(root, "카메라 홀/내비게이션 Safe Area 적용", prefs.getBoolean("safe_area", true));

        new AlertDialog.Builder(owner)
                .setTitle("ADOFAI 2.4 Mobile 설정")
                .setView(root)
                .setPositiveButton("적용", (dialog, which) -> {
                    prefs.edit()
                            .putFloat("ui_scale", uiScale.value() / 100f)
                            .putFloat("touch_scale", touchScale.value() / 100f)
                            .putFloat("drag_scale", drag.value() / 100f)
                            .putInt("long_press_ms", longPress.value())
                            .putBoolean("touch_assist", touchAssist.isChecked())
                            .putBoolean("edge_guard", edgeGuard.isChecked())
                            .putBoolean("safe_area", safeArea.isChecked())
                            .apply();
                })
                .setNeutralButton("기본값", (dialog, which) -> prefs.edit().clear().apply())
                .setNegativeButton("취소", null)
                .show();
    }

    public static float uiScale() { return prefs().getFloat("ui_scale", 1.0f); }
    public static float touchScale() { return prefs().getFloat("touch_scale", 1.25f); }
    public static float dragScale() { return prefs().getFloat("drag_scale", 1.0f); }
    public static int longPressMs() { return prefs().getInt("long_press_ms", 380); }
    public static boolean touchAssist() { return prefs().getBoolean("touch_assist", true); }
    public static boolean edgeGuard() { return prefs().getBoolean("edge_guard", true); }
    public static boolean safeArea() { return prefs().getBoolean("safe_area", true); }

    private static SharedPreferences prefs() {
        Activity owner = activity != null ? activity : currentActivity();
        if (owner == null) throw new IllegalStateException("no Unity Activity");
        return owner.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private static final class Slider {
        final SeekBar seek;
        final TextView value;
        final int min;
        Slider(SeekBar seek, TextView value, int min) { this.seek = seek; this.value = value; this.min = min; }
        int value() { return min + seek.getProgress(); }
    }

    private static Slider slider(LinearLayout root, String label, int initial, final int min, int max) {
        TextView title = new TextView(root.getContext());
        title.setText(label);
        title.setTextSize(14f);
        root.addView(title);
        final TextView value = new TextView(root.getContext());
        SeekBar seek = new SeekBar(root.getContext());
        seek.setMax(max - min);
        seek.setProgress(Math.max(0, Math.min(max - min, initial - min)));
        value.setText(String.valueOf(initial));
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                value.setText(String.valueOf(min + progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        root.addView(seek);
        root.addView(value);
        return new Slider(seek, value, min);
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
