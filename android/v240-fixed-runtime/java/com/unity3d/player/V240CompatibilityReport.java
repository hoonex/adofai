package com.unity3d.player;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import java.lang.reflect.Field;

/** Read-only device ABI report. Long-press the existing settings button to copy it. */
final class V240CompatibilityReport {
    private static final String TAG = "ADOFAI.V240CompatReport";
    private static final String SETTINGS_TAG = "adofai-v240-settings-button";

    private V240CompatibilityReport() {}

    static void install() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return;

        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                View decor = owner.getWindow().getDecorView();
                View settings = decor.findViewWithTag(SETTINGS_TAG);
                if (settings == null) return;

                settings.setContentDescription(
                        "ADOFAI 모바일 에디터 설정. 길게 눌러 호환성 리포트 복사");
                settings.setOnLongClickListener(new View.OnLongClickListener() {
                    @Override public boolean onLongClick(View view) {
                        copyToClipboard(owner);
                        return true;
                    }
                });
            }
        });
    }

    private static void copyToClipboard(Activity owner) {
        try {
            String report = nativeGetCompatibilityReport();
            if (report == null || report.length() == 0) {
                report = "V240 compatibility report\nprobe=unavailable\n";
            }

            ClipboardManager clipboard =
                    (ClipboardManager) owner.getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) throw new IllegalStateException("clipboard unavailable");
            clipboard.setPrimaryClip(ClipData.newPlainText("ADOFAI v2.4 compatibility report", report));
            Toast.makeText(owner, "호환성 리포트를 복사했습니다.", Toast.LENGTH_SHORT).show();
        } catch (Throwable error) {
            Log.w(TAG, "failed to copy compatibility report", error);
            Toast.makeText(owner, "호환성 리포트 복사 실패", Toast.LENGTH_SHORT).show();
        }
    }

    private static Activity currentActivity() {
        try {
            Class<?> unityPlayer = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = unityPlayer.getField("currentActivity");
            Object value = field.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable error) {
            Log.w(TAG, "UnityPlayer.currentActivity unavailable", error);
            return null;
        }
    }

    private static native String nativeGetCompatibilityReport();
}
