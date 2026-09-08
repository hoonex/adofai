package dev.hoonex.adofai.gamepatcher;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_APK = 240;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView status;
    private Button chooseButton;
    private volatile boolean busy;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(buildUi());
    }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(22), dp(20), dp(30));
        scroll.addView(root);

        TextView title = text("ADOFAI 2.4.0 Custom Bugfix", 25, Color.WHITE);
        root.addView(title);

        TextView desc = text(
            "2023년 V2.4.0 Custom APK의 기존 에디터/ZIP URL 로더/native runtime을 그대로 보존합니다.\n\n" +
            "적용: Android 저장소 권한 호환성. 기존 FileSelector가 들어있는 변형이면 그 파일선택기의 Activity/저장소/취소 처리도 교체합니다. " +
            "libOctober.so나 classes2.dex는 원본 필수 항목이 아닙니다.",
            14, Color.rgb(190, 190, 200)
        );
        desc.setPadding(0, dp(10), 0, dp(18));
        root.addView(desc);

        chooseButton = button("V2.4.0 CUSTOM APK 선택 → 수정본 만들기");
        chooseButton.setOnClickListener(v -> chooseSource());
        root.addView(chooseButton);

        Button permission = button("ALL FILES ACCESS 설정 열기 (필요 시)");
        permission.setOnClickListener(v -> {
            try {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } catch (Throwable ignored) {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            }
        });
        root.addView(permission);

        status = text(
            "원본 APK를 선택하세요. 수정본은 Downloads/ADOFAI/ADOFAI-2.4.0-Custom-Bugfix.apk 로 저장됩니다.",
            14, Color.rgb(165, 195, 240)
        );
        status.setPadding(0, dp(18), 0, 0);
        root.addView(status);
        return scroll;
    }

    private void chooseSource() {
        if (busy) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/vnd.android.package-archive");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {
            "application/vnd.android.package-archive", "application/octet-stream", "application/zip"
        });
        startActivityForResult(intent, PICK_APK);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_APK || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Throwable ignored) {
        }
        startPatch(uri);
    }

    private void startPatch(Uri uri) {
        if (busy) return;
        busy = true;
        chooseButton.setEnabled(false);
        setStatus("2.4.0 APK 검사 시작…", false);
        worker.execute(() -> {
            try {
                V240PatchPipeline.Result result = V240PatchPipeline.patch(
                    this,
                    uri,
                    message -> runOnUiThread(() -> setStatus(message, false))
                );
                runOnUiThread(() -> {
                    busy = false;
                    chooseButton.setEnabled(true);
                    setStatus(
                        "수정 완료\n\n" +
                        "파일: " + V240PatchPipeline.OUTPUT_NAME + "\n" +
                        "저장 위치: Downloads/ADOFAI\n" +
                        "Package: " + result.packageName + "\n" +
                        "파일선택기 처리: " + result.pickerPatchMode + "\n" +
                        "기존 native/URL loader: 보존\n" +
                        "새 signer SHA-256: " + result.signerSha256 + "\n" +
                        "크기: " + formatBytes(result.outputBytes),
                        false
                    );
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    busy = false;
                    chooseButton.setEnabled(true);
                    setStatus("수정 실패: " + error.getMessage(), true);
                });
            }
        });
    }

    private void setStatus(String message, boolean error) {
        status.setText(message == null ? "" : message);
        status.setTextColor(error ? Color.rgb(245, 120, 120) : Color.rgb(165, 195, 240));
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(16);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        );
        p.setMargins(0, dp(5), 0, dp(5));
        button.setLayoutParams(p);
        button.setMinHeight(dp(54));
        return button;
    }

    private TextView text(String value, int size, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private String formatBytes(long bytes) {
        double mib = bytes / (1024.0 * 1024.0);
        return String.format(java.util.Locale.US, "%.1f MiB", mib);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
