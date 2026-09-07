package dev.hoonex.adofai.exporter;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class CompatibilityActivity extends Activity {
    private static final int REQUEST_SAVE_REPORT = 2001;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView statusView;
    private ProgressBar progressBar;
    private Button analyzeButton;
    private Button runtimeButton;
    private File pendingReport;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int padding = dp(24);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("ADOFAI 3.x Compatibility Inspector");
        title.setTextSize(24f);
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        TextView description = new TextView(this);
        description.setText("설치된 ADOFAI를 직접 읽어 IL2CPP 메타데이터, libOctober 로드 부트스트랩, split APK 구조, 저장소 권한을 검사합니다. 게임 파일 전체를 복사하지 않습니다.");
        description.setTextSize(15f);
        LinearLayout.LayoutParams descriptionParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        descriptionParams.topMargin = dp(16);
        root.addView(description, descriptionParams);

        analyzeButton = new Button(this);
        analyzeButton.setText("3.3.x 호환성 보고서 만들기");
        analyzeButton.setAllCaps(false);
        LinearLayout.LayoutParams analyzeParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        analyzeParams.topMargin = dp(24);
        root.addView(analyzeButton, analyzeParams);

        runtimeButton = new Button(this);
        runtimeButton.setText("런타임 ZIP 추출 열기");
        runtimeButton.setAllCaps(false);
        LinearLayout.LayoutParams runtimeParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        runtimeParams.topMargin = dp(12);
        root.addView(runtimeButton, runtimeParams);

        progressBar = new ProgressBar(this);
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        progressParams.topMargin = dp(18);
        root.addView(progressBar, progressParams);

        statusView = new TextView(this);
        statusView.setText("준비됨. 호환성 보고서는 보통 수십 KB 이하입니다.");
        statusView.setTextSize(14f);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.topMargin = dp(16);
        root.addView(statusView, statusParams);

        setContentView(root);

        analyzeButton.setOnClickListener(v -> beginAnalysis());
        runtimeButton.setOnClickListener(v -> startActivity(new Intent(this, MainActivity.class)));
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void beginAnalysis() {
        analyzeButton.setEnabled(false);
        runtimeButton.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("3.3.x 설치 구조와 IL2CPP 메타데이터 검사 중…");

        executor.execute(() -> {
            try {
                CompatibilityInspector.ReportResult result =
                        new CompatibilityInspector(this).buildReport();
                pendingReport = result.file;
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    statusView.setText("분석 완료: " + result.classification + "\nJSON 저장 위치를 선택하세요.");
                    launchSavePicker(result.suggestedName);
                });
            } catch (Exception e) {
                showError("분석 실패: " + e.getClass().getSimpleName() + ": " + safeMessage(e));
            }
        });
    }

    private void launchSavePicker(String suggestedName) {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        intent.putExtra(Intent.EXTRA_TITLE, suggestedName);
        startActivityForResult(intent, REQUEST_SAVE_REPORT);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_SAVE_REPORT) {
            return;
        }
        if (resultCode != RESULT_OK || data == null || data.getData() == null || pendingReport == null) {
            statusView.setText("보고서 저장이 취소되었습니다.");
            analyzeButton.setEnabled(true);
            runtimeButton.setEnabled(true);
            return;
        }

        Uri target = data.getData();
        File source = pendingReport;
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("JSON 저장 중…");
        executor.execute(() -> {
            try (InputStream in = new BufferedInputStream(new FileInputStream(source));
                 OutputStream rawOut = getContentResolver().openOutputStream(target, "w")) {
                if (rawOut == null) {
                    throw new IllegalStateException("저장 스트림을 열 수 없습니다.");
                }
                try (OutputStream out = new BufferedOutputStream(rawOut)) {
                    byte[] buffer = new byte[64 * 1024];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                    out.flush();
                }
                source.delete();
                pendingReport = null;
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    statusView.setText("저장 완료. 생성된 JSON 파일만 ChatGPT에 올리면 됩니다.");
                    analyzeButton.setEnabled(true);
                    runtimeButton.setEnabled(true);
                });
            } catch (Exception e) {
                showError("저장 실패: " + e.getClass().getSimpleName() + ": " + safeMessage(e));
            }
        });
    }

    private void showError(String message) {
        runOnUiThread(() -> {
            progressBar.setVisibility(View.GONE);
            analyzeButton.setEnabled(true);
            runtimeButton.setEnabled(true);
            statusView.setText(message);
        });
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? "원인 메시지 없음" : message;
    }
}
