package dev.hoonex.adofai.gamepatcher;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Full-game 3.3.1 upgrade flow.
 *
 * This activity never ships proprietary game binaries or silently removes the
 * Play-installed game. On device it verifies that install, creates a signed
 * complete split-APK set from its own installed bytes, and requests installation
 * only when the user has explicitly removed the differently signed original.
 */
public final class MainActivity extends Activity {
    private static final int BACKGROUND = Color.rgb(15, 16, 21);
    private static final int PANEL = Color.rgb(27, 29, 38);
    private static final int MUTED = Color.rgb(171, 178, 194);
    private static final int ACCENT = Color.rgb(130, 167, 255);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private volatile boolean busy;
    private boolean checkedForExisting;
    private boolean originalInstalled;
    private PatchPipeline.PreparedSet prepared;
    private TextView stateText;
    private TextView detailText;
    private Button prepareButton;
    private Button uninstallButton;
    private Button installButton;
    private CheckBox backupConsent;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        setContentView(buildUi());
    }

    @Override protected void onResume() {
        super.onResume();
        refreshState();
        if (!checkedForExisting && !busy) {
            checkedForExisting = true;
            recoverPrepared();
        }
    }

    @Override protected void onDestroy() {
        worker.shutdown();
        super.onDestroy();
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BACKGROUND);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(22), dp(32), dp(22), dp(34));
        scroll.addView(root);

        TextView kicker = text("ADOFAI  /  ANDROID", 12, ACCENT, true);
        kicker.setLetterSpacing(0.14f);
        root.addView(kicker);
        TextView title = text("Custom 3.3.1", 31, Color.WHITE, true);
        title.setPadding(0, dp(6), 0, dp(8));
        root.addView(title);
        root.addView(text(
            "설치된 정품 ADOFAI를 기반으로 에디터 기능을 통합한 전체 게임 APK 세트를 생성합니다. " +
            "테스트 에디터만 설치하는 앱이 아닙니다.",
            14, MUTED, false
        ));

        LinearLayout statusPanel = panel();
        statusPanel.setPadding(dp(15), dp(16), dp(15), dp(16));
        LinearLayout.LayoutParams statusParams =
                new LinearLayout.LayoutParams(-1, -2);
        statusParams.topMargin = dp(24);
        root.addView(statusPanel, statusParams);
        statusPanel.addView(text("작업 상태", 12, ACCENT, true));
        stateText = text("설치된 게임 확인 중…", 16, Color.WHITE, true);
        stateText.setPadding(0, dp(9), 0, dp(9));
        statusPanel.addView(stateText);
        detailText = text("", 12, MUTED, false);
        statusPanel.addView(detailText);

        TextView step1 = text("01   전체 게임 빌드", 18, Color.WHITE, true);
        step1.setPadding(0, dp(27), 0, dp(9));
        root.addView(step1);
        root.addView(text(
            "정확히 ADOFAI 3.3.1 (300382)인지 확인한 후 원본 split을 복사합니다. " +
            "에디터 DEX/arm64 네이티브 payload를 추가하고 모든 split을 동일한 기기 키로 서명합니다. " +
            "게임 버전이 달라졌다면 안전하게 중단합니다.",
            13, MUTED, false
        ));
        prepareButton = button("전체 Custom 게임 빌드", true);
        prepareButton.setOnClickListener(v -> prepareGame());
        root.addView(prepareButton);

        TextView step2 = text("02   원본 데이터 보호", 18, Color.WHITE, true);
        step2.setPadding(0, dp(22), 0, dp(9));
        root.addView(step2);
        root.addView(text(
            "Google Play 설치본과 Custom 빌드는 서명 키가 다릅니다. 새 게임을 설치하기 전에 " +
            "원본 제거가 필요할 수 있습니다. 제거 시 게임 내부 데이터와 설정이 손실될 수 있습니다. " +
            "레벨 파일과 설정을 먼저 외부 위치에 백업하세요.",
            13, MUTED, false
        ));
        backupConsent = new CheckBox(this);
        backupConsent.setText("게임 파일과 설정을 외부 저장소에 백업했고, 삭제 위험을 이해했습니다.");
        backupConsent.setTextColor(Color.WHITE);
        backupConsent.setTextSize(13f);
        backupConsent.setPadding(0, dp(8), 0, dp(8));
        backupConsent.setOnCheckedChangeListener((button, checked) -> updateButtons());
        root.addView(backupConsent);
        uninstallButton = button("정품 제거 — 시스템 삭제 화면", false);
        uninstallButton.setOnClickListener(v -> confirmOriginalRemoval());
        root.addView(uninstallButton);

        TextView step3 = text("03   Custom 게임 설치", 18, Color.WHITE, true);
        step3.setPadding(0, dp(22), 0, dp(9));
        root.addView(step3);
        root.addView(text(
            "패치된 전체 split 세트를 무결성 검사 후 Android 시스템 설치 프로그램으로 전달합니다. " +
            "자동 삭제·백그라운드 설치는 하지 않습니다.",
            13, MUTED, false
        ));
        installButton = button("Custom ADOFAI 설치", true);
        installButton.setOnClickListener(v -> installGame());
        root.addView(installButton);

        TextView note = text(
            "지원 범위: Android ARM64, 공식 ADOFAI 3.3.1 / versionCode 300382. " +
            "기기별 에디터 입력과 게임 플레이의 실사용 검증은 별도로 필요합니다.",
            12, MUTED, false
        );
        note.setPadding(0, dp(24), 0, 0);
        root.addView(note);
        updateButtons();
        return scroll;
    }

    private void refreshState() {
        originalInstalled = InstalledGame.isInstalled(this);
        if (!busy) {
            if (prepared != null) {
                setStatus(originalInstalled
                    ? "게임 세트 준비 완료 · 원본 백업 필요"
                    : "설치 준비 완료 · 시스템 설치 가능",
                    "완성된 분할 APK " + prepared.apks.size() + "개. 서명: " +
                    prepared.signerSha256 + "\n원본: " +
                    (originalInstalled ? "설치됨" : "설치되지 않음"), false);
            } else {
                setStatus(originalInstalled ? "원본 게임 감지됨" : "원본 게임 미설치",
                    originalInstalled
                        ? "빌드를 누르면 3.3.1 버전과 원본 split 구성을 검증합니다."
                        : "먼저 공식 ADOFAI 3.3.1을 설치해야 새 빌드를 생성할 수 있습니다.",
                    false);
            }
        }
        updateButtons();
    }

    private void recoverPrepared() {
        if (busy) return;
        busy = true;
        updateButtons();
        worker.execute(() -> {
            PatchPipeline.PreparedSet restored = null;
            String warning = null;
            try {
                restored = PatchPipeline.loadPrepared(this);
            } catch (Exception error) {
                warning = error.getMessage();
            }
            final PatchPipeline.PreparedSet result = restored;
            final String errorText = warning;
            runOnUiThread(() -> {
                if (isFinishing() || isDestroyed()) return;
                prepared = result;
                busy = false;
                refreshState();
                if (errorText != null) {
                    setStatus("이전 패치 파일 검증 실패",
                        errorText + "\n원본이 설치돼 있다면 다시 빌드하세요.", true);
                }
            });
        });
    }

    private void prepareGame() {
        if (busy || !originalInstalled) return;
        busy = true;
        prepared = null;
        backupConsent.setChecked(false);
        setStatus("게임 패치 준비 중…",
            "원본 파일을 수정하지 않고 별도 빌드를 생성합니다. 앱을 종료하지 마세요.", false);
        updateButtons();
        worker.execute(() -> {
            try {
                PatchPipeline.PreparedSet result = PatchPipeline.prepare(
                    this, message -> runOnUiThread(() -> {
                        if (!isFinishing() && !isDestroyed()) {
                            setStatus("Custom 게임 빌드 중", message, false);
                        }
                    })
                );
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    prepared = result;
                    busy = false;
                    refreshState();
                });
            } catch (Exception error) {
                showFailure("빌드 실패", error);
            }
        });
    }

    private void confirmOriginalRemoval() {
        if (busy || prepared == null || !originalInstalled || !backupConsent.isChecked()) return;
        new AlertDialog.Builder(this)
            .setTitle("ADOFAI 원본 제거 확인")
            .setMessage(
                "설치한 정품 앱을 제거하면 게임 내부 저장 데이터가 사라질 수 있습니다. " +
                "백업 여부를 자동 검증할 수 없습니다.\n\n" +
                "다음 단계는 Android 시스템 삭제 화면을 여는 것뿐이며, " +
                "여기서 자동으로 앱을 삭제하지 않습니다."
            )
            .setNegativeButton("취소", null)
            .setPositiveButton("시스템 삭제 화면", (dialog, which) -> {
                Intent intent = new Intent(Intent.ACTION_DELETE,
                    Uri.parse("package:" + InstalledGame.PACKAGE_NAME));
                startActivity(intent);
            }).show();
    }

    private void installGame() {
        if (busy || prepared == null || originalInstalled || !backupConsent.isChecked()) return;
        if (Build.VERSION.SDK_INT >= 26 &&
                !getPackageManager().canRequestPackageInstalls()) {
            setStatus("앱 설치 권한 필요",
                "Android 설정에서 '이 앱에서 허용'을 켠 뒤 다시 설치 버튼을 누르세요.", false);
            startActivity(new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                Uri.parse("package:" + getPackageName())));
            return;
        }
        busy = true;
        updateButtons();
        setStatus("Android 설치 세션 준비 중…", "서명된 모든 APK를 전송합니다.", false);
        worker.execute(() -> {
            try {
                PatchPipeline.PreparedSet checked = PatchPipeline.loadPrepared(this);
                if (checked == null) throw new IllegalStateException("검증된 APK 세트가 없습니다.");
                int sessionId = PreparedInstaller.install(this, checked);
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    busy = false;
                    setStatus("설치 요청 전달 완료", "시스템 설치 화면에서 완료 여부를 확인하세요. " +
                        "세션 ID: " + sessionId, false);
                    updateButtons();
                });
            } catch (Exception error) {
                showFailure("설치 준비 실패", error);
            }
        });
    }

    private void showFailure(String title, Exception error) {
        runOnUiThread(() -> {
            if (isFinishing() || isDestroyed()) return;
            busy = false;
            setStatus(title, String.valueOf(error.getMessage()), true);
            updateButtons();
        });
    }

    private void updateButtons() {
        if (prepareButton == null) return;
        prepareButton.setEnabled(!busy && originalInstalled);
        uninstallButton.setEnabled(!busy && prepared != null &&
            originalInstalled && backupConsent.isChecked());
        installButton.setEnabled(!busy && prepared != null &&
            !originalInstalled && backupConsent.isChecked());
    }

    private void setStatus(String title, String detail, boolean error) {
        stateText.setText(title);
        stateText.setTextColor(error ? Color.rgb(255, 142, 142) : Color.WHITE);
        detailText.setText(detail);
    }

    private LinearLayout panel() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(PANEL);
        bg.setCornerRadius(dp(16));
        layout.setBackground(bg);
        return layout;
    }

    private Button button(String value, boolean primary) {
        Button button = new Button(this);
        button.setText(value);
        button.setTextSize(14);
        button.setAllCaps(false);
        button.setTextColor(primary ? BACKGROUND : Color.WHITE);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(primary ? ACCENT : Color.rgb(53, 57, 70));
        bg.setCornerRadius(dp(12));
        button.setBackground(bg);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(52));
        p.topMargin = dp(10);
        button.setLayoutParams(p);
        return button;
    }

    private TextView text(String value, int size, int color, boolean bold) {
        TextView result = new TextView(this);
        result.setText(value);
        result.setTextColor(color);
        result.setTextSize(size);
        result.setLineSpacing(dp(3), 1f);
        if (bold) result.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return result;
    }

    private int dp(float value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
