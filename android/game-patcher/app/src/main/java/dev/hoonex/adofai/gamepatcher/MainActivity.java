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
import android.os.StatFs;
import android.database.Cursor;
import android.provider.OpenableColumns;
import android.provider.Settings;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
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
    private static final int SOURCE_APK_PICKER = 4401;
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
    private Button sourceApkButton;
    private File selectedSourceApk;
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
            "설치된 ADOFAI 3.3.1 또는 사용자가 직접 선택한 완전한 단일 APK를 " +
            "기반으로 에디터가 통합된 게임을 별도 생성합니다. 실제 기기 검증 전의 실험용 빌더입니다.",
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
        root.addView(text(
            "3.3.1이 휴대폰에 설치되지 않아도 Drive/파일 앱에서 단일 ARM64 APK를 선택할 수 있습니다. " +
            "선택한 파일은 이 앱의 전용 저장소로만 복사하며, 발행자 서명을 검증하지 않으므로 " +
            "출처가 불명확한 APK는 사용하지 마세요.",
            12, MUTED, false
        ));
        sourceApkButton = button("Drive / 파일에서 3.3.1 APK 선택", false);
        sourceApkButton.setOnClickListener(v -> chooseSourceApk());
        root.addView(sourceApkButton);
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
                setStatus(selectedSourceApk != null ? "선택한 3.3.1 APK 준비됨" :
                    (originalInstalled ? "원본 게임 감지됨" : "원본 게임 미설치"),
                    selectedSourceApk != null
                        ? "Drive에서 선택한 단일 APK로 빌드할 수 있습니다. 출처·서명은 확인되지 않았습니다."
                        : (originalInstalled
                            ? "빌드를 누르면 설치된 3.3.1 및 원본 split을 검증합니다."
                            : "휴대폰에 3.3.1을 설치하거나 Drive에서 단일 APK를 선택하세요."),
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
        if (busy || (!originalInstalled && selectedSourceApk == null)) return;
        final File selectedInput = selectedSourceApk;
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
                    }), selectedInput
                );
                if (selectedInput != null) selectedInput.delete();
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    prepared = result;
                    selectedSourceApk = null;
                    busy = false;
                    refreshState();
                });
            } catch (Exception error) {
                showFailure("빌드 실패", error);
            }
        });
    }

    private void chooseSourceApk() {
        if (busy) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, SOURCE_APK_PICKER);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != SOURCE_APK_PICKER || resultCode != RESULT_OK ||
                data == null || data.getData() == null || busy) return;
        final Uri source = data.getData();
        busy = true;
        setStatus("선택한 APK 복사 중…", "원본 Drive 파일을 변경하지 않습니다.", false);
        updateButtons();
        worker.execute(() -> {
            try {
                File root = getExternalFilesDir(null);
                if (root == null) throw new IllegalStateException("앱 전용 저장소가 없습니다.");
                File dir = new File(root, "selected-source");
                if (!dir.isDirectory() && !dir.mkdirs()) {
                    throw new IllegalStateException("APK 작업 폴더를 만들 수 없습니다.");
                }
                long announced = -1L;
                try (Cursor cursor = getContentResolver().query(
                        source, new String[]{OpenableColumns.SIZE}, null, null, null)) {
                    if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                        announced = cursor.getLong(0);
                    }
                }
                final long maximum = 6L * 1024L * 1024L * 1024L;
                if (announced > maximum) throw new IllegalStateException("APK 최대 6GiB");
                long free = new StatFs(root.getAbsolutePath()).getAvailableBytes();
                if (announced > 0 && (announced > (free - 768L * 1024 * 1024) / 3)) {
                    throw new IllegalStateException(
                            "선택한 APK를 패치하기에 저장공간이 부족합니다. " +
                            "게임 크기의 약 3배 이상 빈 공간이 필요합니다.");
                }
                File temp = new File(dir, "incoming.apk");
                try {
                    long copied = 0;
                    try (InputStream in = getContentResolver().openInputStream(source);
                         FileOutputStream out = new FileOutputStream(temp, false)) {
                        if (in == null) throw new IllegalStateException("Drive 파일을 열 수 없습니다.");
                        byte[] chunk = new byte[1024 * 1024];
                        int count;
                        while ((count = in.read(chunk)) != -1) {
                            copied += count;
                            if (copied > maximum) throw new IllegalStateException("APK 최대 6GiB 초과");
                            out.write(chunk, 0, count);
                        }
                        out.getFD().sync();
                    }
                    if (announced > 0 && announced != copied) {
                        throw new IllegalStateException("APK 파일이 불완전하게 복사됐습니다.");
                    }
                    InstalledGame.inspectStandalone(this, temp);
                    File dest = new File(dir, "source.apk");
                    if (dest.exists() && !dest.delete()) {
                        throw new IllegalStateException("이전 선택 파일을 교체할 수 없습니다.");
                    }
                    if (!temp.renameTo(dest)) {
                        throw new IllegalStateException("선택 APK의 이름 변경에 실패했습니다.");
                    }
                    runOnUiThread(() -> {
                        if (isFinishing() || isDestroyed()) return;
                        selectedSourceApk = dest;
                        prepared = null;
                        busy = false;
                        refreshState();
                    });
                } finally {
                    if (temp.exists()) temp.delete();
                }
            } catch (Exception error) {
                showFailure("APK 선택 실패", error);
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
        prepareButton.setEnabled(!busy && (originalInstalled || selectedSourceApk != null));
        if (sourceApkButton != null) sourceApkButton.setEnabled(!busy);
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
