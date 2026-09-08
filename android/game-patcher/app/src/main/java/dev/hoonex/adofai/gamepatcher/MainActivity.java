package dev.hoonex.adofai.gamepatcher;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
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
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView status;
    private Button buildButton;
    private Button uninstallButton;
    private Button installButton;
    private volatile PatchPipeline.PreparedSet prepared;
    private volatile boolean busy;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(buildUi());
        refreshState();
    }

    @Override protected void onResume() {
        super.onResume();
        refreshPrepared();
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

        TextView title = text("ADOFAI 3.3.1 Game Patcher", 26, Color.WHITE);
        root.addView(title);
        TextView desc = text(
            "폰에서 설치된 Play판 split을 읽어 Editor DEX/native payload를 주입하고, 모든 split을 동일한 로컬 키로 재서명합니다. 원본 게임은 자동으로 삭제하지 않습니다.",
            14, Color.rgb(190, 190, 200)
        );
        desc.setPadding(0, dp(8), 0, dp(16));
        root.addView(desc);

        Button inspect = button("1. 설치본 확인");
        inspect.setOnClickListener(v -> refreshState());
        root.addView(inspect);

        buildButton = button("2. 패치 게임 만들기");
        buildButton.setOnClickListener(v -> startBuild());
        root.addView(buildButton);

        uninstallButton = button("3. 원본 Play판 제거");
        uninstallButton.setOnClickListener(v -> confirmUninstall());
        root.addView(uninstallButton);

        installButton = button("4. 패치 게임 설치");
        installButton.setOnClickListener(v -> installPrepared());
        root.addView(installButton);

        status = text("상태 확인 중…", 14, Color.rgb(170, 195, 235));
        status.setPadding(0, dp(18), 0, 0);
        root.addView(status);
        return scroll;
    }

    private void refreshState() {
        if (busy) return;
        try {
            InstalledGame game = InstalledGame.inspect(this);
            setStatus("검증된 설치본\n" + game.describe(), false);
            buildButton.setEnabled(true);
        } catch (Throwable error) {
            if (InstalledGame.isInstalled(this)) {
                setStatus(error.getMessage(), true);
            } else {
                setStatus("원본 ADOFAI가 설치되어 있지 않습니다. 이미 패치 세트를 만든 뒤 삭제한 상태라면 4번 설치를 진행하세요.", false);
            }
            buildButton.setEnabled(false);
        }
        refreshPrepared();
    }

    private void refreshPrepared() {
        try {
            prepared = PatchPipeline.loadPrepared(this);
        } catch (Throwable ignored) {
            prepared = null;
        }
        boolean ready = prepared != null;
        uninstallButton.setEnabled(ready && InstalledGame.isInstalled(this) && !busy);
        installButton.setEnabled(ready && !busy);
    }

    private void startBuild() {
        if (busy) return;
        busy = true;
        setControls(false);
        setStatus("패치 준비 시작… 큰 asset split 재서명은 시간이 걸릴 수 있습니다.", false);
        worker.execute(() -> {
            try {
                PatchPipeline.PreparedSet result = PatchPipeline.prepare(this,
                    message -> runOnUiThread(() -> setStatus(message, false)));
                prepared = result;
                runOnUiThread(() -> {
                    busy = false;
                    setStatus(
                        "패치 세트 준비 완료\n" + result.apks.size() + "개 split\nSigner SHA-256: " + result.signerSha256 +
                        "\n\n다음은 원본 게임의 로컬 데이터 백업 여부를 확인한 뒤 3번을 누르세요.",
                        false
                    );
                    setControls(true);
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    busy = false;
                    setStatus("패치 생성 실패: " + error.getMessage(), true);
                    setControls(true);
                });
            }
        });
    }

    private void confirmUninstall() {
        if (prepared == null) {
            setStatus("먼저 패치 세트를 만드세요.", true);
            return;
        }
        new AlertDialog.Builder(this)
            .setTitle("원본 Play판 제거")
            .setMessage(
                "패치판은 로컬 서명이라 Play판 위에 덮어쓸 수 없습니다.\n\n" +
                "게임의 앱 내부 데이터가 필요하면 먼저 백업하세요. 다음 버튼은 Android 시스템 삭제 확인 화면만 엽니다. 이 앱이 자동 삭제하지 않습니다."
            )
            .setPositiveButton("시스템 삭제 화면 열기", (dialog, which) -> {
                Intent intent = new Intent(Intent.ACTION_DELETE, Uri.parse("package:" + InstalledGame.PACKAGE_NAME));
                startActivity(intent);
            })
            .setNegativeButton("취소", null)
            .show();
    }

    private void installPrepared() {
        if (prepared == null) {
            setStatus("준비된 패치 세트가 없습니다.", true);
            return;
        }
        if (InstalledGame.isInstalled(this)) {
            setStatus("서명 충돌 방지를 위해 3번에서 원본 Play판을 먼저 제거해야 합니다.", true);
            return;
        }
        if (Build.VERSION.SDK_INT >= 26 && !getPackageManager().canRequestPackageInstalls()) {
            setStatus("이 패처의 '알 수 없는 앱 설치' 권한을 허용한 뒤 4번을 다시 누르세요.", false);
            Intent settings = new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                Uri.parse("package:" + getPackageName()));
            startActivity(settings);
            return;
        }
        try {
            int session = PreparedInstaller.install(this, prepared);
            setStatus("PackageInstaller 세션 " + session + " 생성 완료. Android 설치 확인 화면을 진행하세요.", false);
        } catch (Throwable error) {
            setStatus("설치 시작 실패: " + error.getMessage(), true);
        }
    }

    private void setControls(boolean enabled) {
        buildButton.setEnabled(enabled && InstalledGame.isInstalled(this));
        refreshPrepared();
    }

    private void setStatus(String message, boolean error) {
        status.setText(message == null ? "" : message);
        status.setTextColor(error ? Color.rgb(245, 120, 120) : Color.rgb(165, 195, 240));
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(17);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        );
        p.setMargins(0, dp(5), 0, dp(5));
        button.setLayoutParams(p);
        button.setMinHeight(dp(52));
        return button;
    }

    private TextView text(String value, int size, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
