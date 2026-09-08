package dev.hoonex.adofai.companion;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.unity3d.player.FileSelector;
import com.unity3d.player.MobileEditorShell;
import com.unity3d.player.OfficialGameBridge;

/** Standalone non-root companion editor for user-authored ADOFAI charts and ZIP bundles. */
public final class MainActivity extends Activity {
    private static final long RETURN_PROBE_DELAY_MS = 700L;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        FileSelector.context = this;

        int padding = dp(20);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);
        root.setBackgroundColor(Color.rgb(18, 18, 22));

        TextView title = new TextView(this);
        title.setText("ADOFAI Companion Editor");
        title.setTextColor(Color.WHITE);
        title.setTextSize(26f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "공식 Google Play ADOFAI는 그대로 유지합니다. .adofai 단일 파일뿐 아니라 " +
                "main.adofai + 음악/이미지가 함께 들어 있는 ZIP bundle과 예전 Open From URL 방식의 ZIP 링크도 엽니다. " +
                "'공식 ADOFAI'를 누르면 현재 bundle을 다시 ZIP으로 묶어 기기 내부 127.0.0.1 URL로 제공하고 " +
                "설치된 공식 3.3.1에 URL handoff를 시도합니다. 돌아오면 GET/HEAD 요청 여부도 표시합니다. " +
                "루트, Zygisk, APK 재서명은 사용하지 않습니다.");
        description.setTextColor(Color.rgb(190, 190, 200));
        description.setTextSize(15f);
        description.setPadding(0, dp(14), 0, dp(18));
        root.addView(description);

        Button openEditor = new Button(this);
        openEditor.setText("에디터 / 맵 열기");
        openEditor.setAllCaps(false);
        openEditor.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                FileSelector.context = MainActivity.this;
                MobileEditorShell.openStandalone();
            }
        });
        root.addView(openEditor);

        TextView hint = new TextView(this);
        hint.setText("에디터에서 Open으로 .adofai/.zip을 열거나 ZIP URL에 직접 링크를 붙여넣은 뒤 편집 → 공식 ADOFAI를 누르세요. 공식 앱에서 돌아오면 ZIP URL 요청 진단이 뜹니다.");
        hint.setTextColor(Color.rgb(160, 185, 230));
        hint.setTextSize(13f);
        hint.setPadding(0, dp(14), 0, 0);
        root.addView(hint);

        setContentView(root);

        if (!handleIncomingIntent(getIntent())) {
            root.post(new Runnable() {
                @Override public void run() {
                    MobileEditorShell.openStandalone();
                }
            });
        }
    }

    @Override protected void onResume() {
        super.onResume();
        FileSelector.context = this;
        View decor = getWindow().getDecorView();
        decor.postDelayed(new Runnable() {
            @Override public void run() {
                String diagnostic = OfficialGameBridge.consumeReturnDiagnostic();
                if (diagnostic != null && diagnostic.length() > 0) {
                    MobileEditorShell.showOfficialHandoffDiagnostic(diagnostic);
                }
            }
        }, RETURN_PROBE_DELAY_MS);
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        FileSelector.context = this;
        handleIncomingIntent(intent);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (FileSelector.handleActivityResult(requestCode, resultCode, data)) return;
        super.onActivityResult(requestCode, resultCode, data);
    }

    private boolean handleIncomingIntent(Intent intent) {
        if (intent == null || !Intent.ACTION_VIEW.equals(intent.getAction())) return false;
        final Uri uri = intent.getData();
        if (uri == null) return false;

        new Thread(new Runnable() {
            @Override public void run() {
                try {
                    final String path = FileSelector.importUri(uri);
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            MobileEditorShell.openStandalonePath(path);
                        }
                    });
                } catch (final Throwable error) {
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            Toast.makeText(MainActivity.this,
                                    "파일 열기 실패: " + error.getMessage(), Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        }, "adofai-import").start();
        return true;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
