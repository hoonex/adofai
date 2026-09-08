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

/** Standalone editor + clean-room playable runtime for user-authored ADOFAI charts. */
public final class MainActivity extends Activity {
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
        title.setText("ADOFAI Custom");
        title.setTextColor(Color.WHITE);
        title.setTextSize(26f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "비루트 독립 실행형 ADOFAI 커스텀 에디터 + 플레이어입니다. " +
                "상용 게임 APK나 에셋을 포함하거나 수정하지 않고, 사용자가 연 .adofai 맵을 직접 편집하고 플레이합니다. " +
                "에디터에서 Play를 누르면 현재 맵이 즉시 플레이 화면으로 전환됩니다.");
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
        hint.setText(".adofai 파일을 이 앱으로 열거나, New → Save As로 새 맵을 만든 뒤 Play를 누르세요.");
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
