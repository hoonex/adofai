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

/** Standalone ADOFAI chart editor that coexists with the official Play build. */
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
        title.setText("ADOFAI Companion Editor");
        title.setTextColor(Color.WHITE);
        title.setTextSize(24f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "공식 Play판 ADOFAI는 그대로 유지합니다. 이 앱은 .adofai 파일만 독립적으로 열고 편집·저장하며, " +
                "Android 시스템 파일 선택기(SAF)를 사용합니다. 저장 후에는 'ADOFAI / 공유'로 공식 게임 열기를 시도하고, " +
                "지원되지 않으면 Android 공유 화면으로 전환합니다.");
        description.setTextColor(Color.rgb(190, 190, 200));
        description.setTextSize(15f);
        description.setPadding(0, dp(14), 0, dp(18));
        root.addView(description);

        Button openEditor = new Button(this);
        openEditor.setText("에디터 열기");
        openEditor.setAllCaps(false);
        openEditor.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                FileSelector.context = MainActivity.this;
                MobileEditorShell.openStandalone();
            }
        });
        root.addView(openEditor);

        TextView hint = new TextView(this);
        hint.setText("파일 앱에서 .adofai를 이 앱으로 열어도 됩니다. 새 맵은 에디터의 New → Save As 순서로 저장하세요.");
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

    @Override
    protected void onResume() {
        super.onResume();
        FileSelector.context = this;
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        FileSelector.context = this;
        handleIncomingIntent(intent);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
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
