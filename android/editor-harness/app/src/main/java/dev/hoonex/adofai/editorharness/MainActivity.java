package dev.hoonex.adofai.editorharness;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.unity3d.player.FileSelector;
import com.unity3d.player.MobileEditorShell;

/**
 * Standalone host for exercising the exact Android-native mobile editor shell
 * without modifying or resigning the installed ADOFAI game package.
 *
 * Preview is intentionally not a success criterion here: the standalone harness
 * has no injected libOctober/current-game runtime bridge. The shell catches that
 * missing native bridge and reports Preview as unavailable instead of crashing.
 */
public final class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int padding = dp(20);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);
        root.setBackgroundColor(Color.rgb(18, 18, 22));

        TextView title = new TextView(this);
        title.setText("ADOFAI Editor Harness");
        title.setTextColor(Color.WHITE);
        title.setTextSize(23f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "이 APK는 게임을 수정하지 않고 실제 모바일 편집기 셸의 Open / Save As / Edit / Save / Reopen, " +
                "터치·스크롤·키보드 동작을 먼저 검증합니다. 처음 파일을 열 때 Android 11+ 파일 접근 권한을 허용해야 할 수 있습니다.\n\n" +
                "Preview는 여기서는 의도적으로 게임 런타임에 연결되지 않습니다. Preview 오류는 이 Harness에서 정상적인 미검증 상태입니다.");
        description.setTextColor(Color.rgb(190, 190, 200));
        description.setTextSize(15f);
        description.setPadding(0, dp(14), 0, dp(18));
        root.addView(description);

        Button showEditor = new Button(this);
        showEditor.setText("Editor 버튼 다시 표시");
        showEditor.setAllCaps(false);
        showEditor.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                installEditorLauncher();
            }
        });
        root.addView(showEditor);

        TextView hint = new TextView(this);
        hint.setText("오른쪽 위의 Editor 버튼을 눌러 시작하세요. 원본 맵 대신 Save As 복사본으로 테스트하는 것을 권장합니다.");
        hint.setTextColor(Color.rgb(160, 185, 230));
        hint.setTextSize(13f);
        hint.setPadding(0, dp(14), 0, 0);
        root.addView(hint);

        setContentView(root);
        installEditorLauncher();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // The shared raw-path picker first consults this explicit Activity. This
        // avoids depending on UnityPlayer reflection inside the standalone host.
        FileSelector.context = this;
    }

    private void installEditorLauncher() {
        FileSelector.context = this;
        MobileEditorShell.installLauncher();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
