package dev.hoonex.adofai.gamepatcher;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.widget.Toast;

public final class InstallResultReceiver extends BroadcastReceiver {
    static final String ACTION_INSTALL_RESULT = "dev.hoonex.adofai.gamepatcher.INSTALL_RESULT";

    @Override public void onReceive(Context context, Intent intent) {
        int status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE);
        String message = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE);
        if (status == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            Intent confirm = intent.getParcelableExtra(Intent.EXTRA_INTENT);
            if (confirm != null) {
                confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(confirm);
            }
            return;
        }
        if (status == PackageInstaller.STATUS_SUCCESS) {
            Toast.makeText(context, "ADOFAI 3.3.1 Editor 패치 설치 완료", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(context, "패치 설치 실패: " + (message == null ? status : message), Toast.LENGTH_LONG).show();
        }
    }
}
