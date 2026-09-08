package com.unity3d.player;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;

/** Transparent proxy Activity that owns SAF results without modifying UnityPlayerActivity.onActivityResult. */
public final class V240PickerActivity extends Activity {
    private static final int PICK = 7240;
    private int requestId;
    private int mode;
    private String title;
    private String mime;
    private boolean launched;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        requestId = getIntent().getIntExtra(V240AndroidBridge.EXTRA_REQUEST_ID, -1);
        mode = getIntent().getIntExtra(V240AndroidBridge.EXTRA_MODE, 0);
        title = getIntent().getStringExtra(V240AndroidBridge.EXTRA_TITLE);
        mime = getIntent().getStringExtra(V240AndroidBridge.EXTRA_MIME);
        if (state != null) launched = state.getBoolean("launched", false);
        if (!launched) launchPicker();
    }

    @Override protected void onSaveInstanceState(Bundle outState) {
        outState.putBoolean("launched", launched);
        super.onSaveInstanceState(outState);
    }

    private void launchPicker() {
        if (requestId <= 0) {
            finish();
            return;
        }
        try {
            Intent intent;
            if (mode == V240AndroidBridge.MODE_OPEN) {
                intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType(mime == null || mime.length() == 0 ? "*/*" : mime);
            } else if (mode == V240AndroidBridge.MODE_SAVE) {
                intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType(mime == null || mime.length() == 0 ? "application/octet-stream" : mime);
                if (title != null && title.length() > 0) intent.putExtra(Intent.EXTRA_TITLE, title);
            } else if (mode == V240AndroidBridge.MODE_FOLDER) {
                intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            } else {
                throw new IllegalArgumentException("unknown picker mode: " + mode);
            }
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION |
                    Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
            launched = true;
            startActivityForResult(intent, PICK);
        } catch (Throwable error) {
            V240AndroidBridge.fail(requestId, error);
            finish();
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK) return;
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            V240AndroidBridge.cancel(requestId);
            finish();
            return;
        }
        Uri uri = data.getData();
        int flags = data.getFlags();
        if (mode == V240AndroidBridge.MODE_OPEN) {
            V240AndroidBridge.handleOpen(this, requestId, uri, flags);
        } else if (mode == V240AndroidBridge.MODE_SAVE) {
            V240AndroidBridge.handleSave(this, requestId, uri, flags, title);
        } else {
            V240AndroidBridge.handleFolder(this, requestId, uri, flags);
        }
        finish();
    }

    @Override public void onBackPressed() {
        V240AndroidBridge.cancel(requestId);
        super.onBackPressed();
    }
}
