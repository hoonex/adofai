package com.unity3d.player;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;

import java.util.ArrayList;

/** Transparent proxy Activity that owns SAF results without modifying UnityPlayerActivity.onActivityResult. */
public final class V240PickerActivity extends Activity {
    private static final int PICK = 7240;
    private int requestId;
    private int mode;
    private String title;
    private String mime;
    private boolean multiselect;
    private boolean launched;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        requestId = getIntent().getIntExtra(V240AndroidBridge.EXTRA_REQUEST_ID, -1);
        mode = getIntent().getIntExtra(V240AndroidBridge.EXTRA_MODE, 0);
        title = getIntent().getStringExtra(V240AndroidBridge.EXTRA_TITLE);
        mime = getIntent().getStringExtra(V240AndroidBridge.EXTRA_MIME);
        multiselect = getIntent().getBooleanExtra(V240AndroidBridge.EXTRA_MULTI, false);
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
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, multiselect);
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
        if (resultCode != RESULT_OK || data == null) {
            V240AndroidBridge.cancel(requestId);
            finish();
            return;
        }

        if (mode == V240AndroidBridge.MODE_OPEN) {
            ArrayList<Uri> uris = new ArrayList<Uri>();
            ClipData clip = data.getClipData();
            if (clip != null) {
                for (int i = 0; i < clip.getItemCount(); i++) {
                    Uri uri = clip.getItemAt(i).getUri();
                    if (uri != null) uris.add(uri);
                }
            }
            if (uris.isEmpty() && data.getData() != null) uris.add(data.getData());
            if (uris.isEmpty()) {
                V240AndroidBridge.cancel(requestId);
            } else {
                V240AndroidBridge.handleOpenResultsAsync(
                        this, requestId, uris.toArray(new Uri[uris.size()]), data.getFlags());
            }
            finish();
            return;
        }

        if (data.getData() == null) {
            V240AndroidBridge.cancel(requestId);
            finish();
            return;
        }
        V240AndroidBridge.handleResultAsync(
                this, requestId, mode, data.getData(), data.getFlags(), title);
        finish();
    }

    @Override public void onBackPressed() {
        V240AndroidBridge.cancel(requestId);
        super.onBackPressed();
    }
}
