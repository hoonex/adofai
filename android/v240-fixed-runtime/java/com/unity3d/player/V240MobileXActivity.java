package com.unity3d.player;

import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

/**
 * Launcher activity for the historical 2.4 APK.
 *
 * Its fully-qualified name is intentionally the same UTF-16 length as
 * com.unity3d.player.UnityPlayerActivity. The final APK patch can therefore replace
 * the binary AndroidManifest.xml string-pool entry in place without rebuilding the
 * whole resource table.
 *
 * The activity also owns SAF results directly, so no extra Activity declaration is
 * required in the legacy binary manifest.
 */
public final class V240MobileXActivity extends UnityPlayerActivity {
    private static final String TAG = "ADOFAI.V240Mobile";
    private static final int PICK_BASE = 7240;
    private static final int PICK_SPAN = 20000;
    private static volatile boolean nativeLoaded;

    private int nextPickerCode = PICK_BASE;
    private int pendingRequestId = -1;
    private int pendingMode;
    private String pendingTitle = "";

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        ensureRuntime();
        applyWindowPolicy();
        V240SettingsOverlay.install();
    }

    @Override protected void onResume() {
        super.onResume();
        ensureRuntime();
        applyWindowPolicy();
        V240SettingsOverlay.install();
    }

    void launchPicker(final int requestId, final int mode, final String title, final String mime) {
        runOnUiThread(new Runnable() {
            @Override public void run() {
                if (pendingRequestId > 0) {
                    V240AndroidBridge.fail(requestId,
                            new IllegalStateException("another file picker is already active"));
                    return;
                }
                try {
                    Intent intent;
                    if (mode == V240AndroidBridge.MODE_OPEN) {
                        intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                        intent.addCategory(Intent.CATEGORY_OPENABLE);
                        intent.setType(empty(mime) ? "*/*" : mime);
                    } else if (mode == V240AndroidBridge.MODE_SAVE) {
                        intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                        intent.addCategory(Intent.CATEGORY_OPENABLE);
                        intent.setType(empty(mime) ? "application/octet-stream" : mime);
                        if (!empty(title)) intent.putExtra(Intent.EXTRA_TITLE, title);
                    } else if (mode == V240AndroidBridge.MODE_FOLDER) {
                        intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
                    } else {
                        throw new IllegalArgumentException("unknown picker mode: " + mode);
                    }
                    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                            Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                            Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION |
                            Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);

                    pendingRequestId = requestId;
                    pendingMode = mode;
                    pendingTitle = title == null ? "" : title;
                    int code = nextPickerCode++;
                    if (nextPickerCode >= PICK_BASE + PICK_SPAN) nextPickerCode = PICK_BASE;
                    startActivityForResult(intent, code);
                } catch (Throwable error) {
                    clearPending();
                    V240AndroidBridge.fail(requestId, error);
                }
            }
        });
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode >= PICK_BASE && requestCode < PICK_BASE + PICK_SPAN && pendingRequestId > 0) {
            int id = pendingRequestId;
            int mode = pendingMode;
            String title = pendingTitle;
            clearPending();

            if (resultCode != RESULT_OK || data == null || data.getData() == null) {
                V240AndroidBridge.cancel(id);
                return;
            }
            Uri uri = data.getData();
            int flags = data.getFlags();
            if (mode == V240AndroidBridge.MODE_OPEN) {
                V240AndroidBridge.handleOpen(this, id, uri, flags);
            } else if (mode == V240AndroidBridge.MODE_SAVE) {
                V240AndroidBridge.handleSave(this, id, uri, flags, title);
            } else if (mode == V240AndroidBridge.MODE_FOLDER) {
                V240AndroidBridge.handleFolder(this, id, uri, flags);
            } else {
                V240AndroidBridge.fail(id, new IllegalStateException("lost picker mode"));
            }
            return;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void clearPending() {
        pendingRequestId = -1;
        pendingMode = 0;
        pendingTitle = "";
    }

    private static synchronized void ensureRuntime() {
        if (nativeLoaded) return;
        try {
            System.loadLibrary("v240fix");
            nativeLoaded = true;
            Log.i(TAG, "2.4 fixed native runtime loaded");
        } catch (Throwable error) {
            Log.e(TAG, "2.4 fixed native runtime failed to load", error);
        }
    }

    private void applyWindowPolicy() {
        try {
            if (Build.VERSION.SDK_INT >= 28) {
                WindowManager.LayoutParams params = getWindow().getAttributes();
                params.layoutInDisplayCutoutMode =
                        WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
                getWindow().setAttributes(params);
            }
        } catch (Throwable error) {
            Log.w(TAG, "window compatibility policy failed", error);
        }
    }

    private static boolean empty(String value) {
        return value == null || value.trim().length() == 0;
    }
}
