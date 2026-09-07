package com.unity3d.player;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Storage Access Framework bridge used by the standalone companion editor.
 *
 * The editor shell still edits an ordinary filesystem path so its atomic temp +
 * rename save semantics remain unchanged. SAF documents are mirrored into an
 * app-private working file, then synchronized back to the selected document only
 * after the shell's local save succeeds.
 */
public final class FileSelector {
    private static final String TAG = "ADOFAI.CompanionSAF";
    private static final int REQUEST_OPEN = 4101;
    private static final int REQUEST_SAVE = 4102;
    private static final int REQUEST_FOLDER = 4103;

    public static Activity context;
    public static volatile boolean isDone = false;

    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile String pendingSaveName = "level.adofai";
    private static final Map<String, String> URI_BY_PATH = new ConcurrentHashMap<String, String>();
    private static final Map<String, String> NAME_BY_PATH = new ConcurrentHashMap<String, String>();

    private FileSelector() {}

    public static void selectFile(String ignoredFileType) {
        isDone = false;
        filePath = "";
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            setPath("");
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    intent.setType("*/*");
                    owner.startActivityForResult(intent, REQUEST_OPEN);
                } catch (Throwable error) {
                    Log.e(TAG, "Could not launch open-document picker", error);
                    setPath("");
                }
            }
        });
    }

    public static void saveAs(String name) {
        isDone = false;
        filePath = "";
        pendingSaveName = ensureExtension(name == null || name.trim().isEmpty() ? "level.adofai" : name.trim());
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            setPath("");
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    intent.setType("application/json");
                    intent.putExtra(Intent.EXTRA_TITLE, pendingSaveName);
                    owner.startActivityForResult(intent, REQUEST_SAVE);
                } catch (Throwable error) {
                    Log.e(TAG, "Could not launch create-document picker", error);
                    setPath("");
                }
            }
        });
    }

    public static void selectFolder() {
        isDone = false;
        folderPath = "";
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            isDone = true;
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    owner.startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQUEST_FOLDER);
                } catch (Throwable error) {
                    Log.e(TAG, "Could not launch folder picker", error);
                    isDone = true;
                }
            }
        });
    }

    public static boolean handleActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != REQUEST_OPEN && requestCode != REQUEST_SAVE && requestCode != REQUEST_FOLDER) return false;

        if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
            if (requestCode == REQUEST_FOLDER) {
                folderPath = "";
                isDone = true;
            } else {
                setPath("");
            }
            return true;
        }

        Uri uri = data.getData();
        takePersistablePermission(uri, data.getFlags());
        try {
            if (requestCode == REQUEST_FOLDER) {
                folderPath = uri.toString();
                isDone = true;
                return true;
            }

            String working = requestCode == REQUEST_OPEN
                    ? importUri(uri)
                    : prepareSaveDocument(uri, pendingSaveName);
            setPath(working);
        } catch (Throwable error) {
            Log.e(TAG, "SAF result handling failed", error);
            setPath("");
        }
        return true;
    }

    public static String importUri(Uri uri) throws Exception {
        Activity owner = context;
        if (owner == null) throw new IllegalStateException("No foreground Activity");
        String name = ensureExtension(queryDisplayName(owner.getContentResolver(), uri, "level.adofai"));
        File working = newWorkingFile(owner, name);
        InputStream input = owner.getContentResolver().openInputStream(uri);
        if (input == null) throw new IllegalStateException("Document provider returned no input stream");
        try {
            FileOutputStream output = new FileOutputStream(working, false);
            try { copy(input, output); output.getFD().sync(); }
            finally { output.close(); }
        } finally {
            input.close();
        }
        remember(working, uri, name);
        return working.getAbsolutePath();
    }

    private static String prepareSaveDocument(Uri uri, String fallbackName) throws Exception {
        Activity owner = context;
        if (owner == null) throw new IllegalStateException("No foreground Activity");
        String name = ensureExtension(queryDisplayName(owner.getContentResolver(), uri, fallbackName));
        File working = newWorkingFile(owner, name);
        if (!working.createNewFile() && !working.isFile()) {
            throw new IllegalStateException("Could not create working chart");
        }
        remember(working, uri, name);
        return working.getAbsolutePath();
    }

    public static boolean syncSavedPath(String localPath) {
        String encoded = URI_BY_PATH.get(localPath);
        if (encoded == null) return true;
        Activity owner = context;
        if (owner == null) return false;
        File source = new File(localPath);
        if (!source.isFile()) return false;
        try {
            OutputStream output = owner.getContentResolver().openOutputStream(Uri.parse(encoded), "wt");
            if (output == null) throw new IllegalStateException("Document provider returned no output stream");
            try {
                FileInputStream input = new FileInputStream(source);
                try { copy(input, output); output.flush(); }
                finally { input.close(); }
            } finally {
                output.close();
            }
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Could not synchronize saved chart to SAF document", error);
            return false;
        }
    }

    public static boolean openInAdofaiOrShare(String localPath) {
        Activity owner = context;
        String encoded = URI_BY_PATH.get(localPath);
        if (owner == null || encoded == null) return false;
        if (!syncSavedPath(localPath)) return false;

        Uri uri = Uri.parse(encoded);
        Intent direct = new Intent(Intent.ACTION_VIEW);
        direct.setPackage("com.fizzd.connectedworlds");
        direct.setDataAndType(uri, "application/json");
        direct.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try {
            owner.startActivity(direct);
            return true;
        } catch (ActivityNotFoundException ignored) {
            Log.i(TAG, "Official ADOFAI does not expose a matching VIEW activity; falling back to Android share sheet");
        } catch (SecurityException ignored) {
            Log.i(TAG, "Direct ADOFAI VIEW was rejected; falling back to Android share sheet");
        }

        try {
            Intent share = new Intent(Intent.ACTION_SEND);
            share.setType("application/json");
            share.putExtra(Intent.EXTRA_STREAM, uri);
            share.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            owner.startActivity(Intent.createChooser(share, "ADOFAI chart"));
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Could not open Android share sheet", error);
            return false;
        }
    }

    public static String displayNameForPath(String path) {
        String value = NAME_BY_PATH.get(path);
        return value != null ? value : new File(path).getName();
    }

    public static void setPath(String path) {
        filePath = path == null ? "" : path;
        isDone = true;
    }

    public static String getFilePath() {
        return filePath;
    }

    public static String getFolderPath() {
        return folderPath;
    }

    private static void remember(File working, Uri uri, String name) {
        URI_BY_PATH.put(working.getAbsolutePath(), uri.toString());
        NAME_BY_PATH.put(working.getAbsolutePath(), name);
    }

    private static File newWorkingFile(Activity owner, String displayName) {
        File root = new File(owner.getFilesDir(), "working");
        File session = new File(root, "doc-" + UUID.randomUUID().toString());
        if (!session.mkdirs() && !session.isDirectory()) {
            throw new IllegalStateException("Could not create private working directory");
        }
        return new File(session, sanitizeName(displayName));
    }

    private static String queryDisplayName(ContentResolver resolver, Uri uri, String fallback) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String value = cursor.getString(index);
                    if (value != null && !value.trim().isEmpty()) return value.trim();
                }
            }
        } catch (Throwable error) {
            Log.w(TAG, "Could not read SAF display name", error);
        } finally {
            if (cursor != null) cursor.close();
        }
        return fallback == null ? "level.adofai" : fallback;
    }

    private static void takePersistablePermission(Uri uri, int resultFlags) {
        Activity owner = context;
        if (owner == null) return;
        int flags = resultFlags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if (flags == 0) return;
        try {
            owner.getContentResolver().takePersistableUriPermission(uri, flags);
        } catch (SecurityException ignored) {
            // Some providers grant session access without persistable permission.
        }
    }

    private static void copy(InputStream input, OutputStream output) throws Exception {
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = input.read(buffer)) >= 0) {
            if (read > 0) output.write(buffer, 0, read);
        }
    }

    private static String ensureExtension(String name) {
        String safe = sanitizeName(name);
        return safe.toLowerCase(java.util.Locale.US).endsWith(".adofai") ? safe : safe + ".adofai";
    }

    private static String sanitizeName(String name) {
        String safe = name == null ? "level.adofai" : name.trim();
        safe = safe.replace('/', '_').replace('\\', '_');
        if (safe.isEmpty() || safe.equals(".") || safe.equals("..")) safe = "level.adofai";
        return safe;
    }
}
