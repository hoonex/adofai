package com.unity3d.player;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.util.Log;
import android.widget.EditText;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** Storage Access Framework + URL bundle bridge for the standalone companion editor. */
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
    private static final Map<String, String> BUNDLE_URI_BY_PATH = new ConcurrentHashMap<String, String>();
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

    /** Historical ADOFAI-style entry point: paste a direct ZIP URL. */
    public static void selectUrlBundle() {
        isDone = false;
        filePath = "";
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            setPath("");
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                final EditText input = new EditText(owner);
                input.setSingleLine(true);
                input.setHint("https://example.com/level.zip");
                input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
                new AlertDialog.Builder(owner)
                        .setTitle("ZIP URL 열기")
                        .setMessage("예전 Open From URL처럼 .zip 링크를 입력하세요. main.adofai와 음악/이미지를 함께 보존합니다.")
                        .setView(input)
                        .setPositiveButton("열기", new DialogInterface.OnClickListener() {
                            @Override public void onClick(DialogInterface dialog, int which) {
                                final String url = input.getText().toString().trim();
                                if (url.length() == 0) {
                                    setPath("");
                                    return;
                                }
                                Toast.makeText(owner, "ZIP 다운로드 중…", Toast.LENGTH_SHORT).show();
                                Thread worker = new Thread(new Runnable() {
                                    @Override public void run() {
                                        try {
                                            final String path = BundleWorkspace.importZipUrl(owner, url);
                                            NAME_BY_PATH.put(path, nameFromUrl(url) + " / " + new File(path).getName());
                                            setPath(path);
                                        } catch (final Throwable error) {
                                            Log.e(TAG, "ZIP URL import failed", error);
                                            owner.runOnUiThread(new Runnable() {
                                                @Override public void run() {
                                                    Toast.makeText(owner, "ZIP URL 열기 실패: " + safeMessage(error), Toast.LENGTH_LONG).show();
                                                }
                                            });
                                            setPath("");
                                        }
                                    }
                                }, "adofai-url-bundle");
                                worker.start();
                            }
                        })
                        .setNegativeButton("취소", new DialogInterface.OnClickListener() {
                            @Override public void onClick(DialogInterface dialog, int which) { setPath(""); }
                        })
                        .setOnCancelListener(new DialogInterface.OnCancelListener() {
                            @Override public void onCancel(DialogInterface dialog) { setPath(""); }
                        })
                        .show();
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
            Toast.makeText(context, "파일 열기 실패: " + safeMessage(error), Toast.LENGTH_LONG).show();
            setPath("");
        }
        return true;
    }

    public static String importUri(Uri uri) throws Exception {
        Activity owner = context;
        if (owner == null) throw new IllegalStateException("No foreground Activity");
        String displayName = queryDisplayName(owner.getContentResolver(), uri, "level.adofai");
        String lower = displayName.toLowerCase(Locale.US);
        InputStream input = owner.getContentResolver().openInputStream(uri);
        if (input == null) throw new IllegalStateException("Document provider returned no input stream");
        try {
            if (lower.endsWith(".zip") || lower.endsWith(".adozip")) {
                String chart = BundleWorkspace.importZip(owner, input, displayName);
                BUNDLE_URI_BY_PATH.put(chart, uri.toString());
                NAME_BY_PATH.put(chart, displayName + " / " + new File(chart).getName());
                return chart;
            }

            String name = ensureExtension(displayName);
            File working = newWorkingFile(owner, name);
            FileOutputStream output = new FileOutputStream(working, false);
            try { copy(input, output); output.getFD().sync(); }
            finally { output.close(); }
            remember(working, uri, name);
            return working.getAbsolutePath();
        } finally {
            input.close();
        }
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

    /** Synchronize direct documents or repack an imported local ZIP bundle. */
    public static boolean syncSavedPath(String localPath) {
        Activity owner = context;
        if (owner == null) return false;

        String bundleEncoded = BUNDLE_URI_BY_PATH.get(localPath);
        if (bundleEncoded != null) {
            try {
                File bundle = BundleWorkspace.packageBundle(owner, localPath);
                return writeFileToUri(owner, bundle, Uri.parse(bundleEncoded));
            } catch (Throwable error) {
                Log.e(TAG, "Could not synchronize ZIP bundle", error);
                return false;
            }
        }

        String encoded = URI_BY_PATH.get(localPath);
        if (encoded == null) return true; // URL-imported bundles stay in the private workspace until exported/handoff.
        File source = new File(localPath);
        if (!source.isFile()) return false;
        try {
            return writeFileToUri(owner, source, Uri.parse(encoded));
        } catch (Throwable error) {
            Log.e(TAG, "Could not synchronize saved chart to SAF document", error);
            return false;
        }
    }

    private static boolean writeFileToUri(Activity owner, File source, Uri uri) throws Exception {
        OutputStream output = owner.getContentResolver().openOutputStream(uri, "wt");
        if (output == null) throw new IllegalStateException("Document provider returned no output stream");
        try {
            FileInputStream input = new FileInputStream(source);
            try { copy(input, output); output.flush(); }
            finally { input.close(); }
        } finally {
            output.close();
        }
        return true;
    }

    public static String displayNameForPath(String path) {
        String value = NAME_BY_PATH.get(path);
        return value != null ? value : new File(path).getName();
    }

    public static void setPath(String path) {
        filePath = path == null ? "" : path;
        isDone = true;
    }

    public static String getFilePath() { return filePath; }
    public static String getFolderPath() { return folderPath; }

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
        while ((read = input.read(buffer)) >= 0) if (read > 0) output.write(buffer, 0, read);
    }

    private static String ensureExtension(String name) {
        String safe = sanitizeName(name);
        return safe.toLowerCase(Locale.US).endsWith(".adofai") ? safe : safe + ".adofai";
    }

    private static String sanitizeName(String name) {
        String safe = name == null ? "level.adofai" : name.trim();
        safe = safe.replace('/', '_').replace('\\', '_');
        if (safe.isEmpty() || safe.equals(".") || safe.equals("..")) safe = "level.adofai";
        return safe;
    }

    private static String nameFromUrl(String url) {
        try {
            Uri uri = Uri.parse(url);
            String last = uri.getLastPathSegment();
            return last == null || last.length() == 0 ? "level.zip" : last;
        } catch (Throwable ignored) {
            return "level.zip";
        }
    }

    private static String safeMessage(Throwable error) {
        String value = error == null ? null : error.getMessage();
        return value == null || value.length() == 0 ? error.getClass().getSimpleName() : value;
    }
}
