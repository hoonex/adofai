package com.unity3d.player;

import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Permissionless Storage Access Framework bridge for the in-game Zygisk editor.
 * Open selects a map folder, mirrors the complete tree into app-private storage,
 * and returns the mirrored .adofai path so relative song/image assets remain usable
 * by the official game runtime. Saves are synchronized back to the original SAF
 * document after the editor's local atomic write succeeds.
 */
public final class FileSelector {
    private static final String TAG = "ADOFAI.ZygiskSAF";
    private static final String FRAGMENT_TAG = "adofai-zygisk-saf";
    private static final int REQUEST_OPEN_TREE = 7201;
    private static final int REQUEST_SAVE = 7202;
    private static final int REQUEST_FOLDER = 7203;
    private static final int MAX_FILES = 5000;
    private static final long MAX_BYTES = 1024L * 1024L * 1024L;

    public static volatile Activity context;
    public static volatile boolean isDone;

    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile String pendingSaveName = "level.adofai";
    private static final Map<String, String> URI_BY_PATH = new ConcurrentHashMap<String, String>();
    private static final Map<String, String> NAME_BY_PATH = new ConcurrentHashMap<String, String>();

    private FileSelector() {}

    public static void selectFile(String ignoredType) {
        isDone = false;
        filePath = "";
        launchTreePicker(REQUEST_OPEN_TREE);
    }

    public static void selectFolder() {
        isDone = false;
        folderPath = "";
        launchTreePicker(REQUEST_FOLDER);
    }

    public static void saveAs(String name) {
        isDone = false;
        filePath = "";
        pendingSaveName = ensureExtension(name == null ? "level.adofai" : name);
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            setPath("");
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    PickerFragment fragment = ensureFragment(owner);
                    Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    intent.setType("application/json");
                    intent.putExtra(Intent.EXTRA_TITLE, pendingSaveName);
                    fragment.startActivityForResult(intent, REQUEST_SAVE);
                } catch (Throwable error) {
                    Log.e(TAG, "Could not launch create-document picker", error);
                    setPath("");
                }
            }
        });
    }

    private static void launchTreePicker(final int requestCode) {
        final Activity owner = context;
        if (owner == null || owner.isFinishing()) {
            if (requestCode == REQUEST_FOLDER) {
                folderPath = "";
                isDone = true;
            } else {
                setPath("");
            }
            return;
        }
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    PickerFragment fragment = ensureFragment(owner);
                    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
                    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                            Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
                    fragment.startActivityForResult(intent, requestCode);
                } catch (Throwable error) {
                    Log.e(TAG, "Could not launch folder picker", error);
                    if (requestCode == REQUEST_FOLDER) {
                        folderPath = "";
                        isDone = true;
                    } else {
                        setPath("");
                    }
                }
            }
        });
    }

    private static PickerFragment ensureFragment(Activity owner) {
        FragmentManager manager = owner.getFragmentManager();
        Fragment existing = manager.findFragmentByTag(FRAGMENT_TAG);
        if (existing instanceof PickerFragment) return (PickerFragment) existing;
        PickerFragment fragment = new PickerFragment();
        manager.beginTransaction().add(fragment, FRAGMENT_TAG).commitAllowingStateLoss();
        manager.executePendingTransactions();
        return fragment;
    }

    public static final class PickerFragment extends Fragment {
        @Override public void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
            FileSelector.handleActivityResult(requestCode, resultCode, data);
        }
    }

    private static void handleActivityResult(final int requestCode, int resultCode, Intent data) {
        if (requestCode != REQUEST_OPEN_TREE && requestCode != REQUEST_SAVE && requestCode != REQUEST_FOLDER) return;
        if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
            if (requestCode == REQUEST_FOLDER) {
                folderPath = "";
                isDone = true;
            } else {
                setPath("");
            }
            return;
        }

        final Uri uri = data.getData();
        takePersistablePermission(uri, data.getFlags());
        if (requestCode == REQUEST_FOLDER) {
            folderPath = uri.toString();
            isDone = true;
            return;
        }
        if (requestCode == REQUEST_SAVE) {
            try {
                setPath(prepareSaveDocument(uri, pendingSaveName));
            } catch (Throwable error) {
                Log.e(TAG, "Could not prepare Save As document", error);
                setPath("");
            }
            return;
        }

        new Thread(new Runnable() {
            @Override public void run() {
                try {
                    setPath(importTree(uri));
                } catch (Throwable error) {
                    Log.e(TAG, "Could not mirror selected ADOFAI map folder", error);
                    setPath("");
                }
            }
        }, "ADOFAI-SAF-import").start();
    }

    private static String importTree(Uri treeUri) throws Exception {
        Activity owner = requireOwner();
        File root = newWorkingDirectory(owner);
        ContentResolver resolver = owner.getContentResolver();
        String rootId = DocumentsContract.getTreeDocumentId(treeUri);
        CopyState state = new CopyState();
        copyChildren(resolver, treeUri, rootId, root, state);
        if (state.chart == null || state.chartUri == null) {
            throw new IllegalArgumentException("Selected folder does not contain a .adofai file");
        }
        remember(state.chart, Uri.parse(state.chartUri), state.chart.getName());
        return state.chart.getAbsolutePath();
    }

    private static void copyChildren(ContentResolver resolver, Uri treeUri, String parentId,
                                     File localParent, CopyState state) throws Exception {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        Cursor cursor = null;
        try {
            cursor = resolver.query(children, new String[] {
                    DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    DocumentsContract.Document.COLUMN_MIME_TYPE
            }, null, null, null);
            if (cursor == null) throw new IllegalStateException("Document provider returned no child cursor");
            while (cursor.moveToNext()) {
                if (++state.files > MAX_FILES) throw new IllegalStateException("Map folder has too many files");
                String documentId = cursor.getString(0);
                String displayName = safeSegment(cursor.getString(1));
                String mime = cursor.getString(2);
                Uri child = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                File local = new File(localParent, displayName);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    if (!local.mkdirs() && !local.isDirectory()) {
                        throw new IllegalStateException("Could not create working directory: " + displayName);
                    }
                    copyChildren(resolver, treeUri, documentId, local, state);
                } else {
                    copyDocument(resolver, child, local, state);
                    if (displayName.toLowerCase(Locale.US).endsWith(".adofai")) {
                        boolean preferred = displayName.equalsIgnoreCase("level.adofai");
                        if (state.chart == null || preferred) {
                            state.chart = local;
                            state.chartUri = child.toString();
                        }
                    }
                }
            }
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    private static void copyDocument(ContentResolver resolver, Uri source, File target,
                                     CopyState state) throws Exception {
        InputStream input = resolver.openInputStream(source);
        if (input == null) throw new IllegalStateException("Document provider returned no input stream");
        FileOutputStream output = new FileOutputStream(target, false);
        try {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read == 0) continue;
                state.bytes += read;
                if (state.bytes > MAX_BYTES) throw new IllegalStateException("Map folder exceeds 1 GiB mirror limit");
                output.write(buffer, 0, read);
            }
            output.flush();
            output.getFD().sync();
        } finally {
            try { input.close(); } finally { output.close(); }
        }
    }

    private static String prepareSaveDocument(Uri uri, String fallbackName) throws Exception {
        Activity owner = requireOwner();
        String name = ensureExtension(queryDisplayName(owner.getContentResolver(), uri, fallbackName));
        File session = newWorkingDirectory(owner);
        File working = new File(session, safeSegment(name));
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
            FileInputStream input = new FileInputStream(source);
            try {
                byte[] buffer = new byte[64 * 1024];
                int read;
                while ((read = input.read(buffer)) >= 0) if (read > 0) output.write(buffer, 0, read);
                output.flush();
            } finally {
                try { input.close(); } finally { output.close(); }
            }
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "Could not synchronize saved chart", error);
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

    public static String getFilePath() { return filePath; }
    public static String getFolderPath() { return folderPath; }

    private static Activity requireOwner() {
        Activity owner = context;
        if (owner == null || owner.isFinishing()) throw new IllegalStateException("No foreground Activity");
        return owner;
    }

    private static File newWorkingDirectory(Activity owner) {
        File root = new File(owner.getFilesDir(), "adofai-editor-working");
        File session = new File(root, "map-" + UUID.randomUUID().toString());
        if (!session.mkdirs() && !session.isDirectory()) {
            throw new IllegalStateException("Could not create private working directory");
        }
        return session;
    }

    private static void remember(File working, Uri uri, String displayName) {
        URI_BY_PATH.put(working.getAbsolutePath(), uri.toString());
        NAME_BY_PATH.put(working.getAbsolutePath(), displayName);
    }

    private static void takePersistablePermission(Uri uri, int resultFlags) {
        Activity owner = context;
        if (owner == null) return;
        int flags = resultFlags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        try {
            owner.getContentResolver().takePersistableUriPermission(uri, flags);
        } catch (SecurityException ignored) {
            // Some providers intentionally grant only session-scoped access.
        }
    }

    private static String queryDisplayName(ContentResolver resolver, Uri uri, String fallback) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String name = cursor.getString(index);
                    if (name != null && !name.trim().isEmpty()) return name.trim();
                }
            }
        } catch (Throwable error) {
            Log.w(TAG, "Could not query document display name", error);
        } finally {
            if (cursor != null) cursor.close();
        }
        return fallback == null ? "level.adofai" : fallback;
    }

    private static String ensureExtension(String name) {
        String safe = safeSegment(name);
        return safe.toLowerCase(Locale.US).endsWith(".adofai") ? safe : safe + ".adofai";
    }

    private static String safeSegment(String name) {
        String safe = name == null ? "unnamed" : name.trim();
        safe = safe.replace('/', '_').replace('\\', '_');
        if (safe.isEmpty() || safe.equals(".") || safe.equals("..")) safe = "unnamed";
        return safe;
    }

    private static final class CopyState {
        int files;
        long bytes;
        File chart;
        String chartUri;
    }
}
