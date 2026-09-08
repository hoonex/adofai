package com.unity3d.player;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.FileObserver;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Android Storage Access Framework backend for the historical ADOFAI 2.4 editor.
 *
 * The old editor expects ordinary filesystem paths. Android document providers return
 * content:// URIs, so this bridge mirrors selected documents into app-private working
 * files and keeps Save-As destinations synchronized back to the provider.
 *
 * Native IL2CPP hooks interact with this class through request ids and polling. That
 * keeps Android Activity lifecycle work out of the game thread and guarantees that a
 * cancelled picker still reaches a terminal state.
 */
public final class V240AndroidBridge {
    public static final String TAG = "ADOFAI.V240Bridge";
    public static final String EXTRA_REQUEST_ID = "dev.hoonex.adofai.v240.REQUEST_ID";
    public static final String EXTRA_MODE = "dev.hoonex.adofai.v240.MODE";
    public static final String EXTRA_TITLE = "dev.hoonex.adofai.v240.TITLE";
    public static final String EXTRA_MIME = "dev.hoonex.adofai.v240.MIME";

    public static final int MODE_OPEN = 1;
    public static final int MODE_SAVE = 2;
    public static final int MODE_FOLDER = 3;

    private static final int MAX_TREE_FILES = 4096;
    private static final long MAX_TREE_BYTES = 512L * 1024L * 1024L;

    private static final AtomicInteger NEXT_ID = new AtomicInteger(24000);
    private static final Map<Integer, Result> RESULTS = new ConcurrentHashMap<Integer, Result>();
    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();
    private static final HandlerThread IO_THREAD = new HandlerThread("adofai-v240-storage");
    private static final Handler IO;

    static {
        IO_THREAD.start();
        IO = new Handler(IO_THREAD.getLooper());
    }

    private V240AndroidBridge() {}

    private static final class Result {
        static final int PENDING = 0;
        static final int OK = 1;
        static final int CANCEL = 2;
        static final int ERROR = 3;
        volatile int state = PENDING;
        volatile String value = "";
    }

    private static final class SaveBinding {
        final Context context;
        final Uri uri;
        final File file;
        final FileObserver observer;
        volatile long generation;

        SaveBinding(final Context context, final Uri uri, final File file) {
            this.context = context.getApplicationContext();
            this.uri = uri;
            this.file = file;
            this.observer = new FileObserver(file.getAbsolutePath(),
                    FileObserver.CLOSE_WRITE | FileObserver.MODIFY | FileObserver.MOVED_TO) {
                @Override public void onEvent(int event, String path) {
                    scheduleSync(SaveBinding.this);
                }
            };
        }
    }

    /** Returns a positive request id, or -1 when no foreground Activity exists. */
    public static int beginOpen(String mime) {
        return begin(MODE_OPEN, "", emptyToDefault(mime, "*/*"));
    }

    public static int beginSave(String suggestedName, String mime) {
        return begin(MODE_SAVE, sanitizeName(emptyToDefault(suggestedName, "level.adofai")),
                emptyToDefault(mime, "application/octet-stream"));
    }

    public static int beginFolder() {
        return begin(MODE_FOLDER, "", "");
    }

    private static int begin(int mode, String title, String mime) {
        Activity activity = currentActivity();
        if (activity == null || activity.isFinishing()) return -1;
        int id = NEXT_ID.incrementAndGet();
        RESULTS.put(id, new Result());
        Intent proxy = new Intent(activity, V240PickerActivity.class);
        proxy.putExtra(EXTRA_REQUEST_ID, id);
        proxy.putExtra(EXTRA_MODE, mode);
        proxy.putExtra(EXTRA_TITLE, title);
        proxy.putExtra(EXTRA_MIME, mime);
        try {
            activity.startActivity(proxy);
            return id;
        } catch (Throwable error) {
            fail(id, error);
            return id;
        }
    }

    /**
     * Poll format used by native runtime:
     * P                pending
     * O:<filesystem>   success
     * C:               cancelled
     * E:<message>      failed
     */
    public static String poll(int id) {
        Result result = RESULTS.get(id);
        if (result == null) return "E:unknown request";
        if (result.state == Result.PENDING) return "P";
        RESULTS.remove(id);
        if (result.state == Result.OK) return "O:" + result.value;
        if (result.state == Result.CANCEL) return "C:";
        return "E:" + result.value;
    }

    static void cancel(int id) {
        complete(id, Result.CANCEL, "");
    }

    static void fail(int id, Throwable error) {
        Log.e(TAG, "picker request failed id=" + id, error);
        complete(id, Result.ERROR, safeMessage(error));
    }

    private static void complete(int id, int state, String value) {
        Result result = RESULTS.get(id);
        if (result == null) return;
        result.value = value == null ? "" : value;
        result.state = state;
    }

    static void handleOpen(Context context, int id, Uri uri, int grantFlags) {
        try {
            persist(context, uri, grantFlags);
            File working = makeWorkingFile(context, displayName(context.getContentResolver(), uri, "level.adofai"));
            try (InputStream in = new BufferedInputStream(requireInput(context.getContentResolver(), uri));
                 OutputStream out = new BufferedOutputStream(new FileOutputStream(working))) {
                copy(in, out);
            }
            complete(id, Result.OK, working.getAbsolutePath());
        } catch (Throwable error) {
            fail(id, error);
        }
    }

    static void handleSave(Context context, int id, Uri uri, int grantFlags, String suggestedName) {
        try {
            persist(context, uri, grantFlags);
            String name = displayName(context.getContentResolver(), uri,
                    emptyToDefault(suggestedName, "level.adofai"));
            File working = makeWorkingFile(context, name);
            if (!working.createNewFile() && !working.isFile()) {
                throw new IllegalStateException("working save file could not be created");
            }
            bindSave(context, uri, working);
            complete(id, Result.OK, working.getAbsolutePath());
        } catch (Throwable error) {
            fail(id, error);
        }
    }

    static void handleFolder(Context context, int id, Uri treeUri, int grantFlags) {
        try {
            persist(context, treeUri, grantFlags);
            File mirror = new File(context.getFilesDir(), "v240-working/tree-" + UUID.randomUUID());
            if (!mirror.mkdirs() && !mirror.isDirectory()) {
                throw new IllegalStateException("tree mirror directory could not be created");
            }
            TreeBudget budget = new TreeBudget();
            String rootId = DocumentsContract.getTreeDocumentId(treeUri);
            mirrorChildren(context.getContentResolver(), treeUri, rootId, mirror, budget);
            complete(id, Result.OK, mirror.getAbsolutePath());
        } catch (Throwable error) {
            fail(id, error);
        }
    }

    /** Explicit flush used before preview/close; FileObserver remains the normal path. */
    public static boolean flushSave(String localPath) {
        SaveBinding binding = SAVE_BINDINGS.get(localPath);
        if (binding == null) return true;
        try {
            syncNow(binding);
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "explicit save flush failed", error);
            return false;
        }
    }

    private static void bindSave(Context context, Uri uri, File file) {
        SaveBinding old = SAVE_BINDINGS.remove(file.getAbsolutePath());
        if (old != null) old.observer.stopWatching();
        SaveBinding binding = new SaveBinding(context, uri, file);
        SAVE_BINDINGS.put(file.getAbsolutePath(), binding);
        binding.observer.startWatching();
    }

    private static void scheduleSync(final SaveBinding binding) {
        final long expected = ++binding.generation;
        IO.postDelayed(new Runnable() {
            @Override public void run() {
                if (binding.generation != expected) return;
                try {
                    syncNow(binding);
                } catch (Throwable error) {
                    Log.e(TAG, "background save sync failed: " + binding.file, error);
                }
            }
        }, 180L);
    }

    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        ContentResolver resolver = binding.context.getContentResolver();
        try (InputStream in = new BufferedInputStream(new FileInputStream(binding.file));
             OutputStream out = new BufferedOutputStream(requireOutput(resolver, binding.uri))) {
            copy(in, out);
        }
    }

    private static final class TreeBudget {
        int files;
        long bytes;
    }

    private static void mirrorChildren(ContentResolver resolver, Uri treeUri, String parentId,
                                       File localParent, TreeBudget budget) throws Exception {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        Cursor cursor = null;
        try {
            cursor = resolver.query(children,
                    new String[] {DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                            DocumentsContract.Document.COLUMN_MIME_TYPE,
                            DocumentsContract.Document.COLUMN_SIZE}, null, null, null);
            if (cursor == null) throw new IllegalStateException("tree provider returned no cursor");
            while (cursor.moveToNext()) {
                String documentId = cursor.getString(0);
                String displayName = sanitizeName(cursor.getString(1));
                String mime = cursor.getString(2);
                long size = cursor.isNull(3) ? 0L : Math.max(0L, cursor.getLong(3));
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    File dir = uniqueChild(localParent, displayName.length() == 0 ? "folder" : displayName);
                    if (!dir.mkdirs() && !dir.isDirectory()) throw new IllegalStateException("mirror mkdir failed");
                    mirrorChildren(resolver, treeUri, documentId, dir, budget);
                    continue;
                }
                if (++budget.files > MAX_TREE_FILES || (budget.bytes += size) > MAX_TREE_BYTES) {
                    throw new IllegalStateException("selected folder is too large to mirror safely");
                }
                File target = uniqueChild(localParent, displayName.length() == 0 ? "file" : displayName);
                Uri documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                try (InputStream in = new BufferedInputStream(requireInput(resolver, documentUri));
                     OutputStream out = new BufferedOutputStream(new FileOutputStream(target))) {
                    copy(in, out);
                }
            }
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    private static File uniqueChild(File parent, String name) {
        File candidate = new File(parent, name);
        if (!candidate.exists()) return candidate;
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : "";
        for (int i = 2; i < 10000; i++) {
            candidate = new File(parent, base + " (" + i + ")" + ext);
            if (!candidate.exists()) return candidate;
        }
        return new File(parent, UUID.randomUUID().toString() + ext);
    }

    private static File makeWorkingFile(Context context, String displayName) {
        File dir = new File(context.getFilesDir(), "v240-working/doc-" + UUID.randomUUID());
        if (!dir.mkdirs() && !dir.isDirectory()) throw new IllegalStateException("working directory could not be created");
        return new File(dir, sanitizeName(displayName));
    }

    private static void persist(Context context, Uri uri, int flags) {
        int allowed = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if (allowed == 0) return;
        try {
            context.getContentResolver().takePersistableUriPermission(uri, allowed);
        } catch (SecurityException ignored) {
            // Some providers intentionally grant only the lifetime of the picker result.
        }
    }

    private static Activity currentActivity() {
        try {
            Class<?> player = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = player.getField("currentActivity");
            Object value = field.get(null);
            if (value instanceof Activity) return (Activity) value;
        } catch (Throwable error) {
            Log.w(TAG, "UnityPlayer.currentActivity unavailable", error);
        }
        return null;
    }

    private static String displayName(ContentResolver resolver, Uri uri, String fallback) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String value = cursor.getString(index);
                    if (value != null && !value.trim().isEmpty()) return sanitizeName(value);
                }
            }
        } catch (Throwable ignored) {
        } finally {
            if (cursor != null) cursor.close();
        }
        return sanitizeName(fallback);
    }

    private static InputStream requireInput(ContentResolver resolver, Uri uri) throws Exception {
        InputStream in = resolver.openInputStream(uri);
        if (in == null) throw new IllegalStateException("document provider returned no input stream");
        return in;
    }

    private static OutputStream requireOutput(ContentResolver resolver, Uri uri) throws Exception {
        OutputStream out = resolver.openOutputStream(uri, "wt");
        if (out == null) throw new IllegalStateException("document provider returned no output stream");
        return out;
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buffer = new byte[256 * 1024];
        int count;
        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        out.flush();
    }

    private static String sanitizeName(String raw) {
        String value = raw == null ? "file" : raw.trim();
        value = value.replace('/', '_').replace('\\', '_').replace('\u0000', '_');
        while (value.startsWith(".")) value = value.substring(1);
        if (value.length() == 0) value = "file";
        if (value.length() > 120) value = value.substring(value.length() - 120);
        return value;
    }

    private static String emptyToDefault(String value, String fallback) {
        return value == null || value.trim().isEmpty() ? fallback : value.trim();
    }

    private static String safeMessage(Throwable error) {
        if (error == null) return "unknown error";
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? error.getClass().getSimpleName() : message;
    }
}
