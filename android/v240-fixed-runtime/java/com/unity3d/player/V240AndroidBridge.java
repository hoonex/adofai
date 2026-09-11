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
import android.os.Process;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Android Storage Access Framework backend for the historical ADOFAI 2.4 editor.
 *
 * The old editor expects ordinary filesystem paths. Android document providers return
 * content:// URIs, so this bridge mirrors selected documents into app-private working
 * files and keeps writable Open / Save-As destinations synchronized back to the provider.
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
    private static final int MAX_TREE_DEPTH = 64;
    private static final long MAX_TREE_BYTES = 512L * 1024L * 1024L;
    private static final int COPY_BUFFER_BYTES = 256 * 1024;
    private static final long MODIFY_DEBOUNCE_MS = 180L;

    private static final AtomicInteger NEXT_ID = new AtomicInteger(24000);
    private static final Map<Integer, Result> RESULTS = new ConcurrentHashMap<Integer, Result>();
    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();
    private static final Object IO_LOCK = new Object();
    private static volatile HandlerThread IO_THREAD;
    private static volatile Handler IO;
    private static final ThreadLocal<byte[]> COPY_BUFFER = new ThreadLocal<byte[]>() {
        @Override protected byte[] initialValue() {
            return new byte[COPY_BUFFER_BYTES];
        }
    };

    private V240AndroidBridge() {}

    private static final class Result {
        static final int PENDING = 0;
        static final int OK = 1;
        static final int CANCEL = 2;
        static final int ERROR = 3;
        final CountDownLatch done = new CountDownLatch(1);
        volatile int state = PENDING;
        volatile String value = "";
    }

    private static final class RequestCancelledException extends IOException {
        RequestCancelledException() { super("picker request cancelled"); }
    }

    private static final class SaveBinding {
        final Context context;
        final Uri uri;
        final File file;
        final FileObserver observer;
        final Runnable syncTask;
        volatile boolean active = true;

        SaveBinding(final Context context, final Uri uri, final File file) {
            this.context = context.getApplicationContext();
            this.uri = uri;
            this.file = file;
            this.syncTask = new Runnable() {
                @Override public void run() {
                    synchronized (SaveBinding.this) {
                        if (!SaveBinding.this.active) return;
                        try {
                            syncNow(SaveBinding.this);
                        } catch (Throwable error) {
                            Log.e(TAG, "background save sync failed: " + SaveBinding.this.file, error);
                        }
                    }
                }
            };

            // Watch the parent directory instead of a single inode. Unity/editor code can
            // save through a temporary file followed by an atomic rename, which detaches a
            // file-only observer. CLOSE_WRITE/MOVED_TO/CREATE are terminal save signals and
            // are pushed back immediately; noisy MODIFY events remain debounced.
            final File parent = file.getParentFile();
            final String watchedName = file.getName();
            this.observer = new FileObserver(parent.getAbsolutePath(),
                    FileObserver.CLOSE_WRITE | FileObserver.MODIFY |
                            FileObserver.MOVED_TO | FileObserver.CREATE) {
                @Override public void onEvent(int event, String path) {
                    if (!SaveBinding.this.active) return;
                    if (path != null && !watchedName.equals(path)) return;
                    int type = event & FileObserver.ALL_EVENTS;
                    if ((type & (FileObserver.CLOSE_WRITE | FileObserver.MOVED_TO |
                            FileObserver.CREATE)) != 0) {
                        scheduleSync(SaveBinding.this, 0L);
                    } else if ((type & FileObserver.MODIFY) != 0) {
                        scheduleSync(SaveBinding.this, MODIFY_DEBOUNCE_MS);
                    }
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

    public static String poll(int id) {
        Result result = RESULTS.get(id);
        if (result == null) return "E:unknown request";
        return consumeResult(id, result);
    }

    /** Blocks only the FileSelector daemon waiter, never the Android main thread. */
    static String await(int id, long timeoutMs) {
        Result result = RESULTS.get(id);
        if (result == null) return "E:unknown request";
        try {
            long waitMs = Math.max(1L, timeoutMs);
            if (!result.done.await(waitMs, TimeUnit.MILLISECONDS)) {
                complete(id, Result.ERROR, "picker timeout");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            complete(id, Result.ERROR, "picker interrupted");
        }
        return consumeResult(id, result);
    }

    private static String consumeResult(int id, Result result) {
        int state;
        String value;
        synchronized (result) {
            state = result.state;
            value = result.value;
        }
        if (state == Result.PENDING) return "P";
        RESULTS.remove(id, result);
        if (state == Result.OK) return "O:" + value;
        if (state == Result.CANCEL) return "C:";
        return "E:" + value;
    }

    static void cancel(int id) {
        complete(id, Result.CANCEL, "");
    }

    static void fail(int id, Throwable error) {
        Log.e(TAG, "picker request failed id=" + id, error);
        complete(id, Result.ERROR, safeMessage(error));
    }

    private static boolean complete(int id, int state, String value) {
        Result result = RESULTS.get(id);
        if (result == null) return false;
        synchronized (result) {
            if (result.state != Result.PENDING) return false;
            result.value = value == null ? "" : value;
            result.state = state;
        }
        result.done.countDown();
        return true;
    }

    private static boolean isPending(int id) {
        Result result = RESULTS.get(id);
        return result != null && result.state == Result.PENDING;
    }

    private static void ensurePending(int id) throws RequestCancelledException {
        if (!isPending(id)) throw new RequestCancelledException();
    }

    static void handleResultAsync(final Context context, final int id, final int mode,
                                  final Uri uri, final int grantFlags, final String suggestedName) {
        final Context appContext = context.getApplicationContext();
        io().post(new Runnable() {
            @Override public void run() {
                if (!isPending(id)) return;
                if (mode == MODE_OPEN) {
                    handleOpen(appContext, id, uri, grantFlags);
                } else if (mode == MODE_SAVE) {
                    handleSave(appContext, id, uri, grantFlags, suggestedName);
                } else if (mode == MODE_FOLDER) {
                    handleFolder(appContext, id, uri, grantFlags);
                } else {
                    fail(id, new IllegalArgumentException("unknown picker mode: " + mode));
                }
            }
        });
    }

    static void handleOpen(Context context, int id, Uri uri, int grantFlags) {
        File working = null;
        boolean bound = false;
        boolean success = false;
        try {
            ensurePending(id);
            persist(context, uri, grantFlags);
            ensurePending(id);
            working = makeWorkingFile(context,
                    displayName(context.getContentResolver(), uri, "level.adofai"));
            try (InputStream in = requireInput(context.getContentResolver(), uri);
                 OutputStream out = new FileOutputStream(working)) {
                copy(in, out, id, Long.MAX_VALUE);
            }
            ensurePending(id);
            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
                bindSave(context, uri, working);
                bound = true;
            }
            success = complete(id, Result.OK, working.getAbsolutePath());
        } catch (RequestCancelledException ignored) {
            // Cancellation already owns the terminal result.
        } catch (Throwable error) {
            if (isPending(id)) fail(id, error);
        } finally {
            if (!success && working != null) {
                if (bound) removeSaveBinding(working.getAbsolutePath());
                deleteRecursively(working.getParentFile());
            }
        }
    }

    static void handleSave(Context context, int id, Uri uri, int grantFlags, String suggestedName) {
        File working = null;
        boolean bound = false;
        boolean success = false;
        try {
            ensurePending(id);
            persist(context, uri, grantFlags);
            ensurePending(id);
            String name = displayName(context.getContentResolver(), uri,
                    emptyToDefault(suggestedName, "level.adofai"));
            working = makeWorkingFile(context, name);
            if (!working.createNewFile() && !working.isFile()) {
                throw new IllegalStateException("working save file could not be created");
            }
            ensurePending(id);
            bindSave(context, uri, working);
            bound = true;
            success = complete(id, Result.OK, working.getAbsolutePath());
        } catch (RequestCancelledException ignored) {
            // Cancellation already owns the terminal result.
        } catch (Throwable error) {
            if (isPending(id)) fail(id, error);
        } finally {
            if (!success && working != null) {
                if (bound) removeSaveBinding(working.getAbsolutePath());
                deleteRecursively(working.getParentFile());
            }
        }
    }

    static void handleFolder(Context context, int id, Uri treeUri, int grantFlags) {
        File mirror = null;
        boolean success = false;
        try {
            ensurePending(id);
            persist(context, treeUri, grantFlags);
            ensurePending(id);
            mirror = new File(context.getFilesDir(), "v240-working/tree-" + UUID.randomUUID());
            if (!mirror.mkdirs() && !mirror.isDirectory()) {
                throw new IllegalStateException("tree mirror directory could not be created");
            }
            TreeBudget budget = new TreeBudget();
            String rootId = DocumentsContract.getTreeDocumentId(treeUri);
            mirrorChildren(context.getContentResolver(), treeUri, rootId, mirror, budget, id, 0);
            ensurePending(id);
            success = complete(id, Result.OK, mirror.getAbsolutePath());
        } catch (RequestCancelledException ignored) {
            // Cancellation already owns the terminal result.
        } catch (Throwable error) {
            if (isPending(id)) fail(id, error);
        } finally {
            if (!success && mirror != null) deleteRecursively(mirror);
        }
    }

    /** Explicit flush used before preview/close; FileObserver remains the normal path. */
    public static boolean flushSave(String localPath) {
        SaveBinding binding = SAVE_BINDINGS.get(localPath);
        if (binding == null) return true;
        try {
            synchronized (binding) {
                if (IO != null) IO.removeCallbacks(binding.syncTask);
                if (!binding.active) return true;
                syncNow(binding);
            }
            return true;
        } catch (Throwable error) {
            Log.e(TAG, "explicit save flush failed", error);
            return false;
        }
    }

    private static void bindSave(Context context, Uri uri, File file) {
        // Retire the previous document only after a final synchronous flush. This closes a
        // real data-loss race where a pending MODIFY debounce could otherwise be cancelled
        // by Open/Save-As before it reached the document provider.
        for (Map.Entry<String, SaveBinding> entry : SAVE_BINDINGS.entrySet()) {
            SaveBinding existing = entry.getValue();
            if (SAVE_BINDINGS.remove(entry.getKey(), existing)) stopBinding(existing, true);
        }
        SaveBinding binding = new SaveBinding(context, uri, file);
        SAVE_BINDINGS.put(file.getAbsolutePath(), binding);
        binding.observer.startWatching();
    }

    private static void removeSaveBinding(String localPath) {
        SaveBinding binding = SAVE_BINDINGS.remove(localPath);
        if (binding != null) stopBinding(binding, true);
    }

    private static void stopBinding(SaveBinding binding, boolean finalFlush) {
        if (binding == null) return;
        synchronized (binding) {
            if (!binding.active) return;
            if (IO != null) IO.removeCallbacks(binding.syncTask);
            if (finalFlush && binding.file.isFile()) {
                try {
                    syncNow(binding);
                } catch (Throwable error) {
                    // Do not silently hide a failed final flush: the next document can still
                    // open, but diagnostics clearly show that provider write-back failed.
                    Log.e(TAG, "final save sync failed: " + binding.file, error);
                }
            }
            binding.active = false;
            binding.observer.stopWatching();
        }
    }

    private static Handler io() {
        Handler handler = IO;
        if (handler != null) return handler;
        synchronized (IO_LOCK) {
            handler = IO;
            if (handler != null) return handler;
            HandlerThread thread = new HandlerThread(
                    "adofai-v240-storage", Process.THREAD_PRIORITY_BACKGROUND);
            thread.start();
            IO_THREAD = thread;
            handler = new Handler(thread.getLooper());
            IO = handler;
            return handler;
        }
    }

    private static void scheduleSync(final SaveBinding binding, long delayMs) {
        synchronized (binding) {
            if (!binding.active) return;
            Handler handler = io();
            handler.removeCallbacks(binding.syncTask);
            if (delayMs <= 0L) handler.post(binding.syncTask);
            else handler.postDelayed(binding.syncTask, delayMs);
        }
    }

    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        ContentResolver resolver = binding.context.getContentResolver();
        try (InputStream in = new FileInputStream(binding.file);
             OutputStream out = requireOutput(resolver, binding.uri)) {
            copy(in, out);
        }
    }

    private static final class TreeBudget {
        int files;
        long bytes;
    }

    private static void mirrorChildren(ContentResolver resolver, Uri treeUri, String parentId,
                                       File localParent, TreeBudget budget, int requestId, int depth)
            throws Exception {
        ensurePending(requestId);
        if (depth > MAX_TREE_DEPTH) {
            throw new IllegalStateException("selected folder is nested too deeply");
        }
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
                ensurePending(requestId);
                String documentId = cursor.getString(0);
                String displayName = sanitizeName(cursor.getString(1));
                String mime = cursor.getString(2);
                long declaredSize = cursor.isNull(3) ? -1L : Math.max(0L, cursor.getLong(3));
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    File dir = uniqueChild(localParent, displayName.length() == 0 ? "folder" : displayName);
                    if (!dir.mkdirs() && !dir.isDirectory()) {
                        throw new IllegalStateException("mirror mkdir failed");
                    }
                    mirrorChildren(resolver, treeUri, documentId, dir, budget, requestId, depth + 1);
                    continue;
                }
                if (++budget.files > MAX_TREE_FILES) {
                    throw new IllegalStateException("selected folder has too many files to mirror safely");
                }
                long remaining = MAX_TREE_BYTES - budget.bytes;
                if (remaining < 0 || (declaredSize >= 0 && declaredSize > remaining)) {
                    throw new IllegalStateException("selected folder is too large to mirror safely");
                }
                File target = uniqueChild(localParent, displayName.length() == 0 ? "file" : displayName);
                Uri documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                try (InputStream in = requireInput(resolver, documentUri);
                     OutputStream out = new FileOutputStream(target)) {
                    budget.bytes += copy(in, out, requestId, remaining);
                } catch (Throwable error) {
                    if (target.exists() && !target.delete()) target.deleteOnExit();
                    throw error;
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
        if (!dir.mkdirs() && !dir.isDirectory()) {
            throw new IllegalStateException("working directory could not be created");
        }
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
        OutputStream out = null;
        try {
            out = resolver.openOutputStream(uri, "wt");
        } catch (Throwable ignored) {
            // Not every DocumentsProvider implements truncate mode even when it supports write.
        }
        if (out == null) {
            try {
                out = resolver.openOutputStream(uri, "w");
            } catch (Throwable ignored) {
            }
        }
        if (out == null) out = resolver.openOutputStream(uri);
        if (out == null) throw new IllegalStateException("document provider returned no output stream");
        return out;
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        copy(in, out, -1, Long.MAX_VALUE);
    }

    private static long copy(InputStream in, OutputStream out, int requestId, long maxBytes)
            throws Exception {
        byte[] buffer = COPY_BUFFER.get();
        long total = 0L;
        int count;
        while ((count = in.read(buffer)) != -1) {
            if (requestId > 0) ensurePending(requestId);
            total += count;
            if (total > maxBytes) {
                throw new IllegalStateException("selected folder is too large to mirror safely");
            }
            out.write(buffer, 0, count);
        }
        if (requestId > 0) ensurePending(requestId);
        out.flush();
        return total;
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursively(child);
            }
        }
        if (!file.delete()) file.deleteOnExit();
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
