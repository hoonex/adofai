package com.unity3d.player;

import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.FileObserver;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.provider.DocumentsContract;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Dedicated tree-backed level opener for .adofai charts.
 *
 * ADOFAI charts commonly reference audio/images by a relative path. ACTION_OPEN_DOCUMENT
 * only grants one file, so a chart opened that way can load while its song/background silently
 * fails. This bridge mirrors the selected map directory and binds only the chosen .adofai file
 * back to its original SAF document. It deliberately does not propagate arbitrary tree deletes,
 * renames, or asset writes.
 */
public final class V240LevelFolderBridge {
    private static final String TAG = "ADOFAI.V240LevelTree";
    private static final String FRAGMENT_TAG = "adofai-v240-level-tree";
    private static final int PICK_TREE = 7341;
    private static final int MAX_FILES = 4096;
    private static final int MAX_DEPTH = 64;
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int COPY_BUFFER_BYTES = 256 * 1024;
    private static final long MODIFY_DEBOUNCE_MS = 180L;

    private static final AtomicInteger NEXT_ID = new AtomicInteger(34000);
    private static final ConcurrentHashMap<Integer, Result> RESULTS =
            new ConcurrentHashMap<Integer, Result>();
    private static final Object IO_LOCK = new Object();
    private static volatile HandlerThread IO_THREAD;
    private static volatile Handler IO;
    private static volatile SaveBinding ACTIVE_BINDING;

    private V240LevelFolderBridge() {}

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
        RequestCancelledException() { super("level folder request cancelled"); }
    }

    private static final class CopyState {
        int files;
        long bytes;
        File chart;
        Uri chartUri;
        int chartRank = Integer.MAX_VALUE;
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
                            Log.e(TAG, "level save sync failed: " + SaveBinding.this.file, error);
                        }
                    }
                }
            };
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

    public static int begin() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return -1;
        final int id = NEXT_ID.incrementAndGet();
        RESULTS.put(id, new Result());
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    PickerFragment fragment = ensureFragment(owner);
                    fragment.launch(id);
                } catch (Throwable error) {
                    fail(id, error);
                }
            }
        });
        return id;
    }

    public static void cancel(int id) {
        complete(id, Result.CANCEL, "");
    }

    public static String await(int id, long timeoutMs) {
        Result result = RESULTS.get(id);
        if (result == null) return "E:unknown request";
        try {
            if (!result.done.await(Math.max(1L, timeoutMs), TimeUnit.MILLISECONDS)) {
                complete(id, Result.ERROR, "picker timeout");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            complete(id, Result.ERROR, "picker interrupted");
        }
        int state;
        String value;
        synchronized (result) {
            state = result.state;
            value = result.value;
        }
        RESULTS.remove(id, result);
        if (state == Result.OK) return "O:" + value;
        if (state == Result.CANCEL) return "C:";
        return "E:" + value;
    }

    /** Called only after a different Save-As target successfully becomes authoritative. */
    public static void releaseActiveLevel(boolean flush) {
        SaveBinding binding;
        synchronized (V240LevelFolderBridge.class) {
            binding = ACTIVE_BINDING;
            ACTIVE_BINDING = null;
        }
        stopBinding(binding, flush);
    }

    public static final class PickerFragment extends Fragment {
        private int pendingId = -1;

        @Override public void onCreate(Bundle state) {
            super.onCreate(state);
            setRetainInstance(true);
            if (state != null) pendingId = state.getInt("requestId", -1);
        }

        @Override public void onSaveInstanceState(Bundle outState) {
            outState.putInt("requestId", pendingId);
            super.onSaveInstanceState(outState);
        }

        void launch(int id) {
            if (pendingId > 0 && pendingId != id) cancel(pendingId);
            pendingId = id;
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION |
                    Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
            startActivityForResult(intent, PICK_TREE);
        }

        @Override public void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
            if (requestCode != PICK_TREE) return;
            final int id = pendingId;
            pendingId = -1;
            if (id <= 0) return;
            if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
                cancel(id);
                return;
            }
            Activity owner = getActivity();
            if (owner == null) {
                fail(id, new IllegalStateException("picker Activity was recreated without an owner"));
                return;
            }
            final Context app = owner.getApplicationContext();
            final Uri treeUri = data.getData();
            final int flags = data.getFlags();
            io().post(new Runnable() {
                @Override public void run() { importTree(app, id, treeUri, flags); }
            });
        }
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

    private static void importTree(Context context, int id, Uri treeUri, int grantFlags) {
        File mirror = null;
        SaveBinding installed = null;
        boolean success = false;
        try {
            ensurePending(id);
            persist(context, treeUri, grantFlags);
            mirror = new File(context.getFilesDir(), "v240-working/map-" + UUID.randomUUID());
            if (!mirror.mkdirs() && !mirror.isDirectory()) {
                throw new IllegalStateException("map mirror directory could not be created");
            }
            CopyState state = new CopyState();
            String rootId = DocumentsContract.getTreeDocumentId(treeUri);
            copyChildren(context.getContentResolver(), treeUri, rootId, mirror, state, id, 0);
            ensurePending(id);
            if (state.chart == null || state.chartUri == null) {
                throw new IllegalArgumentException("selected folder does not contain an .adofai level");
            }

            // Compatibility rewrites belong to the private mirror, never the authoritative SAF
            // document merely because the user opened it. Run them before constructing/starting
            // the write-back FileObserver so no backport or targeted legacy workaround can be
            // mistaken for an editor save.
            V240ChartBackport.backportForV240(state.chart);
            V240HallLegacyFix.applyIfNeeded(state.chart);
            V240OpaqueEventBridge.prepareForV240(state.chart);
            ensurePending(id);

            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
                installed = new SaveBinding(context, state.chartUri, state.chart);
                replaceActiveBinding(installed);
            }
            success = complete(id, Result.OK, state.chart.getAbsolutePath());
        } catch (RequestCancelledException ignored) {
        } catch (Throwable error) {
            if (isPending(id)) fail(id, error);
        } finally {
            if (!success) {
                if (installed != null) {
                    synchronized (V240LevelFolderBridge.class) {
                        if (ACTIVE_BINDING == installed) ACTIVE_BINDING = null;
                    }
                    stopBinding(installed, false);
                }
                deleteRecursively(mirror);
            }
        }
    }

    private static void copyChildren(ContentResolver resolver, Uri treeUri, String parentId,
                                     File localParent, CopyState state, int requestId, int depth)
            throws Exception {
        ensurePending(requestId);
        if (depth > MAX_DEPTH) throw new IllegalStateException("map folder is nested too deeply");
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        Cursor cursor = null;
        try {
            cursor = resolver.query(children, new String[] {
                    DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    DocumentsContract.Document.COLUMN_MIME_TYPE,
                    DocumentsContract.Document.COLUMN_SIZE
            }, null, null, null);
            if (cursor == null) throw new IllegalStateException("tree provider returned no cursor");
            while (cursor.moveToNext()) {
                ensurePending(requestId);
                String documentId = cursor.getString(0);
                String displayName = safeSegment(cursor.getString(1));
                String mime = cursor.getString(2);
                long declaredSize = cursor.isNull(3) ? -1L : Math.max(0L, cursor.getLong(3));
                Uri childUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    File dir = uniqueChild(localParent, displayName.length() == 0 ? "folder" : displayName);
                    if (!dir.mkdirs() && !dir.isDirectory()) {
                        throw new IllegalStateException("map mirror mkdir failed");
                    }
                    copyChildren(resolver, treeUri, documentId, dir, state, requestId, depth + 1);
                    continue;
                }
                if (++state.files > MAX_FILES) throw new IllegalStateException("map folder has too many files");
                long remaining = MAX_BYTES - state.bytes;
                if (remaining < 0L || (declaredSize >= 0L && declaredSize > remaining)) {
                    throw new IllegalStateException("map folder exceeds safe mirror size");
                }
                File target = uniqueChild(localParent, displayName.length() == 0 ? "file" : displayName);
                try (InputStream in = requireInput(resolver, childUri);
                     OutputStream out = new FileOutputStream(target)) {
                    state.bytes += copy(in, out, requestId, remaining);
                } catch (Throwable error) {
                    if (target.exists() && !target.delete()) target.deleteOnExit();
                    throw error;
                }
                int rank = chartRank(displayName);
                if (rank < state.chartRank) {
                    state.chartRank = rank;
                    state.chart = target;
                    state.chartUri = childUri;
                }
            }
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    private static int chartRank(String name) {
        if (name == null) return Integer.MAX_VALUE;
        String lower = name.toLowerCase(Locale.US);
        if (!lower.endsWith(".adofai")) return Integer.MAX_VALUE;
        if ("level.adofai".equals(lower)) return 0;
        if ("main.adofai".equals(lower)) return 1;
        return 2;
    }

    private static synchronized void replaceActiveBinding(SaveBinding next) {
        SaveBinding previous = ACTIVE_BINDING;
        if (previous == next) return;
        if (previous != null) stopBinding(previous, true);
        ACTIVE_BINDING = next;
        next.observer.startWatching();
    }

    private static void stopBinding(SaveBinding binding, boolean flush) {
        if (binding == null) return;
        synchronized (binding) {
            if (!binding.active) return;
            if (IO != null) IO.removeCallbacks(binding.syncTask);
            if (flush && binding.file.isFile()) {
                try { syncNow(binding); }
                catch (Throwable error) { Log.e(TAG, "final level save sync failed", error); }
            }
            binding.active = false;
            binding.observer.stopWatching();
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
        File restored = V240OpaqueEventBridge.buildRestoredExport(binding.file);
        File source = restored != null ? restored : binding.file;
        try {
            ContentResolver resolver = binding.context.getContentResolver();
            try (InputStream in = new FileInputStream(source);
                 OutputStream out = requireOutput(resolver, binding.uri)) {
                copy(in, out, -1, Long.MAX_VALUE);
            }
        } finally {
            V240OpaqueEventBridge.releaseRestoredExport(restored);
        }
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

    private static void fail(int id, Throwable error) {
        Log.e(TAG, "level folder request failed id=" + id, error);
        complete(id, Result.ERROR, safeMessage(error));
    }

    private static Handler io() {
        Handler handler = IO;
        if (handler != null) return handler;
        synchronized (IO_LOCK) {
            handler = IO;
            if (handler != null) return handler;
            HandlerThread thread = new HandlerThread(
                    "adofai-v240-level-tree", Process.THREAD_PRIORITY_BACKGROUND);
            thread.start();
            IO_THREAD = thread;
            handler = new Handler(thread.getLooper());
            IO = handler;
            return handler;
        }
    }

    private static void persist(Context context, Uri uri, int flags) {
        int allowed = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION |
                Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if (allowed == 0) return;
        try { context.getContentResolver().takePersistableUriPermission(uri, allowed); }
        catch (SecurityException ignored) {}
    }

    private static Activity currentActivity() {
        try {
            Class<?> player = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = player.getField("currentActivity");
            Object value = field.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable error) {
            Log.w(TAG, "UnityPlayer.currentActivity unavailable", error);
            return null;
        }
    }

    private static InputStream requireInput(ContentResolver resolver, Uri uri) throws Exception {
        InputStream in = resolver.openInputStream(uri);
        if (in == null) throw new IllegalStateException("document provider returned no input stream");
        return in;
    }

    private static OutputStream requireOutput(ContentResolver resolver, Uri uri) throws Exception {
        OutputStream out = null;
        try { out = resolver.openOutputStream(uri, "wt"); } catch (Throwable ignored) {}
        if (out == null) {
            try { out = resolver.openOutputStream(uri, "w"); } catch (Throwable ignored) {}
        }
        if (out == null) out = resolver.openOutputStream(uri);
        if (out == null) throw new IllegalStateException("document provider returned no output stream");
        return out;
    }

    private static long copy(InputStream in, OutputStream out, int requestId, long maxBytes)
            throws Exception {
        byte[] buffer = new byte[COPY_BUFFER_BYTES];
        long total = 0L;
        int count;
        while ((count = in.read(buffer)) != -1) {
            if (requestId > 0) ensurePending(requestId);
            total += count;
            if (total > maxBytes) throw new IllegalStateException("map folder exceeds safe mirror size");
            out.write(buffer, 0, count);
        }
        if (requestId > 0) ensurePending(requestId);
        out.flush();
        return total;
    }

    private static File uniqueChild(File parent, String name) {
        File candidate = new File(parent, name);
        if (!candidate.exists()) return candidate;
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : "";
        for (int i = 2; i < 10000; ++i) {
            candidate = new File(parent, base + " (" + i + ")" + ext);
            if (!candidate.exists()) return candidate;
        }
        return new File(parent, UUID.randomUUID().toString() + ext);
    }

    private static String safeSegment(String raw) {
        String value = raw == null ? "unnamed" : raw.trim();
        StringBuilder out = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); ++i) {
            char ch = value.charAt(i);
            if (ch < 0x20 || ch == '/' || ch == '\\') out.append('_');
            else out.append(ch);
        }
        value = out.toString();
        while (value.startsWith(".")) value = value.substring(1);
        if (value.length() == 0) value = "unnamed";
        if (value.length() > 120) value = value.substring(value.length() - 120);
        return value;
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteRecursively(child);
        }
        if (!file.delete()) file.deleteOnExit();
    }

    private static String safeMessage(Throwable error) {
        if (error == null) return "unknown error";
        String value = error.getMessage();
        return value == null || value.trim().length() == 0
                ? error.getClass().getSimpleName() : value;
    }
}
