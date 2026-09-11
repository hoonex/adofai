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
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Opens downloaded TUF ZIP bundles without requiring the user to extract them first. */
public final class V240ArchiveOpenBridge {
    private static final String TAG = "ADOFAI.V240Archive";
    private static final String FRAGMENT_TAG = "adofai-v240-archive-open";
    private static final int PICK_FILE = 7342;
    private static final long MAX_TRANSPORT_BYTES = 1024L * 1024L * 1024L;
    private static final int BUFFER_BYTES = 256 * 1024;

    private static final AtomicInteger NEXT_ID = new AtomicInteger(44000);
    private static final ConcurrentHashMap<Integer, Result> RESULTS =
            new ConcurrentHashMap<Integer, Result>();
    private static final Object IO_LOCK = new Object();
    private static volatile HandlerThread IO_THREAD;
    private static volatile Handler IO;

    private V240ArchiveOpenBridge() {}

    private static final class Result {
        static final int PENDING = 0;
        static final int OK = 1;
        static final int CANCEL = 2;
        static final int ERROR = 3;
        final CountDownLatch done = new CountDownLatch(1);
        volatile int state = PENDING;
        volatile String value = "";
    }

    public static int begin() {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return -1;
        final int id = NEXT_ID.incrementAndGet();
        RESULTS.put(id, new Result());
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    ensureFragment(owner).launch(id);
                } catch (Throwable error) {
                    fail(id, error);
                }
            }
        });
        return id;
    }

    public static void cancel(int id) { complete(id, Result.CANCEL, ""); }

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
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            // Several Android document providers report community ZIPs as octet-stream, so use
            // a broad type and validate the selected filename/content locally.
            intent.setType("*/*");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
            startActivityForResult(intent, PICK_FILE);
        }

        @Override public void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
            if (requestCode != PICK_FILE) return;
            final int id = pendingId;
            pendingId = -1;
            if (id <= 0) return;
            if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
                cancel(id);
                return;
            }
            Activity owner = getActivity();
            if (owner == null) {
                fail(id, new IllegalStateException("picker Activity unavailable"));
                return;
            }
            final Context app = owner.getApplicationContext();
            final Uri uri = data.getData();
            final int flags = data.getFlags();
            io().post(new Runnable() {
                @Override public void run() { importSelected(app, id, uri, flags); }
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

    private static void importSelected(Context context, int id, Uri uri, int grantFlags) {
        File session = null;
        boolean success = false;
        try {
            if (!isPending(id)) return;
            persist(context, uri, grantFlags);
            ContentResolver resolver = context.getContentResolver();
            String name = displayName(resolver, uri, "level.zip");
            session = new File(context.getFilesDir(), "v240-working/archive-" + UUID.randomUUID());
            if (!session.mkdirs() && !session.isDirectory()) {
                throw new IllegalStateException("could not create archive session");
            }
            File local = new File(session, safeName(name));
            long total = 0L;
            byte[] buffer = new byte[BUFFER_BYTES];
            try (InputStream in = resolver.openInputStream(uri);
                 FileOutputStream out = new FileOutputStream(local, false)) {
                if (in == null) throw new IllegalStateException("document provider returned no input stream");
                int count;
                while ((count = in.read(buffer)) != -1) {
                    if (!isPending(id)) return;
                    if (count == 0) continue;
                    total += count;
                    if (total > MAX_TRANSPORT_BYTES) {
                        throw new IllegalStateException("selected bundle exceeds 1 GiB safety limit");
                    }
                    out.write(buffer, 0, count);
                }
                out.flush();
                out.getFD().sync();
            }
            if (!isPending(id)) return;

            String lower = name.toLowerCase(Locale.US);
            File result;
            if (lower.endsWith(".zip") || lower.endsWith(".adozip")) {
                result = V240ZipLevelImporter.importArchive(local);
            } else if (lower.endsWith(".adofai")) {
                // Direct chart fallback. Asset-complete opens should use the folder route.
                result = local;
            } else {
                throw new IllegalArgumentException("select an .adofai, .zip, or .adozip map bundle");
            }
            success = complete(id, Result.OK, result.getAbsolutePath());
        } catch (Throwable error) {
            if (isPending(id)) fail(id, error);
        } finally {
            if (!success && session != null) deleteRecursively(session);
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

    private static void fail(int id, Throwable error) {
        Log.e(TAG, "archive open failed id=" + id, error);
        String message = error == null ? "unknown error" : error.getMessage();
        complete(id, Result.ERROR, message == null ? error.getClass().getSimpleName() : message);
    }

    private static Handler io() {
        Handler handler = IO;
        if (handler != null) return handler;
        synchronized (IO_LOCK) {
            handler = IO;
            if (handler != null) return handler;
            HandlerThread thread = new HandlerThread(
                    "adofai-v240-archive", Process.THREAD_PRIORITY_BACKGROUND);
            thread.start();
            IO_THREAD = thread;
            handler = new Handler(thread.getLooper());
            IO = handler;
            return handler;
        }
    }

    private static void persist(Context context, Uri uri, int flags) {
        int allowed = flags & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (allowed == 0) return;
        try { context.getContentResolver().takePersistableUriPermission(uri, allowed); }
        catch (SecurityException ignored) {}
    }

    private static String displayName(ContentResolver resolver, Uri uri, String fallback) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String value = cursor.getString(index);
                    if (value != null && value.trim().length() > 0) return value.trim();
                }
            }
        } catch (Throwable ignored) {
        } finally {
            if (cursor != null) cursor.close();
        }
        return fallback;
    }

    private static String safeName(String raw) {
        String value = raw == null ? "level.zip" : raw.trim();
        StringBuilder out = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); ++i) {
            char ch = value.charAt(i);
            if (ch < 0x20 || ch == '/' || ch == '\\') out.append('_');
            else out.append(ch);
        }
        value = out.toString();
        while (value.startsWith(".")) value = value.substring(1);
        if (value.length() == 0) value = "level.zip";
        if (value.length() > 180) value = value.substring(value.length() - 180);
        return value;
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

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteRecursively(child);
        }
        if (!file.delete()) file.deleteOnExit();
    }
}
