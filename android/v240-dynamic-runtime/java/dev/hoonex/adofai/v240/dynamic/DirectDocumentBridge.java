package dev.hoonex.adofai.v240.dynamic;

import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Hot-swappable SAF document importer used by the v2.4 native SFB hook.
 *
 * Unlike the embedded V240AndroidBridge direct-document path, compatibility rewrites are
 * completed before the legacy save FileObserver is attached. For .adofai opens the system
 * picker also permits selecting sibling song/image files in the same operation; only the chart
 * path is returned to the game when the original SFB call was single-select.
 */
public final class DirectDocumentBridge {
    private static final String TAG = "ADOFAI.V240DirectDynamic";
    private static final String FRAGMENT_TAG = "adofai-v240-dynamic-document";
    private static final int PICK_DOCUMENT = 9342;
    private static final int MAX_FILES = 128;
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int BUFFER_BYTES = 256 * 1024;
    private static final char PATH_SEPARATOR = '\u001f';

    private static final AtomicInteger NEXT_ID = new AtomicInteger(74000);
    private static final ConcurrentHashMap<Integer, Result> RESULTS =
            new ConcurrentHashMap<Integer, Result>();

    private static final Object COMPAT_LOCK = new Object();
    private static volatile boolean COMPAT_INITIALIZED;
    private static Method BACKPORT;
    private static Method HALL_FIX;
    private static Method OPAQUE_PREPARE;
    private static Method MAP_REPAIR;
    private static Method BIND_SAVE;

    private static volatile String LAST_DIAGNOSTICS =
            "directBridge=ready\ndirectImports=0\ndirectLastState=0\n";
    private static final AtomicInteger IMPORTS = new AtomicInteger();

    private DirectDocumentBridge() {}

    private static final class Result {
        static final int PENDING = 0;
        static final int OK = 1;
        static final int CANCEL = 2;
        static final int ERROR = 3;
        final CountDownLatch done = new CountDownLatch(1);
        volatile int state = PENDING;
        volatile String value = "";
    }

    private static final class Copied {
        final Uri uri;
        final File file;
        Copied(Uri uri, File file) { this.uri = uri; this.file = file; }
    }

    private static final class ImportStats {
        int filesCopied;
        int chartCount;
        int backportChanged;
        int hallChanged;
        int opaquePrepared;
        int mapRepairs;
        int saveBound;
        int errors;
        boolean bundleMode;
        boolean multiAssetSelection;
    }

    public static int begin(String mime, String extensions, boolean requestedMultiselect) {
        final Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return -1;
        final int id = NEXT_ID.incrementAndGet();
        RESULTS.put(id, new Result());
        final String safeMime = mime == null || mime.length() == 0 ? "*/*" : mime;
        final String safeExtensions = extensions == null ? "" : extensions;
        final boolean bundleMode = isOnlyLevelExtension(safeExtensions);
        final boolean allowMultiple = requestedMultiselect || bundleMode;
        owner.runOnUiThread(new Runnable() {
            @Override public void run() {
                try {
                    PickerFragment fragment = ensureFragment(owner);
                    fragment.launch(id, safeMime, safeExtensions,
                            requestedMultiselect, allowMultiple, bundleMode);
                } catch (Throwable error) {
                    Log.e(TAG, "document picker launch failed", error);
                    complete(id, Result.ERROR, safeMessage(error));
                }
            }
        });
        return id;
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
        int state = result.state;
        String value = result.value;
        RESULTS.remove(id, result);
        if (state == Result.OK) return "O:" + value;
        if (state == Result.CANCEL) return "C:";
        return "E:" + value;
    }

    public static String diagnostics() {
        return LAST_DIAGNOSTICS;
    }

    public static final class PickerFragment extends Fragment {
        private int pendingId = -1;
        private String pendingExtensions = "";
        private boolean requestedMultiselect;
        private boolean bundleMode;

        @Override public void onCreate(Bundle state) {
            super.onCreate(state);
            setRetainInstance(true);
            if (state != null) {
                pendingId = state.getInt("requestId", -1);
                pendingExtensions = state.getString("extensions", "");
                requestedMultiselect = state.getBoolean("requestedMulti", false);
                bundleMode = state.getBoolean("bundleMode", false);
            }
        }

        @Override public void onSaveInstanceState(Bundle outState) {
            outState.putInt("requestId", pendingId);
            outState.putString("extensions", pendingExtensions);
            outState.putBoolean("requestedMulti", requestedMultiselect);
            outState.putBoolean("bundleMode", bundleMode);
            super.onSaveInstanceState(outState);
        }

        void launch(int id, String mime, String extensions,
                    boolean requestedMulti, boolean allowMultiple, boolean levelBundleMode) {
            if (pendingId > 0 && pendingId != id) complete(pendingId, Result.CANCEL, "");
            pendingId = id;
            pendingExtensions = extensions;
            requestedMultiselect = requestedMulti;
            bundleMode = levelBundleMode;
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType(mime == null || mime.length() == 0 ? "*/*" : mime);
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, allowMultiple);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
            startActivityForResult(intent, PICK_DOCUMENT);
        }

        @Override public void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
            if (requestCode != PICK_DOCUMENT) return;
            final int id = pendingId;
            final String extensions = pendingExtensions;
            final boolean requestedMulti = requestedMultiselect;
            final boolean levelBundleMode = bundleMode;
            pendingId = -1;
            pendingExtensions = "";
            requestedMultiselect = false;
            bundleMode = false;
            if (id <= 0) return;
            if (resultCode != Activity.RESULT_OK || data == null) {
                complete(id, Result.CANCEL, "");
                return;
            }
            final Activity owner = getActivity();
            if (owner == null) {
                complete(id, Result.ERROR, "picker Activity unavailable");
                return;
            }
            final ArrayList<Uri> uris = extractUris(data);
            final int grantFlags = data.getFlags();
            if (uris.isEmpty()) {
                complete(id, Result.ERROR, "no document selected");
                return;
            }
            new Thread(new Runnable() {
                @Override public void run() {
                    importDocuments(owner.getApplicationContext(), id, uris, grantFlags,
                            extensions, requestedMulti, levelBundleMode);
                }
            }, "adofai-v240-dynamic-import").start();
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

    private static ArrayList<Uri> extractUris(Intent data) {
        ArrayList<Uri> uris = new ArrayList<Uri>();
        ClipData clip = data.getClipData();
        if (clip != null) {
            int count = Math.min(MAX_FILES, clip.getItemCount());
            for (int i = 0; i < count; i++) {
                Uri uri = clip.getItemAt(i).getUri();
                if (uri != null && !uris.contains(uri)) uris.add(uri);
            }
        }
        Uri single = data.getData();
        if (single != null && !uris.contains(single)) uris.add(single);
        return uris;
    }

    private static void importDocuments(Context context, int id, List<Uri> uris, int grantFlags,
                                        String extensions, boolean requestedMulti,
                                        boolean bundleMode) {
        File root = null;
        ImportStats stats = new ImportStats();
        stats.bundleMode = bundleMode;
        stats.multiAssetSelection = uris.size() > 1;
        boolean success = false;
        try {
            if (uris.size() > MAX_FILES) throw new IllegalStateException("too many selected files");
            root = new File(context.getFilesDir(), "v240-dynamic-working/doc-" + UUID.randomUUID());
            if (!root.mkdirs() && !root.isDirectory()) {
                throw new IllegalStateException("working directory could not be created");
            }
            ArrayList<Copied> copied = new ArrayList<Copied>();
            long total = 0L;
            ContentResolver resolver = context.getContentResolver();
            for (Uri uri : uris) {
                if (uri == null) continue;
                persist(context, uri, grantFlags);
                String name = displayName(resolver, uri, "file");
                File target = uniqueChild(root, sanitizeName(name));
                try (InputStream in = requireInput(resolver, uri);
                     OutputStream out = new FileOutputStream(target)) {
                    total += copyBounded(in, out, MAX_BYTES - total);
                }
                if (total > MAX_BYTES) throw new IllegalStateException("selected files exceed import limit");
                copied.add(new Copied(uri, target));
                stats.filesCopied++;
            }
            if (copied.isEmpty()) throw new IllegalArgumentException("no readable document selected");

            ensureCompat(context);
            Copied chosenChart = null;
            int bestRank = Integer.MAX_VALUE;
            for (Copied item : copied) {
                if (!isLevelFile(item.file)) continue;
                stats.chartCount++;
                int rank = chartRank(item.file.getName());
                if (rank < bestRank) {
                    bestRank = rank;
                    chosenChart = item;
                }
                if (invokeBoolean(BACKPORT, item.file)) stats.backportChanged++;
                if (invokeBoolean(HALL_FIX, item.file)) stats.hallChanged++;
                if (invokeBoolean(OPAQUE_PREPARE, item.file)) stats.opaquePrepared++;
            }
            for (Copied item : copied) {
                if (!isLevelFile(item.file)) continue;
                invokeVoid(MAP_REPAIR, item.file);
                stats.mapRepairs++;
            }

            if (bundleMode && chosenChart == null) {
                throw new IllegalArgumentException("selected documents do not contain an .adofai level");
            }
            if (chosenChart != null &&
                    (grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
                BIND_SAVE.invoke(null, context, chosenChart.uri, chosenChart.file);
                stats.saveBound = 1;
            }

            String encoded;
            if (!requestedMulti && chosenChart != null) {
                encoded = chosenChart.file.getAbsolutePath();
            } else if (!requestedMulti) {
                encoded = copied.get(0).file.getAbsolutePath();
            } else {
                StringBuilder paths = new StringBuilder();
                if (chosenChart != null) paths.append(chosenChart.file.getAbsolutePath());
                for (Copied item : copied) {
                    if (chosenChart != null && item.file.equals(chosenChart.file)) continue;
                    if (paths.length() > 0) paths.append(PATH_SEPARATOR);
                    paths.append(item.file.getAbsolutePath());
                }
                encoded = paths.toString();
            }
            success = complete(id, Result.OK, encoded);
        } catch (Throwable error) {
            stats.errors++;
            Log.e(TAG, "dynamic document import failed", error);
            complete(id, Result.ERROR, safeMessage(error));
        } finally {
            int importNumber = IMPORTS.incrementAndGet();
            LAST_DIAGNOSTICS = diagnosticsFor(importNumber, success, stats);
            if (!success && root != null) deleteRecursively(root);
        }
    }

    private static void ensureCompat(Context context) throws Exception {
        if (COMPAT_INITIALIZED) return;
        synchronized (COMPAT_LOCK) {
            if (COMPAT_INITIALIZED) return;
            ClassLoader loader = context.getClassLoader();
            Class<?> backport = Class.forName("com.unity3d.player.V240ChartBackport", true, loader);
            Class<?> hall = Class.forName("com.unity3d.player.V240HallLegacyFix", true, loader);
            Class<?> opaque = Class.forName("com.unity3d.player.V240OpaqueEventBridge", true, loader);
            Class<?> map = Class.forName("com.unity3d.player.V240MapCompatibility", true, loader);
            Class<?> bridge = Class.forName("com.unity3d.player.V240AndroidBridge", true, loader);
            BACKPORT = backport.getDeclaredMethod("backportForV240", File.class);
            HALL_FIX = hall.getDeclaredMethod("applyIfNeeded", File.class);
            OPAQUE_PREPARE = opaque.getDeclaredMethod("prepareForV240", File.class);
            MAP_REPAIR = map.getDeclaredMethod("repairMap", File.class);
            BIND_SAVE = bridge.getDeclaredMethod("bindSave", Context.class, Uri.class, File.class);
            BACKPORT.setAccessible(true);
            HALL_FIX.setAccessible(true);
            OPAQUE_PREPARE.setAccessible(true);
            MAP_REPAIR.setAccessible(true);
            BIND_SAVE.setAccessible(true);
            COMPAT_INITIALIZED = true;
        }
    }

    private static boolean invokeBoolean(Method method, File file) throws Exception {
        Object value = method.invoke(null, file);
        return value instanceof Boolean && ((Boolean) value).booleanValue();
    }

    private static void invokeVoid(Method method, File file) throws Exception {
        method.invoke(null, file);
    }

    private static boolean complete(int id, int state, String value) {
        Result result = RESULTS.get(id);
        if (result == null) return false;
        synchronized (result) {
            if (result.state != Result.PENDING) return false;
            result.state = state;
            result.value = value == null ? "" : value;
        }
        result.done.countDown();
        return true;
    }

    private static boolean isOnlyLevelExtension(String raw) {
        if (raw == null) return false;
        String[] parts = raw.toLowerCase(Locale.US).split("[,;|\\s]+");
        int count = 0;
        for (String part : parts) {
            String value = normalizeExtension(part);
            if (value.length() == 0) continue;
            count++;
            if (!"adofai".equals(value)) return false;
        }
        return count == 1;
    }

    private static String normalizeExtension(String raw) {
        if (raw == null) return "";
        String value = raw.trim().toLowerCase(Locale.US);
        while (value.startsWith("*.") || value.startsWith(".")) {
            value = value.startsWith("*.") ? value.substring(2) : value.substring(1);
        }
        return value;
    }

    private static boolean isLevelFile(File file) {
        return file != null && file.isFile() &&
                file.getName().toLowerCase(Locale.US).endsWith(".adofai");
    }

    private static int chartRank(String name) {
        if (name == null) return Integer.MAX_VALUE;
        String lower = name.toLowerCase(Locale.US);
        if ("level.adofai".equals(lower)) return 0;
        if ("main.adofai".equals(lower)) return 1;
        return lower.endsWith(".adofai") ? 2 : Integer.MAX_VALUE;
    }

    private static void persist(Context context, Uri uri, int flags) {
        int allowed = flags & (Intent.FLAG_GRANT_READ_URI_PERMISSION |
                Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if (allowed == 0) return;
        try {
            context.getContentResolver().takePersistableUriPermission(uri, allowed);
        } catch (Throwable ignored) {
        }
    }

    private static InputStream requireInput(ContentResolver resolver, Uri uri) throws Exception {
        InputStream in = resolver.openInputStream(uri);
        if (in == null) throw new IllegalStateException("document provider returned no input stream");
        return in;
    }

    private static long copyBounded(InputStream in, OutputStream out, long remaining) throws Exception {
        if (remaining < 0L) throw new IllegalStateException("import size limit exceeded");
        byte[] buffer = new byte[BUFFER_BYTES];
        long copied = 0L;
        int count;
        while ((count = in.read(buffer)) != -1) {
            copied += count;
            if (copied > remaining) throw new IllegalStateException("import size limit exceeded");
            out.write(buffer, 0, count);
        }
        out.flush();
        return copied;
    }

    private static String displayName(ContentResolver resolver, Uri uri, String fallback) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[] {OpenableColumns.DISPLAY_NAME},
                    null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String value = cursor.getString(index);
                    if (value != null && value.trim().length() > 0) return value;
                }
            }
        } catch (Throwable ignored) {
        } finally {
            if (cursor != null) cursor.close();
        }
        return fallback;
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

    private static String sanitizeName(String raw) {
        if (raw == null || raw.length() == 0) return "file";
        String value = raw.replace('/', '_').replace('\\', '_').replace('\u0000', '_').trim();
        if (value.length() == 0 || ".".equals(value) || "..".equals(value)) return "file";
        if (value.length() > 180) value = value.substring(0, 180);
        return value;
    }

    private static Activity currentActivity() {
        try {
            Class<?> unityPlayer = Class.forName("com.unity3d.player.UnityPlayer");
            Field current = unityPlayer.getField("currentActivity");
            Object value = current.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static String diagnosticsFor(int imports, boolean success, ImportStats stats) {
        return "directBridge=dynamic-document-v1\n" +
                "directImports=" + imports + "\n" +
                "directLastState=" + (success ? 4 : 2) + "\n" +
                "directFilesCopied=" + stats.filesCopied + "\n" +
                "directChartCount=" + stats.chartCount + "\n" +
                "directBundleMode=" + (stats.bundleMode ? 1 : 0) + "\n" +
                "directMultiAssetSelection=" + (stats.multiAssetSelection ? 1 : 0) + "\n" +
                "directBackportChanged=" + stats.backportChanged + "\n" +
                "directHallChanged=" + stats.hallChanged + "\n" +
                "directOpaquePrepared=" + stats.opaquePrepared + "\n" +
                "directMapRepairs=" + stats.mapRepairs + "\n" +
                "directSaveBound=" + stats.saveBound + "\n" +
                "directImportErrors=" + stats.errors + "\n";
    }

    private static String safeMessage(Throwable error) {
        if (error == null) return "unknown error";
        String message = error.getMessage();
        if (message == null || message.length() == 0) message = error.getClass().getSimpleName();
        if (message.length() > 240) message = message.substring(0, 240);
        return message;
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
