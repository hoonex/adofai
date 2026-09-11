package com.unity3d.player;

import java.io.File;
import java.util.ArrayList;
import java.util.Locale;

/** Compatibility facade consumed by the native SFB hooks. */
public final class FileSelector {
    public static volatile boolean isDone = true;
    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile int generation = 0;
    private static int activeRequestId = -1;

    private static final int BACKEND_DOCUMENT = 0;
    private static final int BACKEND_LEVEL_TREE = 1;
    private static final int BACKEND_ARCHIVE = 2;

    private FileSelector() {}

    /** Legacy ABI retained for older native payloads. */
    public static void selectFile(String extensions) {
        selectFile(extensions, false);
    }

    /** Exact SFB ABI: preserve caller filters and multiselect intent. */
    public static void selectFile(String extensions, boolean multiselect) {
        // A standalone .adofai almost always references sibling song/image assets. When the
        // caller is specifically opening one level, use a tree picker and mirror the complete
        // map directory so relative paths keep working.
        if (!multiselect && isOnlyLevelExtension(extensions)) {
            start(V240LevelFolderBridge.begin(), false, BACKEND_LEVEL_TREE, false);
            return;
        }
        // TUF distributes levels as ZIP bundles. If the picker is specifically level/archive
        // oriented, let the archive bridge copy + safely unpack the bundle and return its chart.
        if (!multiselect && isLevelArchiveFilter(extensions)) {
            start(V240ArchiveOpenBridge.begin(), false, BACKEND_ARCHIVE, false);
            return;
        }

        String[] mimeTypes = mimeTypesForExtensions(extensions);
        String primaryMime = mimeTypes != null && mimeTypes.length == 1 ? mimeTypes[0] : "*/*";
        start(V240AndroidBridge.beginOpen(primaryMime, mimeTypes, multiselect), false,
                BACKEND_DOCUMENT, false);
    }

    public static void saveAs(String suggestedName) {
        // Release a tree-backed old chart only after the new Save-As document is successfully
        // prepared, so cancellation never loses the authoritative old save target.
        start(V240AndroidBridge.beginSave(suggestedName, mimeForFilename(suggestedName)), false,
                BACKEND_DOCUMENT, true);
    }

    public static void selectFolder() {
        start(V240AndroidBridge.beginFolder(), true, BACKEND_DOCUMENT, false);
    }

    public static String getFilePath() { return filePath; }
    public static String getFolderPath() { return folderPath; }

    private static synchronized void start(final int requestId, final boolean folder,
                                           final int backend, final boolean releaseLevelOnSuccess) {
        final int myGeneration = ++generation;
        final int previousRequestId = activeRequestId;
        activeRequestId = requestId > 0 ? requestId : -1;
        isDone = false;
        if (folder) folderPath = ""; else filePath = "";

        if (previousRequestId > 0 && previousRequestId != requestId) {
            // Request id ranges are intentionally distinct, but cancelling all backends keeps
            // transitions race-free without coupling this ABI facade to their internal ranges.
            V240AndroidBridge.cancel(previousRequestId);
            V240LevelFolderBridge.cancel(previousRequestId);
            V240ArchiveOpenBridge.cancel(previousRequestId);
        }
        if (requestId <= 0) {
            activeRequestId = -1;
            isDone = true;
            return;
        }

        Thread waiter = new Thread(new Runnable() {
            @Override public void run() {
                String result = "";
                boolean ok = false;
                try {
                    String state;
                    if (backend == BACKEND_LEVEL_TREE) {
                        state = V240LevelFolderBridge.await(requestId, 600_000L);
                    } else if (backend == BACKEND_ARCHIVE) {
                        state = V240ArchiveOpenBridge.await(requestId, 600_000L);
                    } else {
                        state = V240AndroidBridge.await(requestId, 600_000L);
                    }
                    if (state != null && state.startsWith("O:")) {
                        result = state.substring(2);
                        ok = result.length() > 0;
                    }
                } catch (Throwable ignored) {
                    result = "";
                }

                if (ok && !folder && isLocalLevelPath(result)) {
                    // One compatibility pass covers direct documents, tree mirrors and TUF ZIPs.
                    // It only adds aliases inside private storage and fails open on malformed maps.
                    V240MapCompatibility.repairMap(new File(result));
                }

                if (ok && releaseLevelOnSuccess) {
                    V240LevelFolderBridge.releaseActiveLevel(true);
                }

                synchronized (FileSelector.class) {
                    if (generation != myGeneration) return;
                    if (folder) folderPath = result; else filePath = result;
                    activeRequestId = -1;
                    isDone = true;
                }
            }
        }, backend == BACKEND_LEVEL_TREE ? "adofai-v240-level-tree" :
                (backend == BACKEND_ARCHIVE ? "adofai-v240-archive" :
                        (folder ? "adofai-v240-folder" : "adofai-v240-file")));
        waiter.setDaemon(true);
        waiter.start();
    }

    private static boolean isLocalLevelPath(String value) {
        if (value == null || value.indexOf(V240AndroidBridge.PATH_SEPARATOR) >= 0) return false;
        return value.toLowerCase(Locale.US).endsWith(".adofai");
    }

    private static boolean isOnlyLevelExtension(String raw) {
        if (raw == null) return false;
        String[] values = normalizedExtensions(raw);
        return values.length == 1 && "adofai".equals(values[0]);
    }

    private static boolean isLevelArchiveFilter(String raw) {
        if (raw == null) return false;
        String[] values = normalizedExtensions(raw);
        if (values.length == 0) return false;
        boolean hasArchive = false;
        for (String value : values) {
            if ("zip".equals(value) || "adozip".equals(value)) {
                hasArchive = true;
            } else if (!"adofai".equals(value)) {
                return false;
            }
        }
        return hasArchive;
    }

    private static String[] normalizedExtensions(String raw) {
        if (raw == null || raw.trim().length() == 0) return new String[0];
        String[] parts = raw.toLowerCase(Locale.US).split("[,;|\\s]+");
        ArrayList<String> values = new ArrayList<String>();
        for (String part : parts) {
            String extension = normalizeExtension(part);
            if (extension.length() > 0 && !values.contains(extension)) values.add(extension);
        }
        return values.toArray(new String[values.size()]);
    }

    private static String mimeForFilename(String name) {
        if (name == null) return "application/octet-stream";
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot + 1 >= name.length()) return "application/octet-stream";
        String mime = mimeForExtension(name.substring(dot + 1));
        return mime == null ? "application/octet-stream" : mime;
    }

    /**
     * Returns exact SAF MIME types only when every requested extension has a reliable MIME.
     * Custom .adofai or unknown extensions intentionally return null so providers do not hide
     * valid files behind inconsistent vendor MIME mappings.
     */
    private static String[] mimeTypesForExtensions(String raw) {
        if (raw == null || raw.trim().length() == 0) return null;
        String[] parts = raw.toLowerCase(Locale.US).split("[,;|\\s]+");
        ArrayList<String> values = new ArrayList<String>();
        for (String part : parts) {
            String extension = normalizeExtension(part);
            if (extension.length() == 0) continue;
            String mime = mimeForExtension(extension);
            if (mime == null) return null;
            if (!values.contains(mime)) values.add(mime);
        }
        return values.isEmpty() ? null : values.toArray(new String[values.size()]);
    }

    private static String normalizeExtension(String raw) {
        if (raw == null) return "";
        String value = raw.trim().toLowerCase(Locale.US);
        while (value.startsWith("*.") || value.startsWith(".")) {
            value = value.startsWith("*.") ? value.substring(2) : value.substring(1);
        }
        return value;
    }

    private static String mimeForExtension(String raw) {
        String value = normalizeExtension(raw);
        if ("png".equals(value)) return "image/png";
        if ("jpg".equals(value) || "jpeg".equals(value)) return "image/jpeg";
        if ("ogg".equals(value)) return "audio/ogg";
        if ("mp3".equals(value)) return "audio/mpeg";
        if ("wav".equals(value)) return "audio/wav";
        if ("zip".equals(value) || "adozip".equals(value)) return "application/zip";
        if ("json".equals(value)) return "application/json";
        return null;
    }
}
