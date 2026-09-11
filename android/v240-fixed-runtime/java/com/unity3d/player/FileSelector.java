package com.unity3d.player;

import java.util.ArrayList;
import java.util.Locale;

/** Compatibility facade consumed by the native SFB hooks. */
public final class FileSelector {
    public static volatile boolean isDone = true;
    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile int generation = 0;
    private static int activeRequestId = -1;

    private FileSelector() {}

    /** Legacy ABI retained for older native payloads. */
    public static void selectFile(String extensions) {
        selectFile(extensions, false);
    }

    /** Exact SFB ABI: preserve caller filters and multiselect intent. */
    public static void selectFile(String extensions, boolean multiselect) {
        // A standalone .adofai almost always references sibling song/image assets. When the
        // caller is specifically opening one level, use a tree picker and mirror the complete
        // map directory so relative paths keep working. Mixed/import filters still use the
        // document picker (ZIP imports are expanded by V240AndroidBridge).
        if (!multiselect && isOnlyLevelExtension(extensions)) {
            start(V240LevelFolderBridge.begin(), false, true);
            return;
        }
        String[] mimeTypes = mimeTypesForExtensions(extensions);
        String primaryMime = mimeTypes != null && mimeTypes.length == 1 ? mimeTypes[0] : "*/*";
        start(V240AndroidBridge.beginOpen(primaryMime, mimeTypes, multiselect), false, false);
    }

    public static void saveAs(String suggestedName) {
        start(V240AndroidBridge.beginSave(suggestedName, mimeForFilename(suggestedName)), false, false);
    }

    public static void selectFolder() {
        start(V240AndroidBridge.beginFolder(), true, false);
    }

    public static String getFilePath() { return filePath; }
    public static String getFolderPath() { return folderPath; }

    private static synchronized void start(final int requestId, final boolean folder,
                                           final boolean levelTree) {
        final int myGeneration = ++generation;
        final int previousRequestId = activeRequestId;
        activeRequestId = requestId > 0 ? requestId : -1;
        isDone = false;
        if (folder) folderPath = ""; else filePath = "";

        if (previousRequestId > 0 && previousRequestId != requestId) {
            // Request id ranges are intentionally distinct, but cancelling both backends keeps
            // transitions race-free without coupling this ABI facade to their internal ranges.
            V240AndroidBridge.cancel(previousRequestId);
            V240LevelFolderBridge.cancel(previousRequestId);
        }
        if (requestId <= 0) {
            activeRequestId = -1;
            isDone = true;
            return;
        }

        Thread waiter = new Thread(new Runnable() {
            @Override public void run() {
                String result = "";
                try {
                    String state = levelTree
                            ? V240LevelFolderBridge.await(requestId, 600_000L)
                            : V240AndroidBridge.await(requestId, 600_000L);
                    if (state != null && state.startsWith("O:")) result = state.substring(2);
                } catch (Throwable ignored) {
                    result = "";
                }
                synchronized (FileSelector.class) {
                    if (generation != myGeneration) return;
                    if (folder) folderPath = result; else filePath = result;
                    activeRequestId = -1;
                    isDone = true;
                }
            }
        }, levelTree ? "adofai-v240-level-tree" :
                (folder ? "adofai-v240-folder" : "adofai-v240-file"));
        waiter.setDaemon(true);
        waiter.start();
    }

    private static boolean isOnlyLevelExtension(String raw) {
        if (raw == null) return false;
        String[] parts = raw.toLowerCase(Locale.US).split("[,;|\\s]+");
        int count = 0;
        for (String part : parts) {
            String extension = normalizeExtension(part);
            if (extension.length() == 0) continue;
            count++;
            if (!"adofai".equals(extension)) return false;
        }
        return count == 1;
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
