package com.unity3d.player;

import java.util.Locale;

/**
 * Compatibility facade consumed by the native SFB hooks.
 *
 * The historical hook protocol polls isDone/getFilePath. Keep that native ABI small,
 * but let the Java waiter sleep on a bridge completion signal instead of polling SAF.
 */
public final class FileSelector {
    public static volatile boolean isDone = true;
    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile int generation = 0;
    private static int activeRequestId = -1;

    private FileSelector() {}

    public static void selectFile(String extensions) {
        start(V240AndroidBridge.beginOpen(mimeForExtensions(extensions)), false);
    }

    public static void saveAs(String suggestedName) {
        start(V240AndroidBridge.beginSave(suggestedName, mimeForFilename(suggestedName)), false);
    }

    public static void selectFolder() {
        start(V240AndroidBridge.beginFolder(), true);
    }

    public static String getFilePath() { return filePath; }
    public static String getFolderPath() { return folderPath; }

    private static synchronized void start(final int requestId, final boolean folder) {
        final int myGeneration = ++generation;
        final int previousRequestId = activeRequestId;
        activeRequestId = requestId > 0 ? requestId : -1;
        isDone = false;
        if (folder) folderPath = ""; else filePath = "";

        if (previousRequestId > 0 && previousRequestId != requestId) {
            V240AndroidBridge.cancel(previousRequestId);
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
                    String state = V240AndroidBridge.await(requestId, 600_000L);
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
        }, folder ? "adofai-v240-folder" : "adofai-v240-file");
        waiter.setDaemon(true);
        waiter.start();
    }

    private static String mimeForFilename(String name) {
        if (name == null) return "application/octet-stream";
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot + 1 >= name.length()) return "application/octet-stream";
        return mimeForExtensions(name.substring(dot + 1));
    }

    private static String mimeForExtensions(String raw) {
        if (raw == null) return "*/*";
        String value = raw.trim().toLowerCase(Locale.US);
        if (value.startsWith("*.")) value = value.substring(2);
        else if (value.startsWith(".")) value = value.substring(1);
        // An ExtensionFilter[] overload can contain several unrelated types. The native
        // bridge deliberately serializes that as a comma-separated broad fallback; SAF
        // cannot reliably filter custom .adofai plus media by extension, so keep */*.
        if (value.length() == 0 || value.indexOf(',') >= 0 || value.indexOf(';') >= 0 ||
                value.indexOf('|') >= 0 || value.indexOf(' ') >= 0) return "*/*";
        if ("png".equals(value)) return "image/png";
        if ("jpg".equals(value) || "jpeg".equals(value)) return "image/jpeg";
        if ("ogg".equals(value)) return "audio/ogg";
        if ("mp3".equals(value)) return "audio/mpeg";
        if ("wav".equals(value)) return "audio/wav";
        if ("zip".equals(value)) return "application/zip";
        if ("json".equals(value)) return "application/json";
        // Custom .adofai files are reported as different MIME types by different SAF
        // providers, so restricting them would hide valid levels on some devices.
        if ("adofai".equals(value)) return "*/*";
        return "*/*";
    }
}
