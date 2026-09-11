package com.unity3d.player;

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

    public static void selectFile(String ignoredExtensions) {
        start(V240AndroidBridge.beginOpen("*/*"), false);
    }

    public static void saveAs(String suggestedName) {
        start(V240AndroidBridge.beginSave(suggestedName, "application/octet-stream"), false);
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
}
