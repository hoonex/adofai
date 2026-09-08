package com.unity3d.player;

/**
 * Compatibility facade consumed by the native SFB hooks.
 *
 * The historical hook protocol polls isDone/getFilePath.  Keep that ABI small and
 * forward the actual Android work to V240AndroidBridge, which owns SAF and working
 * file synchronization.
 */
public final class FileSelector {
    public static volatile boolean isDone = true;
    private static volatile String filePath = "";
    private static volatile String folderPath = "";
    private static volatile int generation = 0;

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
        isDone = false;
        if (folder) folderPath = ""; else filePath = "";
        if (requestId <= 0) {
            isDone = true;
            return;
        }

        Thread waiter = new Thread(new Runnable() {
            @Override public void run() {
                String result = "";
                try {
                    for (int i = 0; i < 7200; i++) { // ten minutes, picker normally completes far sooner
                        if (generation != myGeneration) return;
                        String state = V240AndroidBridge.poll(requestId);
                        if (state == null || state.length() == 0 || "P".equals(state)) {
                            Thread.sleep(80L);
                            continue;
                        }
                        if (state.startsWith("O:")) result = state.substring(2);
                        break;
                    }
                } catch (Throwable ignored) {
                    result = "";
                }
                if (generation != myGeneration) return;
                if (folder) folderPath = result; else filePath = result;
                isDone = true;
            }
        }, folder ? "adofai-v240-folder" : "adofai-v240-file");
        waiter.setDaemon(true);
        waiter.start();
    }
}
