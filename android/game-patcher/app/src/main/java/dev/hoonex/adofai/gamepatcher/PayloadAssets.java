package dev.hoonex.adofai.gamepatcher;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

final class PayloadAssets {
    static final class Payload {
        final File classes2Dex;
        final File libOctober;

        Payload(File classes2Dex, File libOctober) {
            this.classes2Dex = classes2Dex;
            this.libOctober = libOctober;
        }
    }

    /** Legacy/general payload path retained for the unrelated patcher code. */
    static Payload stage(Context context, File workDir) throws Exception {
        if (!workDir.exists() && !workDir.mkdirs()) {
            throw new IllegalStateException("could not create payload work directory: " + workDir);
        }
        File dex = new File(workDir, "payload-classes2.dex");
        File lib = new File(workDir, "payload-libOctober.so");
        copyAsset(context, "payload/classes2.dex", dex);
        copyAsset(context, "payload/libOctober.so", lib);
        if (dex.length() < 1024L || lib.length() < 4096L) {
            throw new IllegalStateException("embedded editor payload is missing or unexpectedly small");
        }
        return new Payload(dex, lib);
    }

    /**
     * 2.4 historical Custom path: only the optional Java picker payload is staged.
     * No native hook asset is required or read.
     */
    static File stageV240PickerDex(Context context, File workDir) throws Exception {
        if (!workDir.exists() && !workDir.mkdirs()) {
            throw new IllegalStateException("could not create payload work directory: " + workDir);
        }
        File dex = new File(workDir, "v240-picker-payload.dex");
        copyAsset(context, "payload/classes2.dex", dex);
        if (dex.length() < 1024L) {
            throw new IllegalStateException("embedded 2.4 picker payload is missing or unexpectedly small");
        }
        return dex;
    }

    private static void copyAsset(Context context, String asset, File output) throws Exception {
        try (InputStream in = new BufferedInputStream(context.getAssets().open(asset));
             OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
            byte[] buffer = new byte[256 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
            out.flush();
        }
    }

    private PayloadAssets() {}
}
