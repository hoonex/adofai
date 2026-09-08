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

    static final class V240Runtime {
        final File runtimeDex;
        final File nativeLibrary;

        V240Runtime(File runtimeDex, File nativeLibrary) {
            this.runtimeDex = runtimeDex;
            this.nativeLibrary = nativeLibrary;
        }
    }

    /** Legacy payload staging retained for the unrelated newer patch paths. */
    static Payload stage(Context context, File workDir) throws Exception {
        ensureDirectory(workDir);
        File dex = new File(workDir, "payload-classes2.dex");
        copyAsset(context, "payload/classes2.dex", dex);
        if (dex.length() < 1024L) {
            throw new IllegalStateException("embedded editor payload is missing or unexpectedly small");
        }

        File lib = null;
        try {
            File candidate = new File(workDir, "payload-libOctober.so");
            copyAsset(context, "payload/libOctober.so", candidate);
            if (candidate.length() >= 4096L) lib = candidate;
        } catch (java.io.FileNotFoundException ignored) {
        } catch (java.io.IOException ignored) {
        }
        return new Payload(dex, lib);
    }

    static File stageV240PickerDex(Context context, File workDir) throws Exception {
        ensureDirectory(workDir);
        File dex = new File(workDir, "v240-picker-payload.dex");
        copyAsset(context, "payload/classes2.dex", dex);
        if (dex.length() < 1024L) {
            throw new IllegalStateException("embedded 2.4 picker payload is missing or unexpectedly small");
        }
        return dex;
    }

    /**
     * Stages the exact runtime built for the user's historical 2.4.0 Custom APK:
     * a secondary DEX containing V240Bootstrap/FileSelector and a new arm64 libv240fix.so.
     */
    static V240Runtime stageV240FixedRuntime(Context context, File workDir) throws Exception {
        ensureDirectory(workDir);
        File dex = new File(workDir, "v240-fixed-runtime.dex");
        File lib = new File(workDir, "libv240fix.so");
        copyAsset(context, "payload/v240-fixed-runtime.dex", dex);
        copyAsset(context, "payload/libv240fix.so", lib);
        if (dex.length() < 4096L) {
            throw new IllegalStateException("embedded v2.4 fixed runtime DEX is missing or unexpectedly small");
        }
        if (lib.length() < 64L * 1024L) {
            throw new IllegalStateException("embedded libv240fix.so is missing or unexpectedly small");
        }
        return new V240Runtime(dex, lib);
    }

    private static void ensureDirectory(File workDir) {
        if (!workDir.exists() && !workDir.mkdirs()) {
            throw new IllegalStateException("could not create payload work directory: " + workDir);
        }
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
