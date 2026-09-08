package dev.hoonex.adofai.gamepatcher;

import com.android.zipflinger.FullFileSource;
import com.android.zipflinger.ZipArchive;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class ApkMutator {
    private static final int BUFFER = 1024 * 1024;

    static void mutateBase(File sourceApk, File outputApk, File payloadDex, File workDir) throws Exception {
        copyFile(sourceApk, outputApk);
        File originalDex = new File(workDir, "base-classes.dex");
        File patchedDex = new File(workDir, "base-classes-patched.dex");
        File originalManifest = new File(workDir, "base-AndroidManifest.xml");
        File patchedManifest = new File(workDir, "base-AndroidManifest-patched.xml");

        extractEntry(sourceApk, "classes.dex", originalDex);
        DexBootstrapPatcher.patch(originalDex, payloadDex, patchedDex);

        extractEntry(sourceApk, "AndroidManifest.xml", originalManifest);
        ManifestStoragePatcher.patch(originalManifest, patchedManifest);

        try (ZipArchive zip = new ZipArchive(outputApk.toPath())) {
            deleteSignatureEntries(zip);
            zip.delete("AndroidManifest.xml");
            zip.delete("classes.dex");
            zip.delete("classes2.dex");

            FullFileSource manifest = new FullFileSource(
                patchedManifest.toPath(), "AndroidManifest.xml", Deflater.NO_COMPRESSION
            );
            manifest.align(4);
            zip.add(manifest);

            FullFileSource primary = new FullFileSource(
                patchedDex.toPath(), "classes.dex", Deflater.NO_COMPRESSION
            );
            primary.align(4);
            zip.add(primary);

            FullFileSource secondary = new FullFileSource(
                payloadDex.toPath(), "classes2.dex", Deflater.NO_COMPRESSION
            );
            secondary.align(4);
            zip.add(secondary);
        }
    }

    static void mutateArm64(File sourceApk, File outputApk, File nativePayload) throws Exception {
        copyFile(sourceApk, outputApk);
        try (ZipArchive zip = new ZipArchive(outputApk.toPath())) {
            deleteSignatureEntries(zip);
            zip.delete("lib/arm64-v8a/libOctober.so");
            FullFileSource library = new FullFileSource(
                nativePayload.toPath(), "lib/arm64-v8a/libOctober.so", Deflater.NO_COMPRESSION
            );
            library.align(16 * 1024);
            zip.add(library);
        }
    }

    static void mutateV240Single(
        File sourceApk,
        File outputApk,
        File payloadDex,
        File nativePayload,
        File workDir,
        String packageName,
        String selectorDexEntry
    ) throws Exception {
        copyFile(sourceApk, outputApk);

        File originalManifest = new File(workDir, "v240-AndroidManifest.xml");
        File patchedManifest = new File(workDir, "v240-AndroidManifest-patched.xml");
        extractEntry(sourceApk, "AndroidManifest.xml", originalManifest);
        ManifestStoragePatcher.patch(originalManifest, patchedManifest, packageName);

        File originalSelectorDex = new File(workDir, "v240-selector-source.dex");
        File patchedSelectorDex = new File(workDir, "v240-selector-patched.dex");
        extractEntry(sourceApk, selectorDexEntry, originalSelectorDex);
        DexOverlayPatcher.patch(originalSelectorDex, payloadDex, patchedSelectorDex);

        try (ZipArchive zip = new ZipArchive(outputApk.toPath())) {
            deleteSignatureEntries(zip);
            zip.delete("AndroidManifest.xml");
            zip.delete(selectorDexEntry);
            zip.delete("lib/arm64-v8a/libOctober.so");

            FullFileSource manifest = new FullFileSource(
                patchedManifest.toPath(), "AndroidManifest.xml", Deflater.NO_COMPRESSION
            );
            manifest.align(4);
            zip.add(manifest);

            FullFileSource selectorDex = new FullFileSource(
                patchedSelectorDex.toPath(), selectorDexEntry, Deflater.NO_COMPRESSION
            );
            selectorDex.align(4);
            zip.add(selectorDex);

            FullFileSource library = new FullFileSource(
                nativePayload.toPath(), "lib/arm64-v8a/libOctober.so", Deflater.NO_COMPRESSION
            );
            library.align(16 * 1024);
            zip.add(library);
        }
    }

    static void assertEntry(File apk, String name) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            ZipEntry entry = zip.getEntry(name);
            if (entry == null || entry.getSize() == 0L) {
                throw new IllegalStateException("patched APK missing entry: " + name);
            }
        }
    }

    private static void deleteSignatureEntries(ZipArchive zip) throws Exception {
        List<String> names = new ArrayList<String>(zip.listEntries());
        for (String name : names) {
            String upper = name.toUpperCase(java.util.Locale.US);
            if (upper.startsWith("META-INF/") &&
                (upper.endsWith(".RSA") || upper.endsWith(".DSA") || upper.endsWith(".EC") ||
                 upper.endsWith(".SF") || upper.endsWith("MANIFEST.MF"))) {
                zip.delete(name);
            }
        }
    }

    private static void extractEntry(File apk, String name, File output) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            ZipEntry entry = zip.getEntry(name);
            if (entry == null) throw new IllegalStateException("APK entry not found: " + name);
            try (InputStream in = new BufferedInputStream(zip.getInputStream(entry));
                 OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
                copy(in, out);
            }
        }
    }

    static void copyFile(File source, File output) throws Exception {
        File parent = output.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("could not create directory: " + parent);
        }
        try (InputStream in = new BufferedInputStream(new FileInputStream(source), BUFFER);
             OutputStream out = new BufferedOutputStream(new FileOutputStream(output), BUFFER)) {
            copy(in, out);
        }
        if (output.length() != source.length()) {
            throw new IllegalStateException("file copy length mismatch: " + source.getName());
        }
    }

    static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buffer = new byte[BUFFER];
        int count;
        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        out.flush();
    }

    private ApkMutator() {}
}
