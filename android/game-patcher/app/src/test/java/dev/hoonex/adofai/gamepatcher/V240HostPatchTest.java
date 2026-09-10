package dev.hoonex.adofai.gamepatcher;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;

import org.junit.Assume;
import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Host-side exact-source patch path for CI.
 *
 * This deliberately reuses the same binary-manifest, DEX bootstrap, ZIP mutation,
 * and v2.4 runtime payload code as the on-device patcher.  It only runs when the
 * workflow supplies the authoritative APK and prebuilt runtime payload paths.
 */
public final class V240HostPatchTest {
    private static final long SOURCE_BYTES = 370_092_054L;
    private static final String SOURCE_SHA256 =
            "630f519ae1ab3391aad95da90ebc296f4f0f8ae4ea41024ace7349d93926ef30";

    @Test
    public void patchAuthoritativeSourceForCiWhenSupplied() throws Exception {
        String sourcePath = System.getenv("V240_SOURCE_APK");
        String runtimeDexPath = System.getenv("V240_RUNTIME_DEX");
        String nativePath = System.getenv("V240_NATIVE_LIB");
        String outputPath = System.getenv("V240_UNSIGNED_OUTPUT");

        Assume.assumeTrue("host patch is opt-in", sourcePath != null && !sourcePath.isEmpty());
        assertNotNull("V240_RUNTIME_DEX", runtimeDexPath);
        assertNotNull("V240_NATIVE_LIB", nativePath);
        assertNotNull("V240_UNSIGNED_OUTPUT", outputPath);

        File source = new File(sourcePath);
        File runtimeDex = new File(runtimeDexPath);
        File nativeLib = new File(nativePath);
        File output = new File(outputPath);
        File work = new File(output.getParentFile(), "v240-host-patch-work");
        deleteTree(work);
        assertTrue(work.mkdirs());

        assertTrue(source.isFile());
        assertEquals(SOURCE_BYTES, source.length());
        assertEquals(SOURCE_SHA256, sha256(source));
        assertTrue(runtimeDex.isFile() && runtimeDex.length() > 0L);
        assertTrue(nativeLib.isFile() && nativeLib.length() > 0L);

        Map<String, Fingerprint> nativeBefore = snapshotNative(source);
        String packageName;
        File manifestFile = new File(work, "source-manifest.xml");
        ApkMutator.extractEntry(source, "AndroidManifest.xml", manifestFile);
        AndroidManifestBlock manifest = AndroidManifestBlock.load(manifestFile);
        packageName = manifest.getPackageName();
        assertTrue(packageName != null && !packageName.trim().isEmpty());

        try (ZipFile zip = new ZipFile(source)) {
            require(zip, "classes.dex");
            require(zip, "lib/arm64-v8a/libil2cpp.so");
            assertTrue("authoritative source must not already have classes2.dex",
                    zip.getEntry("classes2.dex") == null);
            assertTrue("authoritative source must not already have libv240fix.so",
                    zip.getEntry("lib/arm64-v8a/libv240fix.so") == null);
        }

        ApkMutator.mutateV240Single(
                source, output, runtimeDex, nativeLib, work, packageName
        );

        assertTrue(output.isFile());
        assertTrue("patched output unexpectedly small", output.length() > 300L * 1024L * 1024L);
        try (ZipFile zip = new ZipFile(output)) {
            require(zip, "AndroidManifest.xml");
            require(zip, "classes.dex");
            require(zip, "classes2.dex");
            require(zip, "lib/arm64-v8a/libil2cpp.so");
            require(zip, "lib/arm64-v8a/libv240fix.so");
        }

        File patchedDex = new File(work, "patched-classes.dex");
        ApkMutator.extractEntry(output, "classes.dex", patchedDex);
        assertTrue("Unity bootstrap injection missing",
                V240DexBootstrapPatcher.containsBootstrapInvoke(patchedDex));

        File patchedManifestFile = new File(work, "patched-manifest.xml");
        ApkMutator.extractEntry(output, "AndroidManifest.xml", patchedManifestFile);
        AndroidManifestBlock patchedManifest = AndroidManifestBlock.load(patchedManifestFile);
        assertEquals(packageName, patchedManifest.getPackageName());
        ManifestStoragePatcher.assertV240Picker(patchedManifest);

        assertNativePreserved(output, nativeBefore);
        deleteTree(work);
    }

    private static void require(ZipFile zip, String name) {
        ZipEntry entry = zip.getEntry(name);
        assertNotNull("missing APK entry: " + name, entry);
        assertTrue("empty APK entry: " + name, entry.getSize() != 0L);
    }

    private static Map<String, Fingerprint> snapshotNative(File apk) throws Exception {
        Map<String, Fingerprint> result = new HashMap<String, Fingerprint>();
        try (ZipFile zip = new ZipFile(apk)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String name = entry.getName();
                if (!entry.isDirectory() && name.startsWith("lib/") && name.endsWith(".so")) {
                    result.put(name, new Fingerprint(entry.getSize(), entry.getCrc()));
                }
            }
        }
        assertTrue("source APK has no native libraries", !result.isEmpty());
        return result;
    }

    private static void assertNativePreserved(File apk, Map<String, Fingerprint> before) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            for (Map.Entry<String, Fingerprint> item : before.entrySet()) {
                ZipEntry current = zip.getEntry(item.getKey());
                assertNotNull("original native library removed: " + item.getKey(), current);
                assertEquals(item.getValue().size, current.getSize());
                assertEquals(item.getValue().crc, current.getCrc());
            }
        }
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        if (!file.delete() && file.exists()) {
            throw new IllegalStateException("could not delete " + file);
        }
    }

    private static final class Fingerprint {
        final long size;
        final long crc;

        Fingerprint(long size, long crc) {
            this.size = size;
            this.crc = crc;
        }
    }
}
