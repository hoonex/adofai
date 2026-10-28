package dev.hoonex.adofai.gamepatcher;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.*;

/**
 * Real ZIP mutation tests using synthetic, non-game inputs.
 * These assert that monolithic native injection does not replace Unity binaries.
 */
public class MonolithicApkMutatorTest {
    @Rule public TemporaryFolder temp = new TemporaryFolder();

    private static final String UNITY_ENTRY = "lib/arm64-v8a/libil2cpp.so";
    private static final String OCTOBER_ENTRY = "lib/arm64-v8a/libOctober.so";

    private File syntheticApk(boolean withEditor) throws Exception {
        File apk = temp.newFile("synthetic" + (withEditor ? "-collision" : "-clean") + ".apk");
        try (ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(apk))) {
            add(zip, UNITY_ENTRY, "original-unity-runtime".getBytes(StandardCharsets.UTF_8));
            add(zip, "classes.dex", "original-classes".getBytes(StandardCharsets.UTF_8));
            if (withEditor) add(zip, OCTOBER_ENTRY, "third-party-editor".getBytes(StandardCharsets.UTF_8));
        }
        return apk;
    }

    private File dummyLibrary() throws Exception {
        File file = temp.newFile("libOctober.so");
        byte[] contents = new byte[8192];
        Arrays.fill(contents, (byte) 0x31);
        Files.write(file.toPath(), contents);
        return file;
    }

    private static void add(ZipOutputStream zip, String name, byte[] bytes) throws Exception {
        ZipEntry entry = new ZipEntry(name);
        zip.putNextEntry(entry);
        zip.write(bytes);
        zip.closeEntry();
    }

    private static byte[] readEntry(File zipFile, String name) throws Exception {
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(name);
            assertNotNull("missing entry " + name, entry);
            try (InputStream stream = zip.getInputStream(entry)) {
                return stream.readAllBytes();
            }
        }
    }

    @Test public void injectedNativeLibraryPreservesOriginalUnityBytes() throws Exception {
        File archive = syntheticApk(false);
        byte[] unityBefore = readEntry(archive, UNITY_ENTRY);
        byte[] dexBefore = readEntry(archive, "classes.dex");
        File nativeLib = dummyLibrary();

        ApkMutator.injectArm64IntoExisting(archive, nativeLib);

        assertArrayEquals(Files.readAllBytes(nativeLib.toPath()), readEntry(archive, OCTOBER_ENTRY));
        assertArrayEquals(unityBefore, readEntry(archive, UNITY_ENTRY));
        assertArrayEquals(dexBefore, readEntry(archive, "classes.dex"));
        try (ZipFile zip = new ZipFile(archive)) {
            assertEquals(3, zip.size());
        }
    }

    @Test public void existingNativeLibraryIsNeverOverwritten() throws Exception {
        File archive = syntheticApk(true);
        byte[] before = Files.readAllBytes(archive.toPath());
        try {
            ApkMutator.injectArm64IntoExisting(archive, dummyLibrary());
            fail("expected conflict to fail closed");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("refusing to overwrite"));
        }
        assertArrayEquals(before, Files.readAllBytes(archive.toPath()));
    }

    @Test public void tinyNativePayloadIsRejectedWithoutEditingGame() throws Exception {
        File archive = syntheticApk(false);
        byte[] before = Files.readAllBytes(archive.toPath());
        File bad = temp.newFile("small-library.so");
        Files.write(bad.toPath(), new byte[] {1, 2, 3});
        try {
            ApkMutator.injectArm64IntoExisting(archive, bad);
            fail("expected undersized native library to be rejected");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("missing verified arm64"));
        }
        assertArrayEquals(before, Files.readAllBytes(archive.toPath()));
    }
}
