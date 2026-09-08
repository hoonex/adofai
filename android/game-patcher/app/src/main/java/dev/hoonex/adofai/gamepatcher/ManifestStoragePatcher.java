package dev.hoonex.adofai.gamepatcher;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;

import java.io.File;
import java.io.FileOutputStream;

/**
 * Patches a binary AndroidManifest.xml without apktool.
 *
 * The legacy mobile editor file browser uses raw shared-storage paths. On
 * Android 11+ that path requires MANAGE_EXTERNAL_STORAGE to be declared;
 * READ/WRITE are retained for older Android versions.
 */
final class ManifestStoragePatcher {
    static final String READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE";
    static final String WRITE_EXTERNAL_STORAGE = "android.permission.WRITE_EXTERNAL_STORAGE";
    static final String MANAGE_EXTERNAL_STORAGE = "android.permission.MANAGE_EXTERNAL_STORAGE";

    private static final String[] REQUIRED = {
        READ_EXTERNAL_STORAGE,
        WRITE_EXTERNAL_STORAGE,
        MANAGE_EXTERNAL_STORAGE,
    };

    static void patch(File source, File output) throws Exception {
        patch(source, output, InstalledGame.PACKAGE_NAME);
    }

    static void patch(File source, File output, String expectedPackage) throws Exception {
        if (expectedPackage == null || expectedPackage.trim().isEmpty()) {
            throw new IllegalArgumentException("expected package is required");
        }
        AndroidManifestBlock manifest = AndroidManifestBlock.load(source);
        assertPackage(manifest, expectedPackage);

        for (String permission : REQUIRED) {
            manifest.addUsesPermission(permission);
        }
        manifest.refreshFull();

        byte[] bytes = manifest.getBytes();
        if (bytes == null || bytes.length == 0) {
            throw new IllegalStateException("manifest encoder returned no data");
        }
        File parent = output.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("could not create manifest output directory: " + parent);
        }
        try (FileOutputStream out = new FileOutputStream(output, false)) {
            out.write(bytes);
            out.getFD().sync();
        }

        AndroidManifestBlock verify = AndroidManifestBlock.load(output);
        assertPackage(verify, expectedPackage);
        assertPermissions(verify);
    }

    static void assertPermissions(AndroidManifestBlock manifest) {
        for (String permission : REQUIRED) {
            if (manifest.getUsesPermission(permission) == null) {
                throw new IllegalStateException("patched manifest missing permission: " + permission);
            }
        }
    }

    private static void assertPackage(AndroidManifestBlock manifest, String expectedPackage) {
        String packageName = manifest.getPackageName();
        if (!expectedPackage.equals(packageName)) {
            throw new IllegalStateException(
                "refusing manifest for unexpected package: " + String.valueOf(packageName)
            );
        }
    }

    private ManifestStoragePatcher() {}
}
