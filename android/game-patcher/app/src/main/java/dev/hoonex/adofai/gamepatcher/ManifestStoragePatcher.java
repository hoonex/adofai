package dev.hoonex.adofai.gamepatcher;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;

import java.io.File;
import java.io.FileOutputStream;

/**
 * Patches the installed game's binary AndroidManifest.xml without apktool.
 *
 * The injected editor's file browser intentionally uses raw shared-storage paths.
 * On Android 11+ that path requires MANAGE_EXTERNAL_STORAGE to be declared by the
 * game before FileSelector can direct the user to the system "All files access"
 * screen. READ/WRITE are retained for the legacy branch on older Android.
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
        AndroidManifestBlock manifest = AndroidManifestBlock.load(source);
        assertPackage(manifest);

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
        assertPackage(verify);
        assertPermissions(verify);
    }

    static void assertPermissions(AndroidManifestBlock manifest) {
        for (String permission : REQUIRED) {
            if (manifest.getUsesPermission(permission) == null) {
                throw new IllegalStateException("patched manifest missing permission: " + permission);
            }
        }
    }

    private static void assertPackage(AndroidManifestBlock manifest) {
        String packageName = manifest.getPackageName();
        if (!InstalledGame.PACKAGE_NAME.equals(packageName)) {
            throw new IllegalStateException(
                "refusing manifest for unexpected package: " + String.valueOf(packageName)
            );
        }
    }

    private ManifestStoragePatcher() {}
}
