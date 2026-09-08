package dev.hoonex.adofai.gamepatcher;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;
import com.reandroid.arsc.chunk.xml.ResXmlElement;

import java.io.File;
import java.io.FileOutputStream;

/** Patches binary AndroidManifest.xml files without apktool. */
final class ManifestStoragePatcher {
    static final String READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE";
    static final String WRITE_EXTERNAL_STORAGE = "android.permission.WRITE_EXTERNAL_STORAGE";
    static final String MANAGE_EXTERNAL_STORAGE = "android.permission.MANAGE_EXTERNAL_STORAGE";
    static final String V240_PICKER_ACTIVITY = "com.unity3d.player.V240PickerActivity";

    private static final String[] REQUIRED = {
        READ_EXTERNAL_STORAGE,
        WRITE_EXTERNAL_STORAGE,
        MANAGE_EXTERNAL_STORAGE,
    };

    /** Legacy raw-path storage patch retained for the unrelated newer patch paths. */
    static void patch(File source, File output) throws Exception {
        patch(source, output, InstalledGame.PACKAGE_NAME);
    }

    /** Legacy raw-path storage patch retained for the unrelated newer patch paths. */
    static void patch(File source, File output, String expectedPackage) throws Exception {
        AndroidManifestBlock manifest = loadChecked(source, expectedPackage);
        for (String permission : REQUIRED) {
            manifest.addUsesPermission(permission);
        }
        writeAndVerify(manifest, output, expectedPackage, true, false);
    }

    /**
     * Exact 2.4.0 Custom path. The fixed runtime uses Storage Access Framework and
     * app-private working files, so broad MANAGE_EXTERNAL_STORAGE is deliberately not
     * introduced. Only the internal proxy Activity required to receive SAF results is
     * declared.
     */
    static void patchV240(File source, File output, String expectedPackage) throws Exception {
        AndroidManifestBlock manifest = loadChecked(source, expectedPackage);
        ResXmlElement picker = manifest.getOrCreateActivity(V240_PICKER_ACTIVITY, false);
        picker.getOrCreateAndroidAttribute(
                AndroidManifestBlock.NAME_exported,
                AndroidManifestBlock.ID_exported
        ).setValueAsBoolean(false);
        writeAndVerify(manifest, output, expectedPackage, false, true);
    }

    static void assertPermissions(AndroidManifestBlock manifest) {
        for (String permission : REQUIRED) {
            if (manifest.getUsesPermission(permission) == null) {
                throw new IllegalStateException("patched manifest missing permission: " + permission);
            }
        }
    }

    static void assertV240Picker(AndroidManifestBlock manifest) {
        ResXmlElement picker = manifest.getActivity(V240_PICKER_ACTIVITY, false);
        if (picker == null) {
            throw new IllegalStateException("patched manifest missing internal v2.4 picker activity");
        }
        if (manifest.getActivity(V240_PICKER_ACTIVITY, true) == null) {
            throw new IllegalStateException("v2.4 picker activity lookup failed after encode");
        }
    }

    private static AndroidManifestBlock loadChecked(File source, String expectedPackage) throws Exception {
        if (expectedPackage == null || expectedPackage.trim().isEmpty()) {
            throw new IllegalArgumentException("expected package is required");
        }
        AndroidManifestBlock manifest = AndroidManifestBlock.load(source);
        assertPackage(manifest, expectedPackage);
        return manifest;
    }

    private static void writeAndVerify(
            AndroidManifestBlock manifest,
            File output,
            String expectedPackage,
            boolean verifyPermissions,
            boolean verifyV240Picker
    ) throws Exception {
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
        if (verifyPermissions) assertPermissions(verify);
        if (verifyV240Picker) assertV240Picker(verify);
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
