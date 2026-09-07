package dev.hoonex.adofai.gamepatcher;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public final class ManifestStoragePatcherTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    @Test public void addsRequiredPermissionsWithoutDuplicates() throws Exception {
        File input = tmp.newFile("AndroidManifest.xml");
        File once = tmp.newFile("AndroidManifest-once.xml");
        File twice = tmp.newFile("AndroidManifest-twice.xml");

        AndroidManifestBlock source = new AndroidManifestBlock();
        source.setPackageName("com.fizzd.connectedworlds");
        source.getOrCreateApplicationElement();
        source.refreshFull();
        try (FileOutputStream out = new FileOutputStream(input, false)) {
            out.write(source.getBytes());
        }

        ManifestStoragePatcher.patch(input, once);
        ManifestStoragePatcher.patch(once, twice);

        AndroidManifestBlock patched = AndroidManifestBlock.load(twice);
        assertEquals("com.fizzd.connectedworlds", patched.getPackageName());
        assertNotNull(patched.getUsesPermission(ManifestStoragePatcher.READ_EXTERNAL_STORAGE));
        assertNotNull(patched.getUsesPermission(ManifestStoragePatcher.WRITE_EXTERNAL_STORAGE));
        assertNotNull(patched.getUsesPermission(ManifestStoragePatcher.MANAGE_EXTERNAL_STORAGE));

        List<String> permissions = patched.getUsesPermissions();
        assertEquals(1, Collections.frequency(permissions, ManifestStoragePatcher.READ_EXTERNAL_STORAGE));
        assertEquals(1, Collections.frequency(permissions, ManifestStoragePatcher.WRITE_EXTERNAL_STORAGE));
        assertEquals(1, Collections.frequency(permissions, ManifestStoragePatcher.MANAGE_EXTERNAL_STORAGE));
    }
}
