package dev.hoonex.adofai.v240tool;

import com.android.zipflinger.FullFileSource;
import com.android.zipflinger.ZipArchive;
import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;
import com.reandroid.arsc.chunk.xml.ResXmlElement;
import org.jf.dexlib2.DexFileFactory;
import org.jf.dexlib2.Opcode;
import org.jf.dexlib2.Opcodes;
import org.jf.dexlib2.builder.MutableMethodImplementation;
import org.jf.dexlib2.builder.instruction.BuilderInstruction35c;
import org.jf.dexlib2.iface.ClassDef;
import org.jf.dexlib2.iface.DexFile;
import org.jf.dexlib2.iface.Method;
import org.jf.dexlib2.iface.MethodImplementation;
import org.jf.dexlib2.iface.instruction.Instruction;
import org.jf.dexlib2.iface.instruction.ReferenceInstruction;
import org.jf.dexlib2.iface.reference.MethodReference;
import org.jf.dexlib2.immutable.ImmutableClassDef;
import org.jf.dexlib2.immutable.ImmutableDexFile;
import org.jf.dexlib2.immutable.ImmutableMethod;
import org.jf.dexlib2.immutable.reference.ImmutableMethodReference;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Standalone exact-source patcher used outside Android/Gradle. */
public final class V240HostPatchCli {
    private static final long SOURCE_BYTES = 370_092_054L;
    private static final String SOURCE_SHA256 = "630f519ae1ab3391aad95da90ebc296f4f0f8ae4ea41024ace7349d93926ef30";
    private static final String ACTIVITY = "Lcom/unity3d/player/UnityPlayerActivity;";
    private static final String BOOTSTRAP = "Lcom/unity3d/player/V240Bootstrap;";
    private static final String PICKER = "com.unity3d.player.V240PickerActivity";

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("usage: java -jar v240-host-patcher.jar <source.apk> <runtime.dex> <libv240fix.so> <output.apk>");
            System.exit(2);
        }
        File source = new File(args[0]).getCanonicalFile();
        File runtimeDex = new File(args[1]).getCanonicalFile();
        File nativeLib = new File(args[2]).getCanonicalFile();
        File output = new File(args[3]).getCanonicalFile();
        requireFile(source, "source APK");
        requireFile(runtimeDex, "runtime DEX");
        requireFile(nativeLib, "native runtime");
        if (source.length() != SOURCE_BYTES) throw new IllegalStateException("source byte length mismatch: " + source.length());
        String sourceSha = sha256(source);
        if (!SOURCE_SHA256.equals(sourceSha)) throw new IllegalStateException("source SHA-256 mismatch: " + sourceSha);

        File parent = output.getParentFile();
        if (parent != null) parent.mkdirs();
        File work = new File(parent == null ? new File(".") : parent, ".v240-host-work");
        deleteTree(work);
        if (!work.mkdirs()) throw new IllegalStateException("could not create work dir: " + work);

        Map<String, Fingerprint> nativeBefore = snapshotNative(source);
        File manifest = new File(work, "AndroidManifest.xml");
        File patchedManifest = new File(work, "AndroidManifest-patched.xml");
        extract(source, "AndroidManifest.xml", manifest);
        AndroidManifestBlock sourceManifest = AndroidManifestBlock.load(manifest);
        String packageName = sourceManifest.getPackageName();
        if (packageName == null || packageName.trim().isEmpty()) throw new IllegalStateException("source package name missing");
        patchManifest(manifest, patchedManifest, packageName);

        File dex = new File(work, "classes.dex");
        File patchedDex = new File(work, "classes-patched.dex");
        extract(source, "classes.dex", dex);
        patchDex(dex, patchedDex);

        Files.copy(source.toPath(), output.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try (ZipArchive zip = new ZipArchive(output.toPath())) {
            deleteSignatures(zip);
            zip.delete("AndroidManifest.xml");
            zip.delete("classes.dex");
            zip.delete("classes2.dex");
            zip.delete("lib/arm64-v8a/libv240fix.so");

            FullFileSource mf = new FullFileSource(patchedManifest.toPath(), "AndroidManifest.xml", Deflater.NO_COMPRESSION);
            mf.align(4);
            zip.add(mf);
            FullFileSource mainDex = new FullFileSource(patchedDex.toPath(), "classes.dex", Deflater.NO_COMPRESSION);
            mainDex.align(4);
            zip.add(mainDex);
            FullFileSource secondDex = new FullFileSource(runtimeDex.toPath(), "classes2.dex", Deflater.NO_COMPRESSION);
            secondDex.align(4);
            zip.add(secondDex);
            FullFileSource so = new FullFileSource(nativeLib.toPath(), "lib/arm64-v8a/libv240fix.so", Deflater.NO_COMPRESSION);
            so.align(16 * 1024);
            zip.add(so);
        }

        try (ZipFile zip = new ZipFile(output)) {
            requireEntry(zip, "AndroidManifest.xml");
            requireEntry(zip, "classes.dex");
            requireEntry(zip, "classes2.dex");
            requireEntry(zip, "lib/arm64-v8a/libil2cpp.so");
            requireEntry(zip, "lib/arm64-v8a/libv240fix.so");
        }
        assertNativePreserved(output, nativeBefore);
        if (!containsBootstrapInvoke(patchedDex)) throw new IllegalStateException("bootstrap invoke missing after patch");
        AndroidManifestBlock verifyManifest = AndroidManifestBlock.load(patchedManifest);
        if (!packageName.equals(verifyManifest.getPackageName())) throw new IllegalStateException("package changed");
        if (verifyManifest.getActivity(PICKER, false) == null && verifyManifest.getActivity(PICKER, true) == null) {
            throw new IllegalStateException("picker activity missing after patch");
        }
        if (output.length() <= 300L * 1024L * 1024L) throw new IllegalStateException("output unexpectedly small: " + output.length());
        System.out.println("V240_HOST_PATCH_OK");
        System.out.println("package=" + packageName);
        System.out.println("source_sha256=" + sourceSha);
        System.out.println("output_bytes=" + output.length());
        System.out.println("output_sha256=" + sha256(output));
    }

    private static void patchManifest(File source, File output, String expectedPackage) throws Exception {
        AndroidManifestBlock manifest = AndroidManifestBlock.load(source);
        if (!expectedPackage.equals(manifest.getPackageName())) throw new IllegalStateException("unexpected package");
        ResXmlElement picker = manifest.getOrCreateActivity(PICKER, false);
        picker.getOrCreateAndroidAttribute(AndroidManifestBlock.NAME_exported, AndroidManifestBlock.ID_exported).setValueAsBoolean(false);
        manifest.refreshFull();
        byte[] bytes = manifest.getBytes();
        try (FileOutputStream out = new FileOutputStream(output, false)) {
            out.write(bytes);
            out.getFD().sync();
        }
        AndroidManifestBlock verify = AndroidManifestBlock.load(output);
        if (!expectedPackage.equals(verify.getPackageName())) throw new IllegalStateException("manifest package changed");
        if (verify.getActivity(PICKER, false) == null && verify.getActivity(PICKER, true) == null) {
            throw new IllegalStateException("picker activity encode failed");
        }
    }

    private static void patchDex(File input, File output) throws Exception {
        DexFile main = DexFileFactory.loadDexFile(input, Opcodes.forApi(35));
        ClassDef activity = findClass(main, ACTIVITY);
        if (activity == null) throw new IllegalStateException("UnityPlayerActivity class not found");
        Method onCreate = null;
        for (Method method : activity.getMethods()) {
            if (isOnCreate(method)) {
                if (onCreate != null) throw new IllegalStateException("multiple UnityPlayerActivity.onCreate matches");
                onCreate = method;
            }
        }
        if (onCreate == null || onCreate.getImplementation() == null) throw new IllegalStateException("UnityPlayerActivity.onCreate implementation missing");
        Method patched = containsBootstrapInvoke(onCreate.getImplementation()) ? onCreate : injectBootstrap(onCreate);
        List<Method> activityMethods = new ArrayList<Method>();
        for (Method method : activity.getMethods()) activityMethods.add(method == onCreate ? patched : method);
        ImmutableClassDef patchedActivity = new ImmutableClassDef(activity.getType(), activity.getAccessFlags(), activity.getSuperclass(),
                activity.getInterfaces(), activity.getSourceFile(), activity.getAnnotations(), activity.getFields(), activityMethods);
        List<ClassDef> classes = new ArrayList<ClassDef>();
        for (ClassDef c : main.getClasses()) classes.add(ACTIVITY.equals(c.getType()) ? patchedActivity : c);
        DexFileFactory.writeDexFile(output.getAbsolutePath(), new ImmutableDexFile(main.getOpcodes(), classes));
        requireFile(output, "patched classes.dex");
    }

    private static Method injectBootstrap(Method method) {
        MutableMethodImplementation impl = new MutableMethodImplementation(method.getImplementation());
        ImmutableMethodReference ref = new ImmutableMethodReference(BOOTSTRAP, "init", Collections.<String>emptyList(), "V");
        impl.addInstruction(0, new BuilderInstruction35c(Opcode.INVOKE_STATIC, 0, 0, 0, 0, 0, 0, ref));
        return new ImmutableMethod(method.getDefiningClass(), method.getName(), method.getParameters(), method.getReturnType(),
                method.getAccessFlags(), method.getAnnotations(), method.getHiddenApiRestrictions(), impl);
    }

    private static boolean containsBootstrapInvoke(File dexFile) throws Exception {
        DexFile dex = DexFileFactory.loadDexFile(dexFile, Opcodes.forApi(35));
        ClassDef activity = findClass(dex, ACTIVITY);
        if (activity == null) return false;
        for (Method method : activity.getMethods()) {
            if (isOnCreate(method) && method.getImplementation() != null) return containsBootstrapInvoke(method.getImplementation());
        }
        return false;
    }

    private static boolean containsBootstrapInvoke(MethodImplementation implementation) {
        for (Instruction instruction : implementation.getInstructions()) {
            if (!(instruction instanceof ReferenceInstruction)) continue;
            Object raw = ((ReferenceInstruction) instruction).getReference();
            if (!(raw instanceof MethodReference)) continue;
            MethodReference method = (MethodReference) raw;
            if (BOOTSTRAP.equals(method.getDefiningClass()) && "init".equals(method.getName()) &&
                    method.getParameterTypes().isEmpty() && "V".equals(method.getReturnType())) return true;
        }
        return false;
    }

    private static boolean isOnCreate(Method method) {
        if (!ACTIVITY.equals(method.getDefiningClass()) || !"onCreate".equals(method.getName()) || !"V".equals(method.getReturnType())) return false;
        List<? extends CharSequence> parameters = method.getParameterTypes();
        return parameters.size() == 1 && "Landroid/os/Bundle;".contentEquals(parameters.get(0));
    }

    private static ClassDef findClass(DexFile dex, String type) {
        for (ClassDef c : dex.getClasses()) if (type.equals(c.getType())) return c;
        return null;
    }

    private static void deleteSignatures(ZipArchive zip) throws Exception {
        List<String> names = new ArrayList<String>(zip.listEntries());
        for (String name : names) {
            String upper = name.toUpperCase(Locale.US);
            if (upper.startsWith("META-INF/") && (upper.endsWith(".RSA") || upper.endsWith(".DSA") || upper.endsWith(".EC") ||
                    upper.endsWith(".SF") || upper.endsWith("MANIFEST.MF"))) zip.delete(name);
        }
    }

    private static void extract(File apk, String name, File output) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            ZipEntry entry = zip.getEntry(name);
            if (entry == null) throw new IllegalStateException("missing APK entry: " + name);
            try (InputStream in = new BufferedInputStream(zip.getInputStream(entry));
                 OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = in.read(buffer)) >= 0) if (n != 0) out.write(buffer, 0, n);
            }
        }
    }

    private static Map<String, Fingerprint> snapshotNative(File apk) throws Exception {
        Map<String, Fingerprint> result = new HashMap<String, Fingerprint>();
        try (ZipFile zip = new ZipFile(apk)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry e = entries.nextElement();
                if (!e.isDirectory() && e.getName().startsWith("lib/") && e.getName().endsWith(".so")) {
                    result.put(e.getName(), new Fingerprint(e.getSize(), e.getCrc()));
                }
            }
        }
        if (result.isEmpty()) throw new IllegalStateException("source APK has no native libraries");
        return result;
    }

    private static void assertNativePreserved(File apk, Map<String, Fingerprint> before) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            for (Map.Entry<String, Fingerprint> item : before.entrySet()) {
                ZipEntry current = zip.getEntry(item.getKey());
                if (current == null) throw new IllegalStateException("original native library removed: " + item.getKey());
                if (current.getSize() != item.getValue().size || current.getCrc() != item.getValue().crc) {
                    throw new IllegalStateException("original native library changed: " + item.getKey());
                }
            }
        }
    }

    private static void requireEntry(ZipFile zip, String name) {
        ZipEntry e = zip.getEntry(name);
        if (e == null || e.getSize() == 0L) throw new IllegalStateException("missing/empty APK entry: " + name);
    }

    private static void requireFile(File file, String label) {
        if (!file.isFile() || file.length() == 0L) throw new IllegalStateException(label + " missing: " + file);
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) if (n != 0) digest.update(buffer, 0, n);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        if (!file.delete() && file.exists()) throw new IllegalStateException("could not delete " + file);
    }

    private static final class Fingerprint {
        final long size;
        final long crc;
        Fingerprint(long size, long crc) { this.size = size; this.crc = crc; }
    }

    private V240HostPatchCli() {}
}
