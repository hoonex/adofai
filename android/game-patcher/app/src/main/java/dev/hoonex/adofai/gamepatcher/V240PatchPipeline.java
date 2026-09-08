package dev.hoonex.adofai.gamepatcher;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import com.reandroid.arsc.chunk.xml.AndroidManifestBlock;

import org.jf.dexlib2.DexFileFactory;
import org.jf.dexlib2.Opcodes;
import org.jf.dexlib2.iface.ClassDef;
import org.jf.dexlib2.iface.DexFile;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class V240PatchPipeline {
    interface Listener {
        void onProgress(String message);
    }

    static final String EXPECTED_VERSION = "2.4.0";
    static final long EXPECTED_SOURCE_BYTES = 370_092_054L;
    static final String EXPECTED_SOURCE_SHA256 =
            "630f519ae1ab3391aad95da90ebc296f4f0f8ae4ea41024ace7349d93926ef30";
    static final String OUTPUT_NAME = "ADOFAI-2.4.0-Custom-Bugfix.apk";

    private static final String LIBIL2CPP = "lib/arm64-v8a/libil2cpp.so";
    private static final String FIX_LIBRARY = "lib/arm64-v8a/libv240fix.so";
    private static final String[] REQUIRED_RUNTIME_CLASSES = {
            "Lcom/unity3d/player/V240Bootstrap;",
            "Lcom/unity3d/player/FileSelector;",
            "Lcom/unity3d/player/V240AndroidBridge;",
            "Lcom/unity3d/player/V240PickerActivity;",
            "Lcom/unity3d/player/V240SettingsOverlay;"
    };

    static final class Result {
        final Uri outputUri;
        final String packageName;
        final String pickerPatchMode;
        final String signerSha256;
        final long outputBytes;

        Result(Uri outputUri, String packageName, String pickerPatchMode,
               String signerSha256, long outputBytes) {
            this.outputUri = outputUri;
            this.packageName = packageName;
            this.pickerPatchMode = pickerPatchMode;
            this.signerSha256 = signerSha256;
            this.outputBytes = outputBytes;
        }
    }

    static Result patch(Context context, Uri sourceUri, Listener listener) throws Exception {
        if (sourceUri == null) throw new IllegalArgumentException("원본 APK가 선택되지 않았습니다.");

        File work = new File(context.getCacheDir(), "v240-exact-apk-patcher");
        deleteTree(work);
        if (!work.mkdirs()) throw new IllegalStateException("작업 폴더 생성 실패");

        File source = new File(work, "source.apk");
        File unsigned = new File(work, "patched-unsigned.apk");
        File signed = new File(work, OUTPUT_NAME);
        try {
            progress(listener, "원본 V2.4.0 Custom.apk 복사 중…");
            copyUri(context.getContentResolver(), sourceUri, source);

            progress(listener, "원본 지문 확인 중…");
            assertExactSource(source);

            PackageInfo info = context.getPackageManager().getPackageArchiveInfo(source.getAbsolutePath(), 0);
            if (info == null) throw new IllegalStateException("Android APK 메타데이터를 읽을 수 없습니다.");
            if (!EXPECTED_VERSION.equals(info.versionName)) {
                throw new IllegalStateException("2.4.0 APK만 지원합니다. 선택된 versionName=" + info.versionName);
            }
            String packageName = info.packageName;
            if (packageName == null || packageName.trim().isEmpty()) {
                throw new IllegalStateException("APK package name을 읽을 수 없습니다.");
            }

            Map<String, EntryFingerprint> originalNative = snapshotNativeLibraries(source);
            try (ZipFile zip = new ZipFile(source)) {
                requireEntry(zip, "AndroidManifest.xml");
                requireEntry(zip, "classes.dex");
                requireEntry(zip, LIBIL2CPP);
                if (zip.getEntry("classes2.dex") != null) {
                    throw new IllegalStateException("authoritative 2.4 source unexpectedly contains classes2.dex");
                }
                if (zip.getEntry("lib/arm64-v8a/libOctober.so") != null) {
                    throw new IllegalStateException("authoritative 2.4 source unexpectedly contains libOctober.so");
                }
            }

            progress(listener, "2.4 전용 Java/native 런타임 준비 중…");
            PayloadAssets.V240Runtime runtime = PayloadAssets.stageV240FixedRuntime(context, work);
            assertRuntimeDex(runtime.runtimeDex);

            progress(listener, "SFB + Android SAF + 모바일 에디터 수정 적용 중…");
            ApkMutator.mutateV240Single(
                    source,
                    unsigned,
                    runtime.runtimeDex,
                    runtime.nativeLibrary,
                    work,
                    packageName
            );

            progress(listener, "수정본 구조 검증 중…");
            assertPatchedStructure(unsigned, work, packageName, originalNative);

            progress(listener, "수정 APK 재서명 및 검증 중…");
            SigningIdentity identity = SigningIdentity.loadOrCreate();
            String signer = SplitSigner.signAndVerify(unsigned, signed, identity);
            if (!identity.sha256.equals(signer)) {
                throw new IllegalStateException("signer 검증 불일치");
            }
            assertSignedStructure(signed, packageName);

            progress(listener, "Downloads에 수정본 저장 중…");
            Uri output = publishToDownloads(context, signed);
            progress(listener, "완료: " + OUTPUT_NAME);
            return new Result(
                    output,
                    packageName,
                    "SFB sync/async native hook + Android SAF picker + mobile editor runtime",
                    signer,
                    signed.length()
            );
        } finally {
            deleteTree(work);
        }
    }

    private static void assertExactSource(File source) throws Exception {
        if (source.length() != EXPECTED_SOURCE_BYTES) {
            throw new IllegalStateException(
                    "이 패처는 업로드된 정확한 V2.4.0 Custom.apk만 수정합니다. size=" + source.length()
            );
        }
        String actual = sha256(source);
        if (!EXPECTED_SOURCE_SHA256.equals(actual)) {
            throw new IllegalStateException(
                    "원본 APK SHA-256 불일치. expected=" + EXPECTED_SOURCE_SHA256 + " actual=" + actual
            );
        }
    }

    private static void assertPatchedStructure(
            File apk,
            File work,
            String packageName,
            Map<String, EntryFingerprint> originalNative
    ) throws Exception {
        ApkMutator.assertEntry(apk, "AndroidManifest.xml");
        ApkMutator.assertEntry(apk, "classes.dex");
        ApkMutator.assertEntry(apk, "classes2.dex");
        ApkMutator.assertEntry(apk, FIX_LIBRARY);
        ApkMutator.assertEntry(apk, LIBIL2CPP);

        File patchedPrimaryDex = new File(work, "verify-v240-primary.dex");
        ApkMutator.extractEntry(apk, "classes.dex", patchedPrimaryDex);
        if (!V240DexBootstrapPatcher.containsBootstrapInvoke(patchedPrimaryDex)) {
            throw new IllegalStateException("UnityPlayerActivity bootstrap injection verification failed");
        }

        File runtimeDex = new File(work, "verify-v240-runtime.dex");
        ApkMutator.extractEntry(apk, "classes2.dex", runtimeDex);
        assertRuntimeDex(runtimeDex);

        File manifestFile = new File(work, "verify-v240-manifest.xml");
        ApkMutator.extractEntry(apk, "AndroidManifest.xml", manifestFile);
        AndroidManifestBlock manifest = AndroidManifestBlock.load(manifestFile);
        if (!packageName.equals(manifest.getPackageName())) {
            throw new IllegalStateException("package name changed during patch");
        }
        ManifestStoragePatcher.assertV240Picker(manifest);

        assertOriginalNativeLibrariesPreserved(apk, originalNative);
    }

    private static void assertSignedStructure(File signed, String packageName) throws Exception {
        try (ZipFile zip = new ZipFile(signed)) {
            requireEntry(zip, "AndroidManifest.xml");
            requireEntry(zip, "classes.dex");
            requireEntry(zip, "classes2.dex");
            requireEntry(zip, FIX_LIBRARY);
            requireEntry(zip, LIBIL2CPP);
        }
        if (signed.length() < 300L * 1024L * 1024L) {
            throw new IllegalStateException("signed output is unexpectedly small: " + signed.length());
        }
        if (packageName == null || packageName.trim().isEmpty()) {
            throw new IllegalStateException("package name lost during signing");
        }
    }

    private static void assertRuntimeDex(File dexFile) throws Exception {
        DexFile dex = DexFileFactory.loadDexFile(dexFile, Opcodes.forApi(35));
        Map<String, Boolean> required = new HashMap<String, Boolean>();
        for (String name : REQUIRED_RUNTIME_CLASSES) required.put(name, Boolean.FALSE);
        for (ClassDef cls : dex.getClasses()) {
            if (required.containsKey(cls.getType())) required.put(cls.getType(), Boolean.TRUE);
        }
        for (Map.Entry<String, Boolean> entry : required.entrySet()) {
            if (!entry.getValue()) {
                throw new IllegalStateException("fixed runtime DEX missing class: " + entry.getKey());
            }
        }
    }

    private static Map<String, EntryFingerprint> snapshotNativeLibraries(File apk) throws Exception {
        Map<String, EntryFingerprint> result = new HashMap<String, EntryFingerprint>();
        try (ZipFile zip = new ZipFile(apk)) {
            java.util.Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String name = entry.getName();
                if (!entry.isDirectory() && name.startsWith("lib/") && name.endsWith(".so")) {
                    result.put(name, new EntryFingerprint(entry.getSize(), entry.getCrc()));
                }
            }
        }
        if (result.isEmpty()) throw new IllegalStateException("source APK has no native libraries");
        return result;
    }

    private static void assertOriginalNativeLibrariesPreserved(
            File patched,
            Map<String, EntryFingerprint> original
    ) throws Exception {
        try (ZipFile zip = new ZipFile(patched)) {
            for (Map.Entry<String, EntryFingerprint> item : original.entrySet()) {
                ZipEntry current = zip.getEntry(item.getKey());
                if (current == null) {
                    throw new IllegalStateException("original native library removed: " + item.getKey());
                }
                EntryFingerprint expected = item.getValue();
                if (current.getSize() != expected.size || current.getCrc() != expected.crc) {
                    throw new IllegalStateException("original native library changed: " + item.getKey());
                }
            }
        }
    }

    private static final class EntryFingerprint {
        final long size;
        final long crc;

        EntryFingerprint(long size, long crc) {
            this.size = size;
            this.crc = crc;
        }
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder result = new StringBuilder(64);
        for (byte b : digest.digest()) result.append(String.format(Locale.US, "%02x", b & 0xff));
        return result.toString();
    }

    private static Uri publishToDownloads(Context context, File source) throws Exception {
        if (Build.VERSION.SDK_INT >= 29) {
            ContentResolver resolver = context.getContentResolver();
            ContentValues values = new ContentValues();
            values.put(MediaStore.Downloads.DISPLAY_NAME, OUTPUT_NAME);
            values.put(MediaStore.Downloads.MIME_TYPE, "application/vnd.android.package-archive");
            values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/ADOFAI");
            values.put(MediaStore.Downloads.IS_PENDING, 1);
            Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
            if (uri == null) throw new IllegalStateException("Downloads 항목 생성 실패");
            boolean ok = false;
            try {
                try (InputStream in = new BufferedInputStream(new FileInputStream(source));
                     OutputStream out = new BufferedOutputStream(resolver.openOutputStream(uri, "w"))) {
                    if (out == null) throw new IllegalStateException("Downloads 출력 스트림 생성 실패");
                    copy(in, out);
                }
                ContentValues done = new ContentValues();
                done.put(MediaStore.Downloads.IS_PENDING, 0);
                resolver.update(uri, done, null, null);
                ok = true;
                return uri;
            } finally {
                if (!ok) resolver.delete(uri, null, null);
            }
        }

        File root = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (root == null) root = context.getFilesDir();
        File out = new File(root, OUTPUT_NAME);
        try (InputStream in = new BufferedInputStream(new FileInputStream(source));
             OutputStream output = new BufferedOutputStream(new FileOutputStream(out))) {
            copy(in, output);
        }
        return Uri.fromFile(out);
    }

    private static ZipEntry requireEntry(ZipFile zip, String name) {
        ZipEntry entry = zip.getEntry(name);
        if (entry == null || entry.getSize() == 0L) {
            throw new IllegalStateException("2.4 APK 필수 항목 없음: " + name);
        }
        return entry;
    }

    private static void copyUri(ContentResolver resolver, Uri uri, File output) throws Exception {
        try (InputStream raw = resolver.openInputStream(uri)) {
            if (raw == null) throw new IllegalStateException("선택한 APK를 열 수 없습니다.");
            try (InputStream in = new BufferedInputStream(raw);
                 OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
                copy(in, out);
            }
        }
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buffer = new byte[1024 * 1024];
        int count;
        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        out.flush();
    }

    private static void deleteTree(File file) throws Exception {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteTree(child);
        }
        if (!file.delete()) throw new IllegalStateException("작업 경로 삭제 실패: " + file);
    }

    private static void progress(Listener listener, String message) {
        if (listener != null) listener.onProgress(message);
    }

    private V240PatchPipeline() {}
}
