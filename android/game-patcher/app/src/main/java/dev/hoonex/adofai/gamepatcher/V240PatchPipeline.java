package dev.hoonex.adofai.gamepatcher;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class V240PatchPipeline {
    interface Listener {
        void onProgress(String message);
    }

    static final String EXPECTED_VERSION = "2.4.0";
    static final String OUTPUT_NAME = "ADOFAI-2.4.0-Custom-Bugfix.apk";
    private static final String OCTOBER = "lib/arm64-v8a/libOctober.so";
    private static final String FILE_SELECTOR = DexOverlayPatcher.FILE_SELECTOR;
    private static final String CUSTOM_FILE_CHOOSER = DexOverlayPatcher.CUSTOM_FILE_CHOOSER;

    static final class Result {
        final Uri outputUri;
        final String packageName;
        final String sourceOctoberSha256;
        final String signerSha256;
        final long outputBytes;

        Result(Uri outputUri, String packageName, String sourceOctoberSha256,
               String signerSha256, long outputBytes) {
            this.outputUri = outputUri;
            this.packageName = packageName;
            this.sourceOctoberSha256 = sourceOctoberSha256;
            this.signerSha256 = signerSha256;
            this.outputBytes = outputBytes;
        }
    }

    static Result patch(Context context, Uri sourceUri, Listener listener) throws Exception {
        if (sourceUri == null) throw new IllegalArgumentException("원본 APK가 선택되지 않았습니다.");

        File work = new File(context.getCacheDir(), "v240-single-apk-patcher");
        deleteTree(work);
        if (!work.mkdirs()) throw new IllegalStateException("작업 폴더 생성 실패");

        File source = new File(work, "source.apk");
        File unsigned = new File(work, "patched-unsigned.apk");
        File signed = new File(work, OUTPUT_NAME);
        try {
            progress(listener, "원본 APK 복사 중…");
            copyUri(context.getContentResolver(), sourceUri, source);
            if (source.length() < 10L * 1024L * 1024L) {
                throw new IllegalStateException("선택한 파일이 정상적인 ADOFAI APK보다 너무 작습니다.");
            }

            progress(listener, "2.4.0 Custom 구조 검증 중…");
            PackageInfo info = context.getPackageManager().getPackageArchiveInfo(source.getAbsolutePath(), 0);
            if (info == null) throw new IllegalStateException("Android APK 메타데이터를 읽을 수 없습니다.");
            if (!EXPECTED_VERSION.equals(info.versionName)) {
                throw new IllegalStateException("2.4.0 APK만 지원합니다. 선택된 versionName=" + info.versionName);
            }
            String packageName = info.packageName;
            if (packageName == null || packageName.trim().isEmpty()) {
                throw new IllegalStateException("APK package name을 읽을 수 없습니다.");
            }

            String sourceOctoberSha;
            try (ZipFile zip = new ZipFile(source)) {
                requireEntry(zip, "AndroidManifest.xml");
                requireEntry(zip, "classes.dex");
                ZipEntry october = requireEntry(zip, OCTOBER);
                sourceOctoberSha = sha256(zip.getInputStream(october));
            }

            progress(listener, "기존 FileSelector 위치 탐색 중…");
            String selectorDexEntry = findLegacyFileSelectorDex(source, work);

            progress(listener, "2.4 버그픽스 payload 준비 중…");
            PayloadAssets.Payload payload = PayloadAssets.stage(context, work);

            progress(listener, selectorDexEntry + " / libOctober.so / manifest 수정 중…");
            ApkMutator.mutateV240Single(
                source, unsigned, payload.classes2Dex, payload.libOctober, work, packageName,
                selectorDexEntry
            );
            ApkMutator.assertEntry(unsigned, selectorDexEntry);
            ApkMutator.assertEntry(unsigned, OCTOBER);
            ApkMutator.assertEntry(unsigned, "AndroidManifest.xml");
            assertPatchedFilePicker(unsigned, selectorDexEntry, work);

            progress(listener, "수정 APK 재서명 및 검증 중…");
            SigningIdentity identity = SigningIdentity.loadOrCreate();
            String signer = SplitSigner.signAndVerify(unsigned, signed, identity);
            if (!identity.sha256.equals(signer)) {
                throw new IllegalStateException("signer 검증 불일치");
            }

            progress(listener, "Downloads에 수정본 저장 중…");
            Uri output = publishToDownloads(context, signed);
            progress(listener, "완료: " + OUTPUT_NAME);
            return new Result(output, packageName, sourceOctoberSha, signer, signed.length());
        } finally {
            deleteTree(work);
        }
    }

    /**
     * Old 2.4 Custom builds are not guaranteed to be multidex. In particular, the user's
     * working build has no classes2.dex. Search every classes*.dex and patch the dex that
     * actually owns FileSelector instead of assuming a fixed secondary-dex layout.
     */
    private static String findLegacyFileSelectorDex(File apk, File work) throws Exception {
        List<String> dexEntries = new ArrayList<String>();
        try (ZipFile zip = new ZipFile(apk)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                if (!entry.isDirectory() && isDexEntryName(entry.getName()) && entry.getSize() > 0L) {
                    dexEntries.add(entry.getName());
                }
            }
        }
        if (dexEntries.isEmpty()) throw new IllegalStateException("APK에 classes.dex가 없습니다.");

        Collections.sort(dexEntries, new Comparator<String>() {
            @Override
            public int compare(String left, String right) {
                return Integer.compare(dexIndex(left), dexIndex(right));
            }
        });

        for (String entryName : dexEntries) {
            File dexFile = new File(work, "scan-" + dexIndex(entryName) + ".dex");
            extractEntry(apk, entryName, dexFile);
            if (dexContainsClass(dexFile, FILE_SELECTOR)) return entryName;
        }
        throw new IllegalStateException(
            "이 APK에는 기존 2.4 Custom FileSelector가 없습니다. 다른 2.4.0 APK는 수정하지 않습니다."
        );
    }

    private static void assertPatchedFilePicker(File apk, String dexEntry, File work) throws Exception {
        File dexFile = new File(work, "verify-file-picker.dex");
        extractEntry(apk, dexEntry, dexFile);
        DexFile dex = DexFileFactory.loadDexFile(dexFile, Opcodes.forApi(35));
        boolean selector = false;
        boolean chooser = false;
        for (ClassDef cls : dex.getClasses()) {
            if (FILE_SELECTOR.equals(cls.getType())) selector = true;
            if (CUSTOM_FILE_CHOOSER.equals(cls.getType())) chooser = true;
        }
        if (!selector || !chooser) {
            throw new IllegalStateException("수정된 dex에서 FileSelector/CustomFileChooser 검증 실패");
        }
    }

    private static boolean dexContainsClass(File dexFile, String type) throws Exception {
        DexFile dex = DexFileFactory.loadDexFile(dexFile, Opcodes.forApi(35));
        for (ClassDef cls : dex.getClasses()) {
            if (type.equals(cls.getType())) return true;
        }
        return false;
    }

    private static boolean isDexEntryName(String name) {
        if ("classes.dex".equals(name)) return true;
        if (!name.startsWith("classes") || !name.endsWith(".dex")) return false;
        String middle = name.substring("classes".length(), name.length() - ".dex".length());
        if (middle.isEmpty()) return false;
        for (int i = 0; i < middle.length(); i++) {
            if (!Character.isDigit(middle.charAt(i))) return false;
        }
        return true;
    }

    private static int dexIndex(String name) {
        if ("classes.dex".equals(name)) return 1;
        String middle = name.substring("classes".length(), name.length() - ".dex".length());
        try {
            return Integer.parseInt(middle);
        } catch (NumberFormatException ignored) {
            return Integer.MAX_VALUE;
        }
    }

    private static void extractEntry(File apk, String name, File output) throws Exception {
        try (ZipFile zip = new ZipFile(apk)) {
            ZipEntry entry = zip.getEntry(name);
            if (entry == null || entry.getSize() <= 0L) {
                throw new IllegalStateException("APK entry not found: " + name);
            }
            try (InputStream in = new BufferedInputStream(zip.getInputStream(entry));
                 OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
                copy(in, out);
            }
        }
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
            throw new IllegalStateException("2.4 Custom APK 필수 항목 없음: " + name);
        }
        return entry;
    }

    private static void copyUri(ContentResolver resolver, Uri uri, File output) throws Exception {
        try (InputStream in = new BufferedInputStream(resolver.openInputStream(uri));
             OutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
            if (in == null) throw new IllegalStateException("선택한 APK를 열 수 없습니다.");
            copy(in, out);
        }
    }

    private static String sha256(InputStream raw) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(raw)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
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
