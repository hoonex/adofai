package dev.hoonex.adofai.gamepatcher;

import android.content.Context;
import android.os.StatFs;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class PatchPipeline {
    interface Listener {
        void onProgress(String message);
    }

    static final class PreparedSet {
        final File directory;
        final List<File> apks;
        final String signerSha256;

        PreparedSet(File directory, List<File> apks, String signerSha256) {
            this.directory = directory;
            this.apks = Collections.unmodifiableList(apks);
            this.signerSha256 = signerSha256;
        }
    }

    static PreparedSet prepare(Context context, Listener listener) throws Exception {
        return prepare(context, listener, null);
    }

    static PreparedSet prepare(Context context, Listener listener, File selectedSource) throws Exception {
        progress(listener, selectedSource == null
                ? "3.3.1 설치본 확인 중…" : "선택한 APK 버전과 구조 검사 중…");
        InstalledGame game = selectedSource == null
                ? InstalledGame.inspect(context)
                : InstalledGame.inspectStandalone(context, selectedSource);

        File externalRoot = context.getExternalFilesDir(null);
        if (externalRoot == null) throw new IllegalStateException("외부 앱 저장소를 사용할 수 없습니다.");
        File outputDir = new File(externalRoot, "prepared-game/apks");
        File workBase = context.getExternalCacheDir();
        if (workBase == null) workBase = context.getCacheDir();
        File workDir = new File(workBase, "adofai-game-patcher-work");
        assertOwned(context, outputDir);
        assertOwned(context, workDir);

        long reserve = 512L * 1024L * 1024L;
        boolean monolithic = game.baseApk.equals(game.arm64Apk);
        // A monolithic APK is copied only once before signing; there is no
        // separate arm64 split output or duplicate PackageInstaller stream.
        long required = game.totalBytes + game.baseApk.length()
                + (monolithic ? 0L : game.arm64Apk.length()) + reserve;
        File spaceProbe = externalRoot;
        StatFs stat = new StatFs(spaceProbe.getAbsolutePath());
        long available = stat.getAvailableBytes();
        if (available < required) {
            throw new IllegalStateException(
                "저장공간이 부족합니다. 필요 약 " + InstalledGame.formatBytes(required) +
                ", 사용 가능 " + InstalledGame.formatBytes(available)
            );
        }

        deleteTree(outputDir.getParentFile());
        deleteTree(workDir);
        if (!outputDir.mkdirs()) throw new IllegalStateException("출력 폴더 생성 실패: " + outputDir);
        if (!workDir.mkdirs()) throw new IllegalStateException("작업 폴더 생성 실패: " + workDir);

        try {
            progress(listener, "편집기 payload 준비 중…");
            PayloadAssets.Payload payload = PayloadAssets.stage(context, workDir);
            SigningIdentity identity = SigningIdentity.loadOrCreate();

            File unsignedBase = new File(workDir, "base-unsigned.apk");
            File unsignedArm64 = monolithic
                    ? unsignedBase : new File(workDir, "arm64-unsigned.apk");

            progress(listener, "base APK에 Editor bootstrap/DEX 주입 중…");
            ApkMutator.mutateBase(game.baseApk, unsignedBase, payload.classes2Dex, workDir);
            ApkMutator.assertEntry(unsignedBase, "classes.dex");
            ApkMutator.assertEntry(unsignedBase, "classes2.dex");

            if (monolithic) {
                progress(listener, "단일 APK에 arm64 libOctober 주입 중…");
                ApkMutator.injectArm64IntoExisting(unsignedBase, payload.libOctober);
            } else {
                progress(listener, "arm64 split에 libOctober 주입 중…");
                ApkMutator.mutateArm64(game.arm64Apk, unsignedArm64, payload.libOctober);
            }
            ApkMutator.assertEntry(unsignedArm64, "lib/arm64-v8a/libOctober.so");

            List<File> outputs = new ArrayList<File>();
            int index = 0;
            for (File source : game.allApks) {
                index++;
                File signingInput;
                if (source.equals(game.baseApk)) signingInput = unsignedBase;
                else if (source.equals(game.arm64Apk)) signingInput = unsignedArm64;
                else signingInput = source;

                String name = source.equals(game.baseApk) ? "base.apk" : source.getName();
                File signed = new File(outputDir, name);
                progress(listener, "split 재서명 " + index + "/" + game.allApks.size() + ": " + name);
                String digest = SplitSigner.signAndVerify(signingInput, signed, identity);
                if (!identity.sha256.equals(digest)) {
                    throw new IllegalStateException("signer drift detected while signing " + name);
                }
                verifyOriginalGameEntries(source, signed);
                outputs.add(signed);
            }

            Collections.sort(outputs, new Comparator<File>() {
                @Override public int compare(File a, File b) {
                    if ("base.apk".equals(a.getName())) return -1;
                    if ("base.apk".equals(b.getName())) return 1;
                    return a.getName().compareTo(b.getName());
                }
            });
            writeReport(new File(outputDir.getParentFile(), "report.json"), game, outputs, identity.sha256);
            progress(listener, "패치/재서명 세트 준비 완료");
            return new PreparedSet(outputDir, outputs, identity.sha256);
        } finally {
            deleteTree(workDir);
        }
    }

    static PreparedSet loadPrepared(Context context) throws Exception {
        File root = context.getExternalFilesDir(null);
        if (root == null) return null;
        File dir = new File(root, "prepared-game/apks");
        File report = new File(root, "prepared-game/report.json");
        File[] list = dir.listFiles((file, name) -> name.endsWith(".apk"));
        // Both monolithic APKs (one stream) and app-bundle split installations
        // (multiple streams) are valid, provided every manifest hash checks out.
        if (!report.isFile() || list == null || list.length < 1) return null;
        String raw = readText(report);
        JSONObject json = new JSONObject(raw);
        if (!InstalledGame.PACKAGE_NAME.equals(json.optString("package"))) return null;
        if (!InstalledGame.EXPECTED_VERSION_NAME.equals(json.optString("sourceVersionName"))) return null;
        if (json.optLong("sourceVersionCode", -1L) != InstalledGame.EXPECTED_VERSION_CODE) return null;
        if (!"adofai-mobile-editor-prepared-v1".equals(json.optString("format"))) {
            throw new IllegalStateException("unknown prepared game report format");
        }
        String signer = json.optString("signerSha256");
        if (!signer.equals(SigningIdentity.loadOrCreate().sha256)) {
            throw new IllegalStateException("prepared game signer differs from device keystore");
        }
        JSONArray manifest = json.optJSONArray("apks");
        if (manifest == null || manifest.length() != list.length) {
            throw new IllegalStateException("prepared game split count does not match report");
        }
        Set<String> expectedNames = new HashSet<String>();
        for (int index = 0; index < manifest.length(); index++) {
            JSONObject entry = manifest.optJSONObject(index);
            if (entry == null) throw new IllegalStateException("invalid prepared game report entry");
            String name = entry.optString("name", "");
            if (!name.endsWith(".apk") || name.indexOf('/') >= 0 || name.indexOf('\\') >= 0 ||
                name.contains("..") || !expectedNames.add(name)) {
                throw new IllegalStateException("invalid or duplicate prepared game split: " + name);
            }
            File apk = new File(dir, name);
            if (!apk.isFile() || !apk.getCanonicalFile().getParentFile().equals(dir.getCanonicalFile()) ||
                apk.length() != entry.optLong("size", -1L) ||
                !sha256(apk).equals(entry.optString("sha256"))) {
                throw new IllegalStateException("prepared game split integrity mismatch: " + name);
            }
        }
        if (!expectedNames.contains("base.apk")) {
            throw new IllegalStateException("prepared game base.apk is missing");
        }
        for (File file : list) {
            if (!expectedNames.contains(file.getName())) {
                throw new IllegalStateException("unreported game split: " + file.getName());
            }
        }
        List<File> apks = new ArrayList<File>();
        Collections.addAll(apks, list);
        Collections.sort(apks, Comparator.comparing(File::getName));
        return new PreparedSet(dir, apks, signer);
    }

    private static void writeReport(File report, InstalledGame game, List<File> apks, String signer) throws Exception {
        JSONObject json = new JSONObject();
        json.put("format", "adofai-mobile-editor-prepared-v1");
        json.put("package", InstalledGame.PACKAGE_NAME);
        json.put("sourceVersionName", game.versionName);
        json.put("sourceVersionCode", game.versionCode);
        json.put("signerSha256", signer);
        JSONArray files = new JSONArray();
        for (File apk : apks) {
            JSONObject item = new JSONObject();
            item.put("name", apk.getName());
            item.put("size", apk.length());
            item.put("sha256", sha256(apk));
            files.put(item);
        }
        json.put("apks", files);
        File parent = report.getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();
        try (FileOutputStream out = new FileOutputStream(report, false)) {
            out.write((json.toString(2) + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
            out.getFD().sync();
        }
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(java.util.Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    /**
     * Preserve official Unity binaries, metadata, and game assets byte-for-byte
     * even when an APK container must be resigned. These critical entries are
     * not ours to replace. Hash only files that actually occur in the source
     * split; the set differs between Android App Bundle configurations.
     */
    private static void verifyOriginalGameEntries(File original, File patched) throws Exception {
        try (ZipFile before = new ZipFile(original);
             ZipFile after = new ZipFile(patched)) {
            for (String name : new String[] {
                "lib/arm64-v8a/libil2cpp.so",
                "lib/arm64-v8a/libunity.so",
                "lib/arm64-v8a/libmain.so",
                "assets/bin/Data/Managed/Metadata/global-metadata.dat"
            }) {
                ZipEntry source = before.getEntry(name);
                if (source == null) continue;
                ZipEntry destination = after.getEntry(name);
                if (destination == null || source.getSize() != destination.getSize() ||
                    !sha256ZipEntry(before, source).equals(sha256ZipEntry(after, destination))) {
                    throw new IllegalStateException("original Unity entry changed: " + name);
                }
            }
        }
    }

    private static String sha256ZipEntry(ZipFile archive, ZipEntry entry) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(archive.getInputStream(entry), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder result = new StringBuilder();
        for (byte value : digest.digest()) {
            result.append(String.format(java.util.Locale.US, "%02x", value & 255));
        }
        return result.toString();
    }

    private static String readText(File file) throws Exception {
        byte[] bytes = new byte[(int) file.length()];
        try (InputStream in = new FileInputStream(file)) {
            int offset = 0;
            while (offset < bytes.length) {
                int count = in.read(bytes, offset, bytes.length - offset);
                if (count < 0) break;
                offset += count;
            }
        }
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void assertOwned(Context context, File path) throws Exception {
        String target = path.getCanonicalPath();
        File externalFiles = context.getExternalFilesDir(null);
        File externalCache = context.getExternalCacheDir();
        String filesRoot = externalFiles == null ? "" : externalFiles.getCanonicalPath() + File.separator;
        String cacheRoot = externalCache == null ? "" : externalCache.getCanonicalPath() + File.separator;
        String internalCache = context.getCacheDir().getCanonicalPath() + File.separator;
        if (!(target.startsWith(filesRoot) || target.startsWith(cacheRoot) || target.startsWith(internalCache))) {
            throw new IllegalStateException("refusing non-owned patch path: " + target);
        }
    }

    private static void deleteTree(File file) throws Exception {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteTree(child);
        }
        if (!file.delete()) throw new IllegalStateException("could not delete work path: " + file);
    }

    private static void progress(Listener listener, String message) {
        if (listener != null) listener.onProgress(message);
    }

    private PatchPipeline() {}
}
