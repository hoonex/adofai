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
        progress(listener, "3.3.1 설치본 확인 중…");
        InstalledGame game = InstalledGame.inspect(context);

        File externalRoot = context.getExternalFilesDir(null);
        if (externalRoot == null) throw new IllegalStateException("외부 앱 저장소를 사용할 수 없습니다.");
        File outputDir = new File(externalRoot, "prepared-game/apks");
        File workBase = context.getExternalCacheDir();
        if (workBase == null) workBase = context.getCacheDir();
        File workDir = new File(workBase, "adofai-game-patcher-work");
        assertOwned(context, outputDir);
        assertOwned(context, workDir);

        long reserve = 512L * 1024L * 1024L;
        long required = game.totalBytes + game.baseApk.length() + game.arm64Apk.length() + reserve;
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
            File unsignedArm64 = new File(workDir, "arm64-unsigned.apk");

            progress(listener, "base APK에 Editor bootstrap/DEX 주입 중…");
            ApkMutator.mutateBase(game.baseApk, unsignedBase, payload.classes2Dex, workDir);
            ApkMutator.assertEntry(unsignedBase, "classes.dex");
            ApkMutator.assertEntry(unsignedBase, "classes2.dex");

            progress(listener, "arm64 split에 libOctober 주입 중…");
            ApkMutator.mutateArm64(game.arm64Apk, unsignedArm64, payload.libOctober);
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
        if (!report.isFile() || list == null || list.length < 2) return null;
        String raw = readText(report);
        JSONObject json = new JSONObject(raw);
        if (!InstalledGame.PACKAGE_NAME.equals(json.optString("package"))) return null;
        if (!InstalledGame.EXPECTED_VERSION_NAME.equals(json.optString("sourceVersionName"))) return null;
        if (json.optLong("sourceVersionCode", -1L) != InstalledGame.EXPECTED_VERSION_CODE) return null;
        List<File> apks = new ArrayList<File>();
        Collections.addAll(apks, list);
        Collections.sort(apks, Comparator.comparing(File::getName));
        return new PreparedSet(dir, apks, json.optString("signerSha256"));
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
