package dev.hoonex.adofai.exporter;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public final class MainActivity extends Activity {
    private static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";
    private static final int REQUEST_SAVE = 1001;

    private static final String[][] TARGET_ENTRIES = new String[][] {
            {"assets/bin/Data/Managed/Metadata/global-metadata.dat", "runtime/global-metadata.dat"},
            {"lib/arm64-v8a/libil2cpp.so", "runtime/lib/arm64-v8a/libil2cpp.so"},
            {"lib/arm64-v8a/libunity.so", "runtime/lib/arm64-v8a/libunity.so"},
            {"assets/bin/Data/globalgamemanagers", "runtime/globalgamemanagers"},
            {"assets/bin/Data/boot.config", "runtime/boot.config"},
            {"assets/bin/Data/ScriptingAssemblies.json", "runtime/ScriptingAssemblies.json"},
            {"assets/bin/Data/RuntimeInitializeOnLoads.json", "runtime/RuntimeInitializeOnLoads.json"}
    };

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView statusView;
    private ProgressBar progressBar;
    private Button exportButton;
    private File pendingExport;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int padding = dp(24);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("ADOFAI Runtime Exporter");
        title.setTextSize(24f);
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        TextView description = new TextView(this);
        description.setText("설치된 ADOFAI에서 패치 분석에 필요한 IL2CPP/Unity 런타임 파일만 추출합니다. 게임 전체 데이터는 복사하지 않습니다.");
        description.setTextSize(15f);
        LinearLayout.LayoutParams descriptionParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        descriptionParams.topMargin = dp(16);
        root.addView(description, descriptionParams);

        exportButton = new Button(this);
        exportButton.setText("ADOFAI 런타임 추출");
        exportButton.setAllCaps(false);
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        buttonParams.topMargin = dp(24);
        root.addView(exportButton, buttonParams);

        progressBar = new ProgressBar(this);
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        progressParams.topMargin = dp(18);
        root.addView(progressBar, progressParams);

        statusView = new TextView(this);
        statusView.setText("준비됨");
        statusView.setTextSize(14f);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.topMargin = dp(16);
        root.addView(statusView, statusParams);

        setContentView(root);
        exportButton.setOnClickListener(v -> beginExport());
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void beginExport() {
        exportButton.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("ADOFAI 설치본 확인 중…");

        executor.execute(() -> {
            try {
                ExportResult result = buildExport();
                pendingExport = result.file;
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    statusView.setText(String.format(
                            Locale.US,
                            "추출 완료: %d개 파일, %.1f MB\n저장 위치를 선택하세요.",
                            result.fileCount,
                            result.file.length() / 1024.0 / 1024.0));
                    launchSavePicker(result.suggestedName);
                });
            } catch (PackageManager.NameNotFoundException e) {
                showError("ADOFAI가 설치되어 있지 않거나 패키지를 확인할 수 없습니다.");
            } catch (Exception e) {
                showError("추출 실패: " + e.getClass().getSimpleName() + ": " + safeMessage(e));
            }
        });
    }

    private ExportResult buildExport() throws Exception {
        PackageManager pm = getPackageManager();
        PackageInfo packageInfo;
        if (Build.VERSION.SDK_INT >= 28) {
            packageInfo = pm.getPackageInfo(TARGET_PACKAGE, PackageManager.GET_SIGNING_CERTIFICATES);
        } else {
            packageInfo = pm.getPackageInfo(TARGET_PACKAGE, PackageManager.GET_SIGNATURES);
        }
        ApplicationInfo appInfo = pm.getApplicationInfo(TARGET_PACKAGE, 0);

        List<String> apkPaths = new ArrayList<>();
        apkPaths.add(appInfo.sourceDir);
        if (appInfo.splitSourceDirs != null) {
            apkPaths.addAll(Arrays.asList(appInfo.splitSourceDirs));
        }

        long versionCode = Build.VERSION.SDK_INT >= 28
                ? packageInfo.getLongVersionCode()
                : packageInfo.versionCode;
        String versionName = packageInfo.versionName == null ? "unknown" : packageInfo.versionName;
        String safeVersion = versionName.replaceAll("[^A-Za-z0-9._-]", "_");
        String suggestedName = "adofai-runtime-" + safeVersion + "-" + versionCode + ".zip";
        File outFile = new File(getCacheDir(), suggestedName);
        if (outFile.exists() && !outFile.delete()) {
            throw new IllegalStateException("기존 임시 ZIP을 지울 수 없습니다.");
        }

        List<ExtractedFile> extractedFiles = new ArrayList<>();
        JSONArray apkArray = new JSONArray();
        Set<String> writtenOutputs = new HashSet<>();

        try (ZipOutputStream out = new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(outFile)))) {
            for (String apkPath : apkPaths) {
                File apkFile = new File(apkPath);
                JSONObject apkInfo = new JSONObject();
                apkInfo.put("name", apkFile.getName());
                apkInfo.put("path_leaf", apkFile.getName());
                apkInfo.put("size_bytes", apkFile.length());
                apkArray.put(apkInfo);

                try (ZipFile apk = new ZipFile(apkFile)) {
                    for (String[] target : TARGET_ENTRIES) {
                        if (writtenOutputs.contains(target[1])) {
                            continue;
                        }
                        ZipEntry entry = apk.getEntry(target[0]);
                        if (entry != null && !entry.isDirectory()) {
                            extractedFiles.add(copyEntry(apk, entry, out, target[1], apkFile.getName()));
                            writtenOutputs.add(target[1]);
                        }
                    }

                    if (apkPath.equals(appInfo.sourceDir) && !writtenOutputs.contains("package/AndroidManifest.xml")) {
                        ZipEntry manifestEntry = apk.getEntry("AndroidManifest.xml");
                        if (manifestEntry != null && !manifestEntry.isDirectory()) {
                            extractedFiles.add(copyEntry(
                                    apk,
                                    manifestEntry,
                                    out,
                                    "package/AndroidManifest.xml",
                                    apkFile.getName()));
                            writtenOutputs.add("package/AndroidManifest.xml");
                        }
                    }
                }
            }

            JSONObject report = buildReport(packageInfo, versionCode, versionName, apkArray, extractedFiles);
            ZipEntry reportEntry = new ZipEntry("report.json");
            out.putNextEntry(reportEntry);
            byte[] reportBytes = report.toString(2).getBytes(StandardCharsets.UTF_8);
            out.write(reportBytes);
            out.closeEntry();
        }

        if (!writtenOutputs.contains("runtime/global-metadata.dat")) {
            outFile.delete();
            throw new IllegalStateException("global-metadata.dat를 찾지 못했습니다. 설치 구조가 예상과 다릅니다.");
        }
        if (!writtenOutputs.contains("runtime/lib/arm64-v8a/libil2cpp.so")) {
            outFile.delete();
            throw new IllegalStateException("arm64 libil2cpp.so를 찾지 못했습니다. 설치 구조 또는 ABI가 예상과 다릅니다.");
        }

        return new ExportResult(outFile, suggestedName, extractedFiles.size());
    }

    private JSONObject buildReport(
            PackageInfo packageInfo,
            long versionCode,
            String versionName,
            JSONArray apkArray,
            List<ExtractedFile> extractedFiles) throws Exception {
        JSONObject report = new JSONObject();
        report.put("format", "adofai-runtime-export-v1");
        report.put("package", TARGET_PACKAGE);
        report.put("version_name", versionName);
        report.put("version_code", versionCode);
        report.put("installer_package", getPackageManager().getInstallerPackageName(TARGET_PACKAGE));
        report.put("device_manufacturer", Build.MANUFACTURER);
        report.put("device_model", Build.MODEL);
        report.put("android_sdk", Build.VERSION.SDK_INT);
        report.put("supported_abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)));
        report.put("signer_sha256", signerDigests(packageInfo));
        report.put("installed_apks", apkArray);

        JSONArray files = new JSONArray();
        for (ExtractedFile file : extractedFiles) {
            JSONObject obj = new JSONObject();
            obj.put("output", file.outputName);
            obj.put("source_apk", file.sourceApk);
            obj.put("source_entry", file.sourceEntry);
            obj.put("size_bytes", file.sizeBytes);
            obj.put("sha256", file.sha256);
            files.put(obj);
        }
        report.put("extracted_files", files);
        return report;
    }

    private JSONArray signerDigests(PackageInfo packageInfo) throws Exception {
        JSONArray result = new JSONArray();
        Signature[] signatures = null;
        if (Build.VERSION.SDK_INT >= 28 && packageInfo.signingInfo != null) {
            signatures = packageInfo.signingInfo.hasMultipleSigners()
                    ? packageInfo.signingInfo.getApkContentsSigners()
                    : packageInfo.signingInfo.getSigningCertificateHistory();
        } else if (packageInfo.signatures != null) {
            signatures = packageInfo.signatures;
        }
        if (signatures == null) {
            return result;
        }
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (Signature signature : signatures) {
            result.put(hex(digest.digest(signature.toByteArray())));
            digest.reset();
        }
        return result;
    }

    private ExtractedFile copyEntry(
            ZipFile sourceZip,
            ZipEntry sourceEntry,
            ZipOutputStream out,
            String outputName,
            String sourceApk) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long total = 0;
        ZipEntry outputEntry = new ZipEntry(outputName);
        out.putNextEntry(outputEntry);
        try (InputStream in = new BufferedInputStream(sourceZip.getInputStream(sourceEntry))) {
            byte[] buffer = new byte[128 * 1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                digest.update(buffer, 0, read);
                total += read;
            }
        }
        out.closeEntry();
        return new ExtractedFile(outputName, sourceApk, sourceEntry.getName(), total, hex(digest.digest()));
    }

    private void launchSavePicker(String suggestedName) {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/zip");
        intent.putExtra(Intent.EXTRA_TITLE, suggestedName);
        startActivityForResult(intent, REQUEST_SAVE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_SAVE) {
            return;
        }
        if (resultCode != RESULT_OK || data == null || data.getData() == null || pendingExport == null) {
            statusView.setText("저장이 취소되었습니다. 다시 추출하면 저장 위치를 다시 선택할 수 있습니다.");
            exportButton.setEnabled(true);
            return;
        }

        Uri target = data.getData();
        File source = pendingExport;
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("선택한 위치에 ZIP 저장 중…");
        executor.execute(() -> {
            try (InputStream in = new BufferedInputStream(new FileInputStream(source));
                 OutputStream out = new BufferedOutputStream(getContentResolver().openOutputStream(target, "w"))) {
                if (out == null) {
                    throw new IllegalStateException("저장 스트림을 열 수 없습니다.");
                }
                byte[] buffer = new byte[128 * 1024];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    out.write(buffer, 0, read);
                }
                out.flush();
                source.delete();
                pendingExport = null;
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    statusView.setText("저장 완료. 생성된 ZIP만 ChatGPT에 올리면 됩니다.");
                    exportButton.setEnabled(true);
                });
            } catch (Exception e) {
                showError("저장 실패: " + e.getClass().getSimpleName() + ": " + safeMessage(e));
            }
        });
    }

    private void showError(String message) {
        runOnUiThread(() -> {
            progressBar.setVisibility(View.GONE);
            exportButton.setEnabled(true);
            statusView.setText(message);
        });
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? "원인 메시지 없음" : message;
    }

    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            out.append(String.format(Locale.US, "%02x", value & 0xff));
        }
        return out.toString();
    }

    private static final class ExportResult {
        final File file;
        final String suggestedName;
        final int fileCount;

        ExportResult(File file, String suggestedName, int fileCount) {
            this.file = file;
            this.suggestedName = suggestedName;
            this.fileCount = fileCount;
        }
    }

    private static final class ExtractedFile {
        final String outputName;
        final String sourceApk;
        final String sourceEntry;
        final long sizeBytes;
        final String sha256;

        ExtractedFile(String outputName, String sourceApk, String sourceEntry, long sizeBytes, String sha256) {
            this.outputName = outputName;
            this.sourceApk = sourceApk;
            this.sourceEntry = sourceEntry;
            this.sizeBytes = sizeBytes;
            this.sha256 = sha256;
        }
    }
}
