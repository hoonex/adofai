package dev.hoonex.adofai.nativeprobe;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Tiny, read-only probe for the exact ADOFAI 3.3.1 split install.
 *
 * It does not copy game binaries. It records only APK/split metadata and the
 * names/sizes/compression modes of lib/arm64-v8a/*.so entries so the phone-only
 * bootstrap strategy can be chosen from exact evidence instead of assumptions.
 */
public final class MainActivity extends Activity {
    private static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";
    private static final String EXPECTED_VERSION_NAME = "3.3.1";
    private static final long EXPECTED_VERSION_CODE = 300382L;
    private static final String ARM64_PREFIX = "lib/arm64-v8a/";
    private static final int REQUEST_SAVE = 2301;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView statusView;
    private ProgressBar progressBar;
    private Button inspectButton;
    private File pendingReport;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int padding = dp(24);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("ADOFAI 3.3.1 Native Inventory");
        title.setTextSize(23f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "설치된 ADOFAI의 split별 arm64 네이티브 라이브러리 이름/크기만 확인합니다. " +
                "게임 바이너리나 에셋은 복사하지 않습니다.");
        description.setTextSize(15f);
        LinearLayout.LayoutParams descriptionParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        descriptionParams.topMargin = dp(16);
        root.addView(description, descriptionParams);

        inspectButton = new Button(this);
        inspectButton.setText("3.3.1 native inventory 만들기");
        inspectButton.setAllCaps(false);
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        buttonParams.topMargin = dp(24);
        root.addView(inspectButton, buttonParams);

        progressBar = new ProgressBar(this);
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        progressParams.topMargin = dp(18);
        root.addView(progressBar, progressParams);

        statusView = new TextView(this);
        statusView.setText("준비됨 — 결과는 몇 KB짜리 JSON입니다.");
        statusView.setTextSize(14f);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.topMargin = dp(16);
        root.addView(statusView, statusParams);

        setContentView(root);
        inspectButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                beginInspection();
            }
        });
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void beginInspection() {
        inspectButton.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("설치된 3.3.1 split 구조 확인 중…");

        executor.execute(new Runnable() {
            @Override public void run() {
                try {
                    final ReportResult result = buildReport();
                    pendingReport = result.file;
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            progressBar.setVisibility(View.GONE);
                            statusView.setText(String.format(
                                    Locale.US,
                                    "확인 완료: arm64 .so %d개 — JSON 저장 위치를 선택하세요.",
                                    result.nativeCount));
                            launchSavePicker(result.suggestedName);
                        }
                    });
                } catch (PackageManager.NameNotFoundException error) {
                    showError("ADOFAI가 설치되어 있지 않거나 패키지를 확인할 수 없습니다.");
                } catch (Exception error) {
                    showError("확인 실패: " + error.getClass().getSimpleName() + ": " + safeMessage(error));
                }
            }
        });
    }

    private ReportResult buildReport() throws Exception {
        PackageManager pm = getPackageManager();
        PackageInfo packageInfo = pm.getPackageInfo(TARGET_PACKAGE, 0);
        ApplicationInfo appInfo = pm.getApplicationInfo(TARGET_PACKAGE, 0);

        long versionCode = Build.VERSION.SDK_INT >= 28
                ? packageInfo.getLongVersionCode()
                : packageInfo.versionCode;
        String versionName = packageInfo.versionName == null ? "unknown" : packageInfo.versionName;
        if (!EXPECTED_VERSION_NAME.equals(versionName) || versionCode != EXPECTED_VERSION_CODE) {
            throw new IllegalStateException(
                    "검증 대상 불일치: expected " + EXPECTED_VERSION_NAME + "/" + EXPECTED_VERSION_CODE +
                    ", got " + versionName + "/" + versionCode);
        }

        List<String> apkPaths = new ArrayList<>();
        apkPaths.add(appInfo.sourceDir);
        if (appInfo.splitSourceDirs != null) {
            apkPaths.addAll(Arrays.asList(appInfo.splitSourceDirs));
        }

        JSONArray installedApks = new JSONArray();
        List<NativeEntry> nativeEntries = new ArrayList<>();
        for (String apkPath : apkPaths) {
            File apkFile = new File(apkPath);
            JSONObject apkObject = new JSONObject();
            apkObject.put("name", apkFile.getName());
            apkObject.put("size_bytes", apkFile.length());
            apkObject.put("is_base", apkPath.equals(appInfo.sourceDir));
            installedApks.put(apkObject);

            try (ZipFile apk = new ZipFile(apkFile)) {
                Enumeration<? extends ZipEntry> entries = apk.entries();
                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();
                    if (entry.isDirectory()) continue;
                    String name = entry.getName();
                    if (!name.startsWith(ARM64_PREFIX) || !name.endsWith(".so")) continue;
                    nativeEntries.add(new NativeEntry(
                            apkFile.getName(),
                            name,
                            entry.getSize(),
                            entry.getCompressedSize(),
                            entry.getMethod()));
                }
            }
        }

        Collections.sort(nativeEntries, new Comparator<NativeEntry>() {
            @Override public int compare(NativeEntry left, NativeEntry right) {
                int split = left.sourceApk.compareTo(right.sourceApk);
                return split != 0 ? split : left.entry.compareTo(right.entry);
            }
        });

        JSONArray nativeArray = new JSONArray();
        boolean hasMain = false;
        boolean hasUnity = false;
        boolean hasIl2cpp = false;
        boolean hasOctober = false;
        for (NativeEntry entry : nativeEntries) {
            JSONObject object = new JSONObject();
            object.put("source_apk", entry.sourceApk);
            object.put("entry", entry.entry);
            object.put("size_bytes", entry.sizeBytes);
            object.put("compressed_size_bytes", entry.compressedSizeBytes);
            object.put("zip_method", zipMethodName(entry.method));
            nativeArray.put(object);

            String leaf = entry.entry.substring(entry.entry.lastIndexOf('/') + 1);
            if ("libmain.so".equals(leaf)) hasMain = true;
            if ("libunity.so".equals(leaf)) hasUnity = true;
            if ("libil2cpp.so".equals(leaf)) hasIl2cpp = true;
            if ("libOctober.so".equals(leaf)) hasOctober = true;
        }

        JSONObject signals = new JSONObject();
        signals.put("libmain_present", hasMain);
        signals.put("libunity_present", hasUnity);
        signals.put("libil2cpp_present", hasIl2cpp);
        signals.put("libOctober_present", hasOctober);
        signals.put("native_library_dir_leaf", appInfo.nativeLibraryDir == null
                ? JSONObject.NULL
                : new File(appInfo.nativeLibraryDir).getName());

        JSONObject report = new JSONObject();
        report.put("format", "adofai-native-inventory-v1");
        report.put("package", TARGET_PACKAGE);
        report.put("version_name", versionName);
        report.put("version_code", versionCode);
        report.put("exact_validated_target", true);
        report.put("device_manufacturer", Build.MANUFACTURER);
        report.put("device_model", Build.MODEL);
        report.put("android_sdk", Build.VERSION.SDK_INT);
        report.put("supported_abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)));
        report.put("installed_apks", installedApks);
        report.put("arm64_native_entries", nativeArray);
        report.put("signals", signals);

        JSONArray notes = new JSONArray();
        notes.put("This report records ZIP entry metadata only; no game native library bytes are exported.");
        notes.put("Library presence alone does not prove Android/Unity automatically loads that library.");
        notes.put("The report is evidence for choosing the next phone-only bootstrap experiment, not device-runtime proof.");
        report.put("evidence_notes", notes);

        String suggestedName = "adofai-native-inventory-3.3.1-300382.json";
        File outFile = new File(getCacheDir(), suggestedName);
        try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outFile))) {
            out.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new ReportResult(outFile, suggestedName, nativeEntries.size());
    }

    private static String zipMethodName(int method) {
        if (method == ZipEntry.STORED) return "stored";
        if (method == ZipEntry.DEFLATED) return "deflated";
        return "method-" + method;
    }

    private void launchSavePicker(String suggestedName) {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        intent.putExtra(Intent.EXTRA_TITLE, suggestedName);
        try {
            startActivityForResult(intent, REQUEST_SAVE);
        } catch (Exception error) {
            pendingReport = null;
            showError("저장 선택기를 열 수 없습니다: " + safeMessage(error));
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_SAVE) return;

        if (resultCode != RESULT_OK || data == null || data.getData() == null || pendingReport == null) {
            inspectButton.setEnabled(true);
            statusView.setText("저장이 취소되었습니다. 다시 실행해도 됩니다.");
            pendingReport = null;
            return;
        }

        final Uri destination = data.getData();
        final File source = pendingReport;
        pendingReport = null;
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText("JSON 저장 중…");
        executor.execute(new Runnable() {
            @Override public void run() {
                try (InputStream in = new BufferedInputStream(new FileInputStream(source));
                     OutputStream out = new BufferedOutputStream(
                             getContentResolver().openOutputStream(destination, "w"))) {
                    if (out == null) throw new IllegalStateException("출력 스트림을 열 수 없습니다.");
                    byte[] buffer = new byte[16 * 1024];
                    int count;
                    while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                    out.flush();
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            progressBar.setVisibility(View.GONE);
                            inspectButton.setEnabled(true);
                            statusView.setText("저장 완료. 생성된 JSON만 ChatGPT에 올리면 됩니다.");
                        }
                    });
                } catch (Exception error) {
                    showError("저장 실패: " + safeMessage(error));
                } finally {
                    source.delete();
                }
            }
        });
    }

    private void showError(final String message) {
        runOnUiThread(new Runnable() {
            @Override public void run() {
                progressBar.setVisibility(View.GONE);
                inspectButton.setEnabled(true);
                statusView.setText(message);
            }
        });
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? "no detail" : message;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static final class NativeEntry {
        final String sourceApk;
        final String entry;
        final long sizeBytes;
        final long compressedSizeBytes;
        final int method;

        NativeEntry(String sourceApk, String entry, long sizeBytes, long compressedSizeBytes, int method) {
            this.sourceApk = sourceApk;
            this.entry = entry;
            this.sizeBytes = sizeBytes;
            this.compressedSizeBytes = compressedSizeBytes;
            this.method = method;
        }
    }

    private static final class ReportResult {
        final File file;
        final String suggestedName;
        final int nativeCount;

        ReportResult(File file, String suggestedName, int nativeCount) {
            this.file = file;
            this.suggestedName = suggestedName;
            this.nativeCount = nativeCount;
        }
    }
}
