package dev.hoonex.adofai.nativeprobe;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ComponentName;
import android.content.DialogInterface;
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
import android.widget.ScrollView;
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
 * Evidence probe for the exact Play-installed ADOFAI 3.3.1 split install.
 *
 * Inventory/report generation is read-only: it records APK/split metadata,
 * native library inventory and PackageManager routing evidence. A separate,
 * explicit user action can launch the already-exported official
 * UnityPlayerActivity with a selected chart URI to test whether the unmodified
 * game consumes Android Intent data. No game APK, signature, data or binaries
 * are modified by this app.
 */
public final class MainActivity extends Activity {
    private static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";
    private static final String TARGET_ACTIVITY = "com.unity3d.player.UnityPlayerActivity";
    private static final String EXPECTED_VERSION_NAME = "3.3.1";
    private static final long EXPECTED_VERSION_CODE = 300382L;
    private static final String ARM64_PREFIX = "lib/arm64-v8a/";
    private static final String REMOTE_PROBE_URL =
            "https://raw.githubusercontent.com/hoonex/adofai/feat/modern-mobile-editor/tests/fixtures/explicit-handoff-probe.adofai";
    private static final int REQUEST_SAVE = 2301;
    private static final int REQUEST_HANDOFF_FILE = 2302;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView statusView;
    private ProgressBar progressBar;
    private Button inspectButton;
    private Button handoffButton;
    private File pendingReport;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int padding = dp(24);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText("ADOFAI 3.3.1 Handoff Probe");
        title.setTextSize(23f);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "첫 버튼은 설치된 공식 Play판 3.3.1의 split/네이티브 구성과 공개 Intent 처리 가능성만 읽습니다. " +
                "두 번째 버튼은 사용자가 선택한 .adofai를 exported UnityPlayerActivity에 명시적으로 전달해 " +
                "공식 게임이 실제로 URI를 소비하는지 확인합니다. APK, 서명, 세이브 데이터는 수정하지 않습니다.");
        description.setTextSize(15f);
        LinearLayout.LayoutParams descriptionParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        descriptionParams.topMargin = dp(16);
        root.addView(description, descriptionParams);

        inspectButton = new Button(this);
        inspectButton.setText("3.3.1 진단 보고서 만들기");
        inspectButton.setAllCaps(false);
        LinearLayout.LayoutParams inspectParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        inspectParams.topMargin = dp(24);
        root.addView(inspectButton, inspectParams);

        handoffButton = new Button(this);
        handoffButton.setText("공식 3.3.1 explicit handoff 테스트");
        handoffButton.setAllCaps(false);
        LinearLayout.LayoutParams handoffParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        handoffParams.topMargin = dp(12);
        root.addView(handoffButton, handoffParams);

        progressBar = new ProgressBar(this);
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        progressParams.topMargin = dp(18);
        root.addView(progressBar, progressParams);

        statusView = new TextView(this);
        statusView.setText("준비됨 — 먼저 진단 후 explicit handoff를 시험할 수 있습니다.");
        statusView.setTextSize(14f);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        statusParams.topMargin = dp(16);
        root.addView(statusView, statusParams);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);

        inspectButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                beginInspection();
            }
        });
        handoffButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                launchHandoffPicker();
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
        statusView.setText("설치된 3.3.1 split/인텐트 구조 확인 중…");

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

    private void assertExactTarget() throws Exception {
        PackageInfo packageInfo = getPackageManager().getPackageInfo(TARGET_PACKAGE, 0);
        long versionCode = Build.VERSION.SDK_INT >= 28
                ? packageInfo.getLongVersionCode()
                : packageInfo.versionCode;
        String versionName = packageInfo.versionName == null ? "unknown" : packageInfo.versionName;
        if (!EXPECTED_VERSION_NAME.equals(versionName) || versionCode != EXPECTED_VERSION_CODE) {
            throw new IllegalStateException(
                    "검증 대상 불일치: expected " + EXPECTED_VERSION_NAME + "/" + EXPECTED_VERSION_CODE +
                    ", got " + versionName + "/" + versionCode);
        }
    }

    private ReportResult buildReport() throws Exception {
        PackageManager pm = getPackageManager();
        assertExactTarget();
        PackageInfo packageInfo = pm.getPackageInfo(TARGET_PACKAGE, 0);
        ApplicationInfo appInfo = pm.getApplicationInfo(TARGET_PACKAGE, 0);

        long versionCode = Build.VERSION.SDK_INT >= 28
                ? packageInfo.getLongVersionCode()
                : packageInfo.versionCode;
        String versionName = packageInfo.versionName == null ? "unknown" : packageInfo.versionName;

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
        report.put("format", "adofai-native-inventory-v2");
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
        report.put("external_file_intents", IntentCapabilityProbe.build(pm, TARGET_PACKAGE));

        JSONArray notes = new JSONArray();
        notes.put("This report records ZIP entry metadata only; no game native library bytes are exported.");
        notes.put("Library presence alone does not prove Android/Unity automatically loads that library.");
        notes.put("ACTION_VIEW/SEND resolution proves only Android routing capability, not that ADOFAI consumes a URI as a custom level.");
        notes.put("Generating this inventory report does not launch ADOFAI. The separate explicit handoff button is an opt-in launch test.");
        report.put("evidence_notes", notes);

        String suggestedName = "adofai-native-inventory-3.3.1-300382.json";
        File outFile = new File(getCacheDir(), suggestedName);
        try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outFile))) {
            out.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new ReportResult(outFile, suggestedName, nativeEntries.size());
    }

    private void launchHandoffPicker() {
        try {
            assertExactTarget();
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_HANDOFF_FILE);
        } catch (Exception error) {
            showError("handoff 준비 실패: " + safeMessage(error));
        }
    }

    private void showHandoffVariantDialog(final Uri chartUri) {
        final String[] variants = new String[] {
                "A · VIEW + content URI + application/json",
                "B · VIEW + content URI (MIME 없음)",
                "C · SEND + EXTRA_STREAM + application/json",
                "D · VIEW + URI + 호환 extra 후보",
                "E · VIEW + HTTPS .adofai URL"
        };
        new AlertDialog.Builder(this)
                .setTitle("공식 게임 handoff 방식 선택")
                .setMessage("게임이 단순히 실행되는 것과 선택한 커스텀 레벨/에디터가 실제로 열리는 것은 다릅니다. 각 방식을 시험한 뒤 결과를 확인하세요.")
                .setItems(variants, new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        launchExplicitHandoff(chartUri, which);
                    }
                })
                .setNegativeButton("취소", null)
                .show();
    }

    private void launchExplicitHandoff(Uri chartUri, int variant) {
        try {
            assertExactTarget();
            ComponentName target = new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY);
            Intent intent;
            String label;

            if (variant == 0) {
                intent = new Intent(Intent.ACTION_VIEW);
                intent.setDataAndType(chartUri, "application/json");
                attachReadGrant(intent, chartUri);
                label = "A: VIEW + content URI + application/json";
            } else if (variant == 1) {
                intent = new Intent(Intent.ACTION_VIEW);
                intent.setData(chartUri);
                attachReadGrant(intent, chartUri);
                label = "B: VIEW + content URI";
            } else if (variant == 2) {
                intent = new Intent(Intent.ACTION_SEND);
                intent.setType("application/json");
                intent.putExtra(Intent.EXTRA_STREAM, chartUri);
                attachReadGrant(intent, chartUri);
                label = "C: SEND + stream";
            } else if (variant == 3) {
                intent = new Intent(Intent.ACTION_VIEW);
                intent.setData(chartUri);
                attachReadGrant(intent, chartUri);
                String encoded = chartUri.toString();
                intent.putExtra("url", encoded);
                intent.putExtra("path", encoded);
                intent.putExtra("filePath", encoded);
                intent.putExtra("levelPath", encoded);
                intent.putExtra("adofai", encoded);
                intent.putExtra(Intent.EXTRA_STREAM, chartUri);
                label = "D: VIEW + compatibility extras";
            } else {
                Uri remote = Uri.parse(REMOTE_PROBE_URL);
                intent = new Intent(Intent.ACTION_VIEW);
                intent.setData(remote);
                intent.putExtra(Intent.EXTRA_TEXT, REMOTE_PROBE_URL);
                label = "E: VIEW + HTTPS .adofai URL";
            }

            intent.setComponent(target);
            startActivity(intent);
            statusView.setText(
                    label + " 로 공식 ADOFAI를 실행했습니다. " +
                    "커스텀 레벨/에디터가 실제로 열렸는지 확인한 뒤 이 앱으로 돌아와 다른 방식을 시험하세요.");
        } catch (Throwable error) {
            showError("explicit handoff 실행 실패: " + error.getClass().getSimpleName() + ": " + safeMessage(error));
        }
    }

    private static void attachReadGrant(Intent intent, Uri uri) {
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.setClipData(ClipData.newRawUri("ADOFAI chart", uri));
    }

    private void takePersistableReadPermission(Uri uri, int resultFlags) {
        int flags = resultFlags & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (flags == 0) return;
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // Some document providers grant only session access. That is enough for an immediate handoff test.
        }
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

        if (requestCode == REQUEST_HANDOFF_FILE) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) {
                statusView.setText("handoff 파일 선택이 취소되었습니다.");
                return;
            }
            Uri chartUri = data.getData();
            takePersistableReadPermission(chartUri, data.getFlags());
            statusView.setText("파일 선택 완료. 공식 게임에 전달할 방식을 선택하세요.");
            showHandoffVariantDialog(chartUri);
            return;
        }

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
                handoffButton.setEnabled(true);
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
