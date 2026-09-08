package dev.hoonex.adofai.exporter;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.os.Build;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

final class CompatibilityInspector {
    static final String TARGET_PACKAGE = "com.fizzd.connectedworlds";

    private static final String METADATA_ENTRY = "assets/bin/Data/Managed/Metadata/global-metadata.dat";
    private static final String IL2CPP_ENTRY = "lib/arm64-v8a/libil2cpp.so";
    private static final String OCTOBER_ENTRY = "lib/arm64-v8a/libOctober.so";
    private static final String MANIFEST_ENTRY = "AndroidManifest.xml";

    private static final String[] CORE_METADATA_SYMBOLS = new String[] {
            "ADOBase",
            "get_isMobile",
            "get_isUnityEditor",
            "ADOStartup",
            "Startup",
            "StandaloneFileBrowser",
            "OpenFilePanel",
            "SaveFilePanel",
            "OpenFolderPanel"
    };

    private static final String[] HOOK_METADATA_SYMBOLS = new String[] {
            "ADOBase",
            "get_isMobile",
            "get_isUnityEditor",
            "ADOStartup",
            "Startup",
            "scrController",
            "QuitToMainMenu",
            "RestartProgress",
            "LevelEventInfo",
            "get_isActive",
            "scnGame",
            "Play",
            "PauseMenu",
            "RefreshLayout",
            "scrEnableIfBeta",
            "Awake",
            "RDC",
            "get_forceUnlockAllLevels",
            "scrPlanet",
            "GetMultipressPenalty",
            "scrMisc",
            "DetermineDifficultyUIMode",
            "get_taroDLCCheck",
            "OttoButtonController",
            "Update",
            "scrUIController",
            "scrRing",
            "ShowHitText",
            "GCNS",
            "get_BundlesLoadPath",
            "IsScreenPointInsideUIElements",
            "StandaloneFileBrowser",
            "OpenFilePanel",
            "SaveFilePanel",
            "OpenFolderPanel"
    };

    private static final String[] BOOTSTRAP_DEX_TOKENS = new String[] {
            "October",
            "loadLibrary",
            "UnityPlayerActivity"
    };

    private final Context context;

    CompatibilityInspector(Context context) {
        this.context = context.getApplicationContext();
    }

    ReportResult buildReport() throws Exception {
        PackageManager pm = context.getPackageManager();
        PackageInfo packageInfo;
        if (Build.VERSION.SDK_INT >= 28) {
            packageInfo = pm.getPackageInfo(TARGET_PACKAGE, PackageManager.GET_SIGNING_CERTIFICATES);
        } else {
            packageInfo = pm.getPackageInfo(TARGET_PACKAGE, PackageManager.GET_SIGNATURES);
        }
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
        JSONArray criticalEntries = new JSONArray();
        JSONArray dexEntries = new JSONArray();
        JSONArray octoberEntries = new JSONArray();
        JSONArray bootstrapCandidates = new JSONArray();
        JSONObject manifestSignals = new JSONObject();
        JSONObject metadataAnalysis = null;
        boolean metadataFound = false;
        boolean il2cppFound = false;
        boolean octoberFound = false;
        boolean bootstrapLikely = false;

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
                    if (entry.isDirectory()) {
                        continue;
                    }
                    String name = entry.getName();

                    if (isDexEntry(name)) {
                        JSONObject dex = new JSONObject();
                        dex.put("source_apk", apkFile.getName());
                        dex.put("entry", name);
                        dex.put("size_bytes", entry.getSize());
                        dexEntries.put(dex);

                        Map<String, Boolean> hits = scanTokens(apk.getInputStream(entry), BOOTSTRAP_DEX_TOKENS);
                        boolean candidate = allFound(hits, BOOTSTRAP_DEX_TOKENS);
                        if (candidate) {
                            bootstrapLikely = true;
                        }
                        if (anyFound(hits)) {
                            JSONObject hitObject = new JSONObject();
                            hitObject.put("source_apk", apkFile.getName());
                            hitObject.put("entry", name);
                            hitObject.put("tokens", booleanMapToJson(hits));
                            hitObject.put("all_bootstrap_tokens_in_same_dex", candidate);
                            bootstrapCandidates.put(hitObject);
                        }
                    }
                }

                ZipEntry metadataEntry = apk.getEntry(METADATA_ENTRY);
                if (!metadataFound && metadataEntry != null && !metadataEntry.isDirectory()) {
                    metadataFound = true;
                    metadataAnalysis = analyzeMetadata(apk, metadataEntry, apkFile.getName());
                    criticalEntries.put(entryInfo(apkFile.getName(), metadataEntry, sha256Entry(apk, metadataEntry)));
                }

                ZipEntry il2cppEntry = apk.getEntry(IL2CPP_ENTRY);
                if (!il2cppFound && il2cppEntry != null && !il2cppEntry.isDirectory()) {
                    il2cppFound = true;
                    criticalEntries.put(entryInfo(apkFile.getName(), il2cppEntry, sha256Entry(apk, il2cppEntry)));
                }

                ZipEntry octoberEntry = apk.getEntry(OCTOBER_ENTRY);
                if (octoberEntry != null && !octoberEntry.isDirectory()) {
                    octoberFound = true;
                    JSONObject october = entryInfo(apkFile.getName(), octoberEntry, sha256Entry(apk, octoberEntry));
                    octoberEntries.put(october);
                    criticalEntries.put(october);
                }

                if (apkPath.equals(appInfo.sourceDir)) {
                    ZipEntry manifestEntry = apk.getEntry(MANIFEST_ENTRY);
                    if (manifestEntry != null && !manifestEntry.isDirectory()) {
                        manifestSignals = inspectManifest(apk, manifestEntry);
                    }
                }
            }
        }

        JSONObject report = new JSONObject();
        report.put("format", "adofai-runtime-compat-v2");
        report.put("package", TARGET_PACKAGE);
        report.put("version_name", versionName);
        report.put("version_code", versionCode);
        report.put("installer_package", pm.getInstallerPackageName(TARGET_PACKAGE));
        report.put("device_manufacturer", Build.MANUFACTURER);
        report.put("device_model", Build.MODEL);
        report.put("android_sdk", Build.VERSION.SDK_INT);
        report.put("supported_abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)));
        report.put("signer_sha256", signerDigests(packageInfo));
        report.put("installed_apks", installedApks);
        report.put("critical_entries", criticalEntries);
        report.put("dex_entries", dexEntries);
        report.put("october_entries", octoberEntries);
        report.put("bootstrap_candidates", bootstrapCandidates);
        report.put("manifest_signals", manifestSignals);
        report.put("metadata", metadataAnalysis == null ? JSONObject.NULL : metadataAnalysis);

        JSONObject readiness = new JSONObject();
        boolean metadataMagicOk = metadataAnalysis != null && metadataAnalysis.optBoolean("magic_ok", false);
        boolean coreSymbolsOk = metadataAnalysis != null && metadataAnalysis.optBoolean("core_symbols_all_present", false);
        boolean allHookSymbols = metadataAnalysis != null && metadataAnalysis.optBoolean("all_hook_symbols_present", false);
        boolean manageStorageDeclared = manifestSignals.optBoolean("manage_external_storage_declared", false);
        readiness.put("metadata_found", metadataFound);
        readiness.put("metadata_magic_ok", metadataMagicOk);
        readiness.put("arm64_libil2cpp_found", il2cppFound);
        readiness.put("libOctober_found", octoberFound);
        readiness.put("existing_october_bootstrap_likely", bootstrapLikely);
        readiness.put("core_hook_names_present", coreSymbolsOk);
        readiness.put("all_configured_hook_names_present", allHookSymbols);
        readiness.put("manage_external_storage_declared", manageStorageDeclared);

        String classification;
        if (!metadataFound || !metadataMagicOk || !il2cppFound) {
            classification = "unsupported_install_layout";
        } else if (!coreSymbolsOk) {
            classification = "hook_names_changed";
        } else if (!octoberFound || !bootstrapLikely) {
            classification = "native_bootstrap_patch_required";
        } else if (!manageStorageDeclared) {
            classification = "storage_manifest_patch_required";
        } else if (!allHookSymbols) {
            classification = "core_compatible_optional_hooks_need_review";
        } else {
            classification = "structurally_promising_device_test_required";
        }
        readiness.put("classification", classification);
        readiness.put("device_runtime_verified", false);
        report.put("patch_readiness", readiness);

        JSONArray notes = new JSONArray();
        notes.put("String presence in IL2CPP metadata is structural evidence only; it does not prove method signatures or runtime hook safety.");
        notes.put("existing_october_bootstrap_likely requires October, loadLibrary, and UnityPlayerActivity tokens in the same DEX entry.");
        notes.put("Real-device editor open/edit/save/reopen behavior remains unverified until a patched build is installed and exercised.");
        report.put("evidence_notes", notes);

        String safeVersion = versionName.replaceAll("[^A-Za-z0-9._-]", "_");
        String suggestedName = "adofai-compat-" + safeVersion + "-" + versionCode + ".json";
        File outFile = new File(context.getCacheDir(), suggestedName);
        try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outFile))) {
            out.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new ReportResult(outFile, suggestedName, classification);
    }

    private JSONObject analyzeMetadata(ZipFile apk, ZipEntry entry, String sourceApk) throws Exception {
        byte[] header = new byte[8];
        int headerRead = 0;
        try (InputStream in = new BufferedInputStream(apk.getInputStream(entry))) {
            while (headerRead < header.length) {
                int read = in.read(header, headerRead, header.length - headerRead);
                if (read < 0) {
                    break;
                }
                headerRead += read;
            }
        }

        int magic = 0;
        int metadataVersion = -1;
        if (headerRead == 8) {
            ByteBuffer buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            magic = buffer.getInt();
            metadataVersion = buffer.getInt();
        }
        boolean magicOk = magic == 0xFAB11BAF;

        Map<String, Boolean> symbolHits;
        try (InputStream in = new BufferedInputStream(apk.getInputStream(entry))) {
            symbolHits = scanTokens(in, HOOK_METADATA_SYMBOLS);
        }

        JSONObject result = new JSONObject();
        result.put("source_apk", sourceApk);
        result.put("entry", entry.getName());
        result.put("size_bytes", entry.getSize());
        result.put("sha256", sha256Entry(apk, entry));
        result.put("magic", String.format(Locale.US, "0x%08x", magic));
        result.put("magic_ok", magicOk);
        result.put("metadata_version", metadataVersion);
        result.put("symbols", booleanMapToJson(symbolHits));
        result.put("core_symbols_all_present", allFound(symbolHits, CORE_METADATA_SYMBOLS));
        result.put("all_hook_symbols_present", allFound(symbolHits, HOOK_METADATA_SYMBOLS));

        JSONArray missingCore = new JSONArray();
        for (String symbol : CORE_METADATA_SYMBOLS) {
            if (!Boolean.TRUE.equals(symbolHits.get(symbol))) {
                missingCore.put(symbol);
            }
        }
        JSONArray missingAll = new JSONArray();
        for (String symbol : HOOK_METADATA_SYMBOLS) {
            if (!Boolean.TRUE.equals(symbolHits.get(symbol))) {
                missingAll.put(symbol);
            }
        }
        result.put("missing_core_symbols", missingCore);
        result.put("missing_hook_symbols", missingAll);
        return result;
    }

    private JSONObject inspectManifest(ZipFile apk, ZipEntry entry) throws Exception {
        String[] tokens = new String[] {
                "android.permission.MANAGE_EXTERNAL_STORAGE",
                "android.permission.READ_EXTERNAL_STORAGE",
                "android.permission.WRITE_EXTERNAL_STORAGE",
                "requestLegacyExternalStorage"
        };
        byte[] bytes = readEntryWithLimit(apk, entry, 8 * 1024 * 1024);
        JSONObject result = new JSONObject();
        result.put("size_bytes", entry.getSize());
        result.put("sha256", sha256(bytes));
        for (String token : tokens) {
            boolean found = contains(bytes, token.getBytes(StandardCharsets.UTF_8))
                    || contains(bytes, utf16Le(token));
            if (token.endsWith("MANAGE_EXTERNAL_STORAGE")) {
                result.put("manage_external_storage_declared", found);
            } else if (token.endsWith("READ_EXTERNAL_STORAGE")) {
                result.put("read_external_storage_declared", found);
            } else if (token.endsWith("WRITE_EXTERNAL_STORAGE")) {
                result.put("write_external_storage_declared", found);
            } else {
                result.put("request_legacy_external_storage_present", found);
            }
        }
        return result;
    }

    private static JSONObject entryInfo(String sourceApk, ZipEntry entry, String digest) throws Exception {
        JSONObject object = new JSONObject();
        object.put("source_apk", sourceApk);
        object.put("entry", entry.getName());
        object.put("size_bytes", entry.getSize());
        object.put("compressed_size_bytes", entry.getCompressedSize());
        object.put("sha256", digest);
        return object;
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

    private static boolean isDexEntry(String name) {
        if ("classes.dex".equals(name)) {
            return true;
        }
        return name.matches("classes[0-9]+\\.dex");
    }

    private static Map<String, Boolean> scanTokens(InputStream raw, String[] tokens) throws Exception {
        byte[][] needles = new byte[tokens.length][];
        int maxNeedle = 1;
        for (int i = 0; i < tokens.length; i++) {
            needles[i] = tokens[i].getBytes(StandardCharsets.UTF_8);
            maxNeedle = Math.max(maxNeedle, needles[i].length);
        }

        boolean[] found = new boolean[tokens.length];
        byte[] readBuffer = new byte[256 * 1024];
        byte[] carry = new byte[0];
        try (InputStream in = new BufferedInputStream(raw)) {
            int read;
            while ((read = in.read(readBuffer)) != -1) {
                byte[] window = new byte[carry.length + read];
                System.arraycopy(carry, 0, window, 0, carry.length);
                System.arraycopy(readBuffer, 0, window, carry.length, read);

                boolean complete = true;
                for (int i = 0; i < needles.length; i++) {
                    if (!found[i] && contains(window, needles[i])) {
                        found[i] = true;
                    }
                    if (!found[i]) {
                        complete = false;
                    }
                }
                if (complete) {
                    break;
                }

                int keep = Math.min(maxNeedle - 1, window.length);
                carry = new byte[keep];
                System.arraycopy(window, window.length - keep, carry, 0, keep);
            }
        }

        Map<String, Boolean> result = new LinkedHashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            result.put(tokens[i], found[i]);
        }
        return result;
    }

    private static boolean anyFound(Map<String, Boolean> hits) {
        for (Boolean value : hits.values()) {
            if (Boolean.TRUE.equals(value)) {
                return true;
            }
        }
        return false;
    }

    private static boolean allFound(Map<String, Boolean> hits, String[] required) {
        for (String token : required) {
            if (!Boolean.TRUE.equals(hits.get(token))) {
                return false;
            }
        }
        return true;
    }

    private static JSONObject booleanMapToJson(Map<String, Boolean> map) throws Exception {
        JSONObject object = new JSONObject();
        for (Map.Entry<String, Boolean> entry : map.entrySet()) {
            object.put(entry.getKey(), entry.getValue());
        }
        return object;
    }

    private static byte[] readEntryWithLimit(ZipFile zip, ZipEntry entry, int maxBytes) throws Exception {
        if (entry.getSize() > maxBytes) {
            throw new IllegalStateException("entry too large for bounded read: " + entry.getName());
        }
        int expected = entry.getSize() > 0 && entry.getSize() <= Integer.MAX_VALUE
                ? (int) entry.getSize()
                : 4096;
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream(expected);
        try (InputStream in = new BufferedInputStream(zip.getInputStream(entry))) {
            byte[] buffer = new byte[64 * 1024];
            int total = 0;
            int read;
            while ((read = in.read(buffer)) != -1) {
                total += read;
                if (total > maxBytes) {
                    throw new IllegalStateException("entry exceeded bounded read: " + entry.getName());
                }
                out.write(buffer, 0, read);
            }
        }
        return out.toByteArray();
    }

    private static boolean contains(byte[] haystack, byte[] needle) {
        if (needle.length == 0) {
            return true;
        }
        if (needle.length > haystack.length) {
            return false;
        }
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    private static byte[] utf16Le(String value) {
        return value.getBytes(StandardCharsets.UTF_16LE);
    }

    private static String sha256Entry(ZipFile zip, ZipEntry entry) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(zip.getInputStream(entry))) {
            byte[] buffer = new byte[256 * 1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        return hex(digest.digest());
    }

    private static String sha256(byte[] bytes) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return hex(digest.digest(bytes));
    }

    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            out.append(String.format(Locale.US, "%02x", value & 0xff));
        }
        return out.toString();
    }

    static final class ReportResult {
        final File file;
        final String suggestedName;
        final String classification;

        ReportResult(File file, String suggestedName, String classification) {
            this.file = file;
            this.suggestedName = suggestedName;
            this.classification = classification;
        }
    }
}
