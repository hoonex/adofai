package com.unity3d.player;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import dalvik.system.DexClassLoader;

/**
 * Stable bootstrap-owned runtime updater.
 *
 * Runtime code is staged only inside app-private code_cache. A downloaded runtime is
 * never executed in the process that downloaded it. The next process marks a candidate
 * boot pending before native/DEX loading; an interrupted launch quarantines that exact
 * commit SHA and rolls back without ever re-downloading the quarantined SHA.
 */
final class V240RuntimeUpdater {
    private static final String TAG = "ADOFAI.V240Updater";
    private static final int BOOTSTRAP_VERSION = 2;
    private static final int MANIFEST_SCHEMA = 1;
    private static final long HEALTH_DELAY_MS = 10_000L;
    private static final long MIN_CHECK_INTERVAL_MS = 5L * 60L * 1000L;
    private static final long MAX_MANIFEST_BYTES = 64L * 1024L;
    private static final long MAX_BUNDLE_BYTES = 32L * 1024L * 1024L;
    private static final long MAX_ENTRY_BYTES = 24L * 1024L * 1024L;
    private static final String CHANNEL = "github-release:v240-runtime-channel";
    private static final String SOURCE_REPOSITORY = "hoonex/adofai";
    private static final String CHANNEL_MANIFEST =
            "https://github.com/hoonex/adofai/releases/download/v240-runtime-channel/manifest.json";
    private static final String RELEASE_PREFIX =
            "/hoonex/adofai/releases/download/v240-runtime-channel/";
    private static final String ENTRY_CLASS =
            "dev.hoonex.adofai.v240.dynamic.RuntimeEntry";
    private static final String PREFS = "adofai-v240-runtime-updater";

    private static boolean started;
    private static volatile boolean cachedRuntimeLoaded;
    private static volatile String loadedVersion = "none";
    private static volatile String availableVersion = "none";
    private static volatile String downloadedVersion = "none";
    private static volatile String shaVerification = "not-checked";
    private static volatile String rollbackReason = "none";
    private static volatile String lastError = "none";
    private static volatile String channelState = "not-checked";
    private static File rootDir;
    private static File loadedDir;

    private V240RuntimeUpdater() {}

    static synchronized boolean startIfReady() {
        if (started) return true;
        Activity owner = currentActivity();
        if (owner == null || owner.isFinishing()) return false;
        started = true;
        Context app = owner.getApplicationContext();
        SharedPreferences prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        rollbackReason = prefs.getString("rollback_reason", "none");
        rootDir = new File(app.getCodeCacheDir(), "v240-runtime");
        if (!rootDir.isDirectory() && !rootDir.mkdirs()) {
            lastError = "code_cache_create_failed";
            Log.w(TAG, "could not create runtime code_cache");
            return true;
        }

        try {
            recoverInterruptedBoot(app);
            loadActiveCandidate(app);
        } catch (Throwable error) {
            lastError = "startup:" + safeMessage(error);
            Log.e(TAG, "cached runtime startup failed; keeping original game path alive", error);
            try { quarantineActive(app, "startup_exception"); } catch (Throwable ignored) {}
        }

        beginAsyncUpdateCheck(app);
        return true;
    }

    static boolean isCachedRuntimeLoaded() {
        return cachedRuntimeLoaded;
    }

    static String diagnosticText() {
        String active = rootDir == null ? "unknown" : readPointerQuiet(new File(rootDir, "active"));
        String previous = rootDir == null ? "unknown" : readPointerQuiet(new File(rootDir, "previous"));
        return "V240 runtime updater\n"
                + "bootstrapVersion=" + BOOTSTRAP_VERSION + "\n"
                + "codeCacheOnly=1\n"
                + "channel=" + CHANNEL + "\n"
                + "availableVersion=" + availableVersion + "\n"
                + "downloadedVersion=" + downloadedVersion + "\n"
                + "cachedRuntimeLoaded=" + (cachedRuntimeLoaded ? 1 : 0) + "\n"
                + "loadedVersion=" + loadedVersion + "\n"
                + "activeVersion=" + active + "\n"
                + "previousVersion=" + previous + "\n"
                + "shaVerification=" + shaVerification + "\n"
                + "channelState=" + channelState + "\n"
                + "rollbackReason=" + rollbackReason + "\n"
                + "lastError=" + lastError + "\n";
    }

    private static void loadActiveCandidate(final Context app) throws Exception {
        String version = readPointer(new File(rootDir, "active"));
        if (version == null) {
            channelState = "java-recovery-mode";
            Log.w(TAG, "no cache runtime active; original game path remains available");
            return;
        }
        if (isVersionQuarantined(version)) {
            recordRollback(app, "active_quarantined:" + version);
            clearActivePointer(version);
            restorePreviousPointer();
            version = readPointer(new File(rootDir, "active"));
            if (version == null) {
                channelState = "java-recovery-mode";
                return;
            }
        }

        File dir = versionDir(version);
        if (!isRuntimeDirValid(dir, version)) {
            lastError = "active_slot_invalid:" + version;
            quarantineActive(app, "invalid_active_slot");
            return;
        }

        final File pending = new File(dir, "boot.pending");
        writeSmallFile(pending, Long.toString(System.currentTimeMillis()));
        loadedDir = dir;
        loadedVersion = version;

        JSONObject meta = readJson(new File(dir, "meta.json"), 32 * 1024);
        if (!version.equals(meta.getString("version"))) {
            throw new IllegalStateException("runtime metadata version mismatch");
        }
        File nativeLib = new File(dir, "libv240fix.so");
        File runtimeDex = new File(dir, "runtime.dex");
        assertHash(nativeLib, meta.getString("nativeSha256"));
        assertHash(runtimeDex, meta.getString("dexSha256"));
        shaVerification = "payload-verified:" + version;

        // Android 14+ requires dynamically loaded code files not to remain writable.
        if (!runtimeDex.setReadOnly() && runtimeDex.canWrite()) {
            throw new IllegalStateException("runtime.dex remained writable");
        }
        if (!nativeLib.setReadOnly() && nativeLib.canWrite()) {
            throw new IllegalStateException("libv240fix.so remained writable");
        }

        // A native process abort cannot be caught in Java. boot.pending is deliberately
        // persisted before this call so the next launch quarantines this exact version.
        System.load(nativeLib.getAbsolutePath());

        File opt = new File(rootDir, "opt-" + version);
        if (!opt.isDirectory() && !opt.mkdirs()) {
            throw new IllegalStateException("dex optimized directory unavailable");
        }
        DexClassLoader loader = new DexClassLoader(
                runtimeDex.getAbsolutePath(), opt.getAbsolutePath(), null,
                V240RuntimeUpdater.class.getClassLoader());
        Class<?> entry = Class.forName(ENTRY_CLASS, true, loader);
        Method install = entry.getMethod("install", Context.class);
        install.invoke(null, app);

        cachedRuntimeLoaded = true;
        channelState = "active:" + version;
        lastError = "none";
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override public void run() {
                markHealthy(pending);
            }
        }, HEALTH_DELAY_MS);
    }

    private static synchronized void markHealthy(File pending) {
        if (!cachedRuntimeLoaded || loadedDir == null) return;
        if (pending.exists() && !pending.delete()) {
            Log.w(TAG, "could not clear boot.pending for " + loadedVersion);
            return;
        }
        try {
            writeSmallFile(new File(loadedDir, "healthy"), Long.toString(System.currentTimeMillis()));
        } catch (Throwable error) {
            Log.w(TAG, "could not persist runtime health marker", error);
        }
    }

    private static void recoverInterruptedBoot(Context app) throws Exception {
        String active = readPointer(new File(rootDir, "active"));
        if (active == null) return;
        if (isVersionQuarantined(active)) {
            recordRollback(app, "active_quarantined:" + active);
            clearActivePointer(active);
            restorePreviousPointer();
            return;
        }
        File dir = versionDir(active);
        if (!new File(dir, "boot.pending").isFile()) return;
        lastError = "rollback_unhealthy_boot:" + active;
        Log.w(TAG, "cached runtime did not reach health deadline; rolling back " + active);
        quarantineVersion(active, "boot_crash");
        recordRollback(app, "boot_crash:" + active);
        restorePreviousPointer();
    }

    private static void quarantineActive(Context app, String reason) throws Exception {
        String active = readPointer(new File(rootDir, "active"));
        if (active != null) {
            quarantineVersion(active, reason);
            recordRollback(app, reason + ":" + active);
        }
        restorePreviousPointer();
    }

    private static void quarantineVersion(String version, String reason) throws Exception {
        if (!isImmutableRuntimeVersion(version)) return;
        File marker = quarantineMarker(version);
        writeSmallFile(marker, reason + "\n" + System.currentTimeMillis() + "\n");
        File dir = versionDir(version);
        File bad = new File(rootDir, "bad-" + version + "-" + System.currentTimeMillis());
        if (dir.exists() && !dir.renameTo(bad)) safeDeleteTree(dir);
        Log.w(TAG, "quarantined runtime " + version + " reason=" + reason);
        clearActivePointer(version);
    }

    private static boolean isVersionQuarantined(String version) {
        return isImmutableRuntimeVersion(version) && quarantineMarker(version).isFile();
    }

    private static File quarantineMarker(String version) {
        if (!isImmutableRuntimeVersion(version)) throw new IllegalArgumentException("unsafe version");
        return new File(rootDir, "quarantine-" + version + ".txt");
    }

    private static void clearActivePointer(String version) throws Exception {
        File active = new File(rootDir, "active");
        String current = readPointer(active);
        if (version.equals(current) && active.exists() && !active.delete()) {
            throw new IllegalStateException("could not clear active pointer");
        }
    }

    private static void recordRollback(Context app, String reason) {
        rollbackReason = reason == null ? "unknown" : reason;
        try {
            app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                    .edit().putString("rollback_reason", rollbackReason).apply();
        } catch (Throwable ignored) {}
    }

    private static void restorePreviousPointer() throws Exception {
        String previous = readPointer(new File(rootDir, "previous"));
        if (previous != null && !isVersionQuarantined(previous)
                && isRuntimeDirValid(versionDir(previous), previous)) {
            writePointer(new File(rootDir, "active"), previous);
        } else {
            File active = new File(rootDir, "active");
            if (active.exists() && !active.delete()) {
                throw new IllegalStateException("could not remove invalid active pointer");
            }
        }
    }

    private static void beginAsyncUpdateCheck(final Context app) {
        final SharedPreferences prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        long now = System.currentTimeMillis();
        long last = prefs.getLong("last_check_ms", 0L);
        if (last > 0L && now >= last && now - last < MIN_CHECK_INTERVAL_MS) return;
        prefs.edit().putLong("last_check_ms", now).apply();

        Thread worker = new Thread(new Runnable() {
            @Override public void run() {
                try {
                    checkForUpdate(app);
                } catch (Throwable error) {
                    channelState = "check-failed";
                    lastError = "update:" + safeMessage(error);
                    Log.w(TAG, "runtime update check failed", error);
                }
            }
        }, "adofai-v240-updater");
        worker.setDaemon(true);
        worker.start();
    }

    private static void checkForUpdate(Context app) throws Exception {
        byte[] manifestBytes = fetchBytes(new URL(CHANNEL_MANIFEST), MAX_MANIFEST_BYTES);
        JSONObject manifest = new JSONObject(new String(manifestBytes, "UTF-8"));
        if (manifest.getInt("schema") != MANIFEST_SCHEMA) {
            throw new IllegalStateException("unsupported manifest schema");
        }
        if (!SOURCE_REPOSITORY.equals(manifest.optString("sourceRepository", ""))) {
            throw new IllegalStateException("unexpected runtime source repository");
        }
        if (manifest.getInt("minBootstrap") > BOOTSTRAP_VERSION) {
            channelState = "bootstrap-too-old";
            return;
        }
        if (!manifest.optBoolean("rollout", false)) {
            channelState = "rollout-disabled";
            return;
        }

        String version = manifest.getString("version");
        if (!isImmutableRuntimeVersion(version)) {
            throw new IllegalStateException("runtime version must be a commit SHA");
        }
        if (!version.equals(manifest.optString("sourceCommit", ""))) {
            throw new IllegalStateException("runtime source commit mismatch");
        }
        availableVersion = version;
        if (isVersionQuarantined(version)) {
            channelState = "quarantined-skip:" + version;
            lastError = "none";
            return;
        }

        String active = readPointer(new File(rootDir, "active"));
        if (version.equals(active) && isRuntimeDirValid(versionDir(version), version)) {
            channelState = "up-to-date:" + version;
            downloadedVersion = version;
            return;
        }

        URL bundleUrl = new URL(manifest.getString("bundleUrl"));
        requireReleaseUrl(bundleUrl);
        String expectedAsset = "/runtime-" + version + ".zip";
        if (!bundleUrl.getPath().endsWith(expectedAsset)) {
            throw new IllegalStateException("runtime asset is not commit-addressed");
        }
        long declaredBytes = manifest.optLong("bundleBytes", -1L);
        if (declaredBytes <= 0L || declaredBytes > MAX_BUNDLE_BYTES) {
            throw new IllegalStateException("bundle size outside safety bound");
        }
        String bundleHash = normalizedHash(manifest.getString("bundleSha256"));
        String nativeHash = normalizedHash(manifest.getString("nativeSha256"));
        String dexHash = normalizedHash(manifest.getString("dexSha256"));

        File download = new File(rootDir, "download.part");
        if (download.exists()) safeDelete(download);
        long actualBytes = downloadFile(bundleUrl, download, MAX_BUNDLE_BYTES);
        if (actualBytes != declaredBytes) {
            safeDelete(download);
            throw new IllegalStateException("bundle byte count mismatch");
        }
        assertHash(download, bundleHash);
        shaVerification = "bundle-verified:" + version;

        File staging = new File(rootDir, "staging-" + version);
        safeDeleteTree(staging);
        if (!staging.mkdirs()) throw new IllegalStateException("staging directory create failed");
        try {
            extractBundle(download, staging);
            File nativeLib = new File(staging, "libv240fix.so");
            File runtimeDex = new File(staging, "runtime.dex");
            assertHash(nativeLib, nativeHash);
            assertHash(runtimeDex, dexHash);
            shaVerification = "payload-verified:" + version;
            if (!runtimeDex.setReadOnly() && runtimeDex.canWrite()) {
                throw new IllegalStateException("downloaded runtime.dex remained writable");
            }
            if (!nativeLib.setReadOnly() && nativeLib.canWrite()) {
                throw new IllegalStateException("downloaded native library remained writable");
            }

            JSONObject meta = new JSONObject();
            meta.put("schema", MANIFEST_SCHEMA);
            meta.put("version", version);
            meta.put("sourceRepository", SOURCE_REPOSITORY);
            meta.put("nativeSha256", nativeHash);
            meta.put("dexSha256", dexHash);
            writeSmallFile(new File(staging, "meta.json"), meta.toString());

            File target = versionDir(version);
            safeDeleteTree(target);
            if (!staging.renameTo(target)) throw new IllegalStateException("runtime slot promotion failed");

            String oldActive = readPointer(new File(rootDir, "active"));
            if (oldActive != null && !oldActive.equals(version)
                    && !isVersionQuarantined(oldActive)
                    && isRuntimeDirValid(versionDir(oldActive), oldActive)) {
                writePointer(new File(rootDir, "previous"), oldActive);
            }
            writePointer(new File(rootDir, "active"), version);
            downloadedVersion = version;
            channelState = "downloaded-restart-required:" + version;
            lastError = "none";
            Log.i(TAG, "runtime " + version + " staged in code_cache; it will load next process start");
        } finally {
            safeDelete(download);
            if (staging.exists()) safeDeleteTree(staging);
        }
    }

    private static void extractBundle(File zip, File staging) throws Exception {
        boolean nativeSeen = false;
        boolean dexSeen = false;
        long total = 0L;
        try (ZipInputStream in = new ZipInputStream(new BufferedInputStream(new FileInputStream(zip)))) {
            ZipEntry entry;
            byte[] buffer = new byte[64 * 1024];
            while ((entry = in.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String name = entry.getName();
                final File out;
                if ("libv240fix.so".equals(name)) {
                    if (nativeSeen) throw new IllegalStateException("duplicate native payload");
                    nativeSeen = true;
                    out = new File(staging, "libv240fix.so");
                } else if ("runtime.dex".equals(name)) {
                    if (dexSeen) throw new IllegalStateException("duplicate dex payload");
                    dexSeen = true;
                    out = new File(staging, "runtime.dex");
                } else {
                    throw new IllegalStateException("unexpected bundle entry: " + name);
                }
                long entryBytes = 0L;
                try (OutputStream output = new BufferedOutputStream(new FileOutputStream(out))) {
                    int count;
                    while ((count = in.read(buffer)) != -1) {
                        entryBytes += count;
                        total += count;
                        if (entryBytes > MAX_ENTRY_BYTES || total > MAX_BUNDLE_BYTES) {
                            throw new IllegalStateException("runtime bundle exceeded extraction bound");
                        }
                        output.write(buffer, 0, count);
                    }
                }
            }
        }
        if (!nativeSeen || !dexSeen) throw new IllegalStateException("runtime bundle missing required payload");
    }

    private static byte[] fetchBytes(URL url, long maxBytes) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        copyUrl(url, out, maxBytes);
        return out.toByteArray();
    }

    private static long downloadFile(URL url, File out, long maxBytes) throws Exception {
        try (OutputStream output = new BufferedOutputStream(new FileOutputStream(out))) {
            return copyUrl(url, output, maxBytes);
        }
    }

    private static long copyUrl(URL initial, OutputStream out, long maxBytes) throws Exception {
        URL current = initial;
        for (int redirect = 0; redirect <= 5; redirect++) {
            requireTrustedHttps(current);
            HttpURLConnection connection = (HttpURLConnection) current.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(7000);
            connection.setReadTimeout(15000);
            connection.setUseCaches(false);
            connection.setRequestProperty("User-Agent", "ADOFAI-V240-Updater/2");
            int code = connection.getResponseCode();
            if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                String location = connection.getHeaderField("Location");
                connection.disconnect();
                if (location == null || location.length() == 0) {
                    throw new IllegalStateException("redirect without location");
                }
                current = new URL(current, location);
                continue;
            }
            if (code != 200) {
                connection.disconnect();
                throw new IllegalStateException("HTTP " + code);
            }
            long declared = connection.getContentLengthLong();
            if (declared > maxBytes) {
                connection.disconnect();
                throw new IllegalStateException("HTTP payload too large");
            }
            long total = 0L;
            try (InputStream in = new BufferedInputStream(connection.getInputStream())) {
                byte[] buffer = new byte[64 * 1024];
                int count;
                while ((count = in.read(buffer)) != -1) {
                    total += count;
                    if (total > maxBytes) throw new IllegalStateException("HTTP payload exceeded bound");
                    out.write(buffer, 0, count);
                }
                out.flush();
                return total;
            } finally {
                connection.disconnect();
            }
        }
        throw new IllegalStateException("too many redirects");
    }

    private static void requireReleaseUrl(URL url) {
        requireTrustedHttps(url);
        if (!"github.com".equalsIgnoreCase(url.getHost()) || !url.getPath().startsWith(RELEASE_PREFIX)) {
            throw new IllegalArgumentException("runtime bundle must come from pinned GitHub release channel");
        }
    }

    private static void requireTrustedHttps(URL url) {
        if (!"https".equalsIgnoreCase(url.getProtocol())) {
            throw new IllegalArgumentException("runtime update requires HTTPS");
        }
        String host = url.getHost().toLowerCase(Locale.US);
        if (!("github.com".equals(host) || host.endsWith(".githubusercontent.com"))) {
            throw new IllegalArgumentException("untrusted runtime update host: " + host);
        }
    }

    private static boolean isRuntimeDirValid(File dir, String version) {
        if (dir == null || !dir.isDirectory() || !isImmutableRuntimeVersion(version)) return false;
        File nativeLib = new File(dir, "libv240fix.so");
        File dex = new File(dir, "runtime.dex");
        File meta = new File(dir, "meta.json");
        return nativeLib.isFile() && nativeLib.length() > 0L
                && dex.isFile() && dex.length() > 0L && meta.isFile() && meta.length() > 0L;
    }

    private static File versionDir(String version) {
        if (!isImmutableRuntimeVersion(version)) throw new IllegalArgumentException("unsafe version");
        return new File(rootDir, "slot-" + version);
    }

    private static boolean isImmutableRuntimeVersion(String version) {
        if (version == null || version.length() != 40) return false;
        for (int i = 0; i < version.length(); i++) {
            char ch = version.charAt(i);
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
        }
        return true;
    }

    private static String normalizedHash(String value) {
        if (value == null) throw new IllegalArgumentException("missing SHA-256");
        String hash = value.trim().toLowerCase(Locale.US);
        if (hash.length() != 64) throw new IllegalArgumentException("invalid SHA-256 length");
        for (int i = 0; i < hash.length(); i++) {
            char ch = hash.charAt(i);
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
                throw new IllegalArgumentException("invalid SHA-256");
            }
        }
        return hash;
    }

    private static void assertHash(File file, String expected) throws Exception {
        expected = normalizedHash(expected);
        String actual = sha256(file);
        if (!expected.equals(actual)) {
            shaVerification = "failed:" + file.getName();
            throw new IllegalStateException("SHA-256 mismatch for " + file.getName());
        }
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[64 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private static JSONObject readJson(File file, int maxBytes) throws Exception {
        if (!file.isFile() || file.length() <= 0L || file.length() > maxBytes) {
            throw new IllegalStateException("invalid metadata file");
        }
        byte[] data = new byte[(int) file.length()];
        int offset = 0;
        try (InputStream in = new FileInputStream(file)) {
            while (offset < data.length) {
                int count = in.read(data, offset, data.length - offset);
                if (count < 0) break;
                offset += count;
            }
        }
        if (offset != data.length) throw new IllegalStateException("short metadata read");
        return new JSONObject(new String(data, "UTF-8"));
    }

    private static String readPointer(File file) {
        try {
            if (!file.isFile() || file.length() <= 0L || file.length() > 128L) return null;
            byte[] data = new byte[(int) file.length()];
            int offset = 0;
            try (InputStream in = new FileInputStream(file)) {
                while (offset < data.length) {
                    int count = in.read(data, offset, data.length - offset);
                    if (count < 0) break;
                    offset += count;
                }
            }
            if (offset != data.length) return null;
            String value = new String(data, "UTF-8").trim();
            return isImmutableRuntimeVersion(value) ? value : null;
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static String readPointerQuiet(File file) {
        String value = readPointer(file);
        return value == null ? "none" : value;
    }

    private static void writePointer(File file, String value) throws Exception {
        if (!isImmutableRuntimeVersion(value)) throw new IllegalArgumentException("unsafe pointer value");
        File temp = new File(rootDir, file.getName() + ".tmp");
        writeSmallFile(temp, value + "\n");
        if (file.exists() && !file.delete()) throw new IllegalStateException("could not replace pointer");
        if (!temp.renameTo(file)) throw new IllegalStateException("pointer promotion failed");
    }

    private static void writeSmallFile(File file, String text) throws Exception {
        File parent = file.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IllegalStateException("parent directory unavailable");
        }
        byte[] data = text.getBytes("UTF-8");
        if (data.length > 64 * 1024) throw new IllegalArgumentException("small file exceeded bound");
        try (FileOutputStream out = new FileOutputStream(file, false)) {
            out.write(data);
            out.flush();
            out.getFD().sync();
        }
    }

    private static void safeDelete(File file) {
        if (file != null && file.exists() && !file.delete()) file.deleteOnExit();
    }

    private static void safeDeleteTree(File target) throws Exception {
        if (target == null || !target.exists()) return;
        String root = rootDir.getCanonicalPath();
        String path = target.getCanonicalPath();
        if (path.equals(root) || !path.startsWith(root + File.separator)) {
            throw new SecurityException("refusing delete outside updater root");
        }
        if (target.isDirectory()) {
            File[] children = target.listFiles();
            if (children != null) for (File child : children) safeDeleteTree(child);
        }
        if (!target.delete()) throw new IllegalStateException("could not delete " + target.getName());
    }

    private static Activity currentActivity() {
        try {
            Class<?> player = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = player.getField("currentActivity");
            Object value = field.get(null);
            return value instanceof Activity ? (Activity) value : null;
        } catch (Throwable error) {
            return null;
        }
    }

    private static String safeMessage(Throwable error) {
        if (error == null) return "unknown";
        String message = error.getMessage();
        if (message == null || message.trim().length() == 0) message = error.getClass().getSimpleName();
        message = message.replace('\n', ' ').replace('\r', ' ');
        return message.length() > 240 ? message.substring(0, 240) : message;
    }
}
