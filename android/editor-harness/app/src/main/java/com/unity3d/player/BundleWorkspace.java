package com.unity3d.player;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Keeps a custom level as a bundle, not just as a detached .adofai file.
 *
 * Old ADOFAI "Open From URL" levels are ZIP archives containing main.adofai
 * plus sibling audio/images. The companion therefore extracts the complete
 * archive into an app-private workspace and edits main.adofai in place.
 */
public final class BundleWorkspace {
    private static final int MAX_ENTRIES = 10000;
    private static final long MAX_DOWNLOAD_BYTES = 256L * 1024L * 1024L;
    private static final long MAX_EXTRACTED_BYTES = 1024L * 1024L * 1024L;
    private static final Map<String, String> ROOT_BY_CHART = new ConcurrentHashMap<String, String>();
    private static final Map<String, String> SOURCE_URL_BY_CHART = new ConcurrentHashMap<String, String>();

    private BundleWorkspace() {}

    public static boolean isBundlePath(String chartPath) {
        return chartPath != null && ROOT_BY_CHART.containsKey(chartPath);
    }

    /** Returns the original HTTPS ZIP URL for a URL-imported chart, otherwise null. */
    public static String sourceHttpsUrlForChart(String chartPath) {
        if (chartPath == null) return null;
        String value = SOURCE_URL_BY_CHART.get(chartPath);
        return value != null && value.startsWith("https://") ? value : null;
    }

    public static String importZip(Context context, InputStream source, String displayName) throws Exception {
        File root = new File(new File(context.getFilesDir(), "working"), "bundle-" + UUID.randomUUID().toString());
        if (!root.mkdirs() && !root.isDirectory()) throw new IOException("Could not create bundle workspace");

        long extracted = 0L;
        int entries = 0;
        ZipInputStream zip = new ZipInputStream(new BufferedInputStream(source));
        try {
            ZipEntry entry;
            byte[] buffer = new byte[64 * 1024];
            while ((entry = zip.getNextEntry()) != null) {
                if (++entries > MAX_ENTRIES) throw new IOException("ZIP contains too many entries");
                String name = entry.getName();
                if (name == null || name.length() == 0) continue;
                File target = safeChild(root, name);
                if (entry.isDirectory()) {
                    if (!target.mkdirs() && !target.isDirectory()) throw new IOException("Could not create ZIP directory");
                    continue;
                }
                File parent = target.getParentFile();
                if (parent != null && !parent.mkdirs() && !parent.isDirectory()) throw new IOException("Could not create ZIP parent directory");
                FileOutputStream output = new FileOutputStream(target, false);
                try {
                    int read;
                    while ((read = zip.read(buffer)) >= 0) {
                        if (read == 0) continue;
                        extracted += read;
                        if (extracted > MAX_EXTRACTED_BYTES) throw new IOException("ZIP expands beyond safety limit");
                        output.write(buffer, 0, read);
                    }
                    output.getFD().sync();
                } finally {
                    output.close();
                }
            }
        } finally {
            zip.close();
        }

        File chart = findChart(root);
        String chartPath = chart.getCanonicalPath();
        ROOT_BY_CHART.put(chartPath, root.getCanonicalPath());
        return chartPath;
    }

    public static String importZipUrl(Context context, String encodedUrl) throws Exception {
        String value = encodedUrl == null ? "" : encodedUrl.trim();
        if (!(value.startsWith("https://") || value.startsWith("http://"))) {
            throw new IllegalArgumentException("ZIP URL must start with https:// or http://");
        }

        File cache = new File(context.getCacheDir(), "url-import");
        if (!cache.mkdirs() && !cache.isDirectory()) throw new IOException("Could not create URL cache");
        File downloaded = new File(cache, "level-" + UUID.randomUUID().toString() + ".zip");
        download(value, downloaded);
        try {
            FileInputStream input = new FileInputStream(downloaded);
            try {
                String chartPath = importZip(context, input, nameFromUrl(value));
                SOURCE_URL_BY_CHART.put(chartPath, value);
                return chartPath;
            } finally {
                input.close();
            }
        } finally {
            // Extraction is complete; the original network copy is no longer needed.
            //noinspection ResultOfMethodCallIgnored
            downloaded.delete();
        }
    }

    public static File packageBundle(Context context, String chartPath) throws Exception {
        File chart = new File(chartPath).getCanonicalFile();
        if (!chart.isFile()) throw new IOException("Chart does not exist");

        String rootPath = ROOT_BY_CHART.get(chart.getAbsolutePath());
        File root;
        boolean standalone = rootPath == null;
        if (standalone) {
            root = new File(new File(context.getCacheDir(), "handoff-stage"), "bundle-" + UUID.randomUUID().toString());
            if (!root.mkdirs() && !root.isDirectory()) throw new IOException("Could not create handoff staging directory");
            copyFile(chart, new File(root, "main.adofai"));
        } else {
            root = new File(rootPath).getCanonicalFile();
            String prefix = root.getPath() + File.separator;
            if (!chart.getPath().startsWith(prefix)) throw new SecurityException("Chart escaped bundle workspace");
        }

        File outDir = new File(context.getCacheDir(), "handoff-zips");
        if (!outDir.mkdirs() && !outDir.isDirectory()) throw new IOException("Could not create handoff ZIP directory");
        File output = new File(outDir, "level-" + UUID.randomUUID().toString() + ".zip");
        zipDirectory(root, output);
        return output;
    }

    private static void download(String initialUrl, File output) throws Exception {
        String current = initialUrl;
        for (int redirect = 0; redirect < 6; redirect++) {
            HttpURLConnection connection = (HttpURLConnection) new URL(current).openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(15000);
            connection.setReadTimeout(30000);
            connection.setRequestProperty("User-Agent", "ADOFAI-Companion/1.0");
            connection.connect();
            try {
                int code = connection.getResponseCode();
                if (code >= 300 && code < 400) {
                    String location = connection.getHeaderField("Location");
                    if (location == null) throw new IOException("Redirect without Location");
                    current = new URL(new URL(current), location).toString();
                    continue;
                }
                if (code < 200 || code >= 300) throw new IOException("HTTP " + code + " while downloading ZIP");
                long declared = connection.getContentLengthLong();
                if (declared > MAX_DOWNLOAD_BYTES) throw new IOException("ZIP download exceeds safety limit");
                InputStream input = new BufferedInputStream(connection.getInputStream());
                FileOutputStream fileOutput = new FileOutputStream(output, false);
                try {
                    byte[] buffer = new byte[64 * 1024];
                    long total = 0L;
                    int read;
                    while ((read = input.read(buffer)) >= 0) {
                        if (read == 0) continue;
                        total += read;
                        if (total > MAX_DOWNLOAD_BYTES) throw new IOException("ZIP download exceeds safety limit");
                        fileOutput.write(buffer, 0, read);
                    }
                    fileOutput.getFD().sync();
                } finally {
                    try { input.close(); } finally { fileOutput.close(); }
                }
                return;
            } finally {
                connection.disconnect();
            }
        }
        throw new IOException("Too many redirects while downloading ZIP");
    }

    private static File findChart(File root) throws Exception {
        List<File> mains = new ArrayList<File>();
        List<File> charts = new ArrayList<File>();
        collectCharts(root, mains, charts);
        if (mains.size() == 1) return mains.get(0);
        if (mains.size() > 1) throw new IOException("ZIP contains multiple main.adofai files");
        if (charts.size() == 1) return charts.get(0);
        if (charts.isEmpty()) throw new IOException("ZIP does not contain an .adofai chart");
        throw new IOException("ZIP contains multiple charts but no unique main.adofai");
    }

    private static void collectCharts(File file, List<File> mains, List<File> charts) {
        if (file.isFile()) {
            String name = file.getName().toLowerCase(Locale.US);
            if (name.endsWith(".adofai")) {
                charts.add(file);
                if (name.equals("main.adofai")) mains.add(file);
            }
            return;
        }
        File[] children = file.listFiles();
        if (children == null) return;
        for (File child : children) collectCharts(child, mains, charts);
    }

    private static File safeChild(File root, String relative) throws Exception {
        File target = new File(root, relative).getCanonicalFile();
        File canonicalRoot = root.getCanonicalFile();
        String prefix = canonicalRoot.getPath() + File.separator;
        if (!target.getPath().startsWith(prefix)) throw new SecurityException("ZIP path traversal rejected: " + relative);
        return target;
    }

    private static void zipDirectory(File root, File output) throws Exception {
        List<File> files = new ArrayList<File>();
        collectFiles(root, files);
        Collections.sort(files, new java.util.Comparator<File>() {
            @Override public int compare(File left, File right) {
                return left.getAbsolutePath().compareTo(right.getAbsolutePath());
            }
        });
        String prefix = root.getCanonicalPath() + File.separator;
        ZipOutputStream zip = new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(output, false)));
        try {
            byte[] buffer = new byte[64 * 1024];
            for (File file : files) {
                String relative = file.getCanonicalPath().substring(prefix.length()).replace(File.separatorChar, '/');
                zip.putNextEntry(new ZipEntry(relative));
                FileInputStream input = new FileInputStream(file);
                try {
                    int read;
                    while ((read = input.read(buffer)) >= 0) if (read > 0) zip.write(buffer, 0, read);
                } finally {
                    input.close();
                }
                zip.closeEntry();
            }
        } finally {
            zip.close();
        }
    }

    private static void collectFiles(File file, List<File> out) {
        if (file.isFile()) { out.add(file); return; }
        File[] children = file.listFiles();
        if (children == null) return;
        for (File child : children) collectFiles(child, out);
    }

    private static void copyFile(File source, File target) throws Exception {
        FileInputStream input = new FileInputStream(source);
        FileOutputStream output = new FileOutputStream(target, false);
        try {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) if (read > 0) output.write(buffer, 0, read);
            output.getFD().sync();
        } finally {
            try { input.close(); } finally { output.close(); }
        }
    }

    private static String nameFromUrl(String value) {
        try {
            String path = new URL(value).getPath();
            int slash = path.lastIndexOf('/');
            String name = slash >= 0 ? path.substring(slash + 1) : path;
            return name.length() == 0 ? "level.zip" : name;
        } catch (Throwable ignored) {
            return "level.zip";
        }
    }
}
