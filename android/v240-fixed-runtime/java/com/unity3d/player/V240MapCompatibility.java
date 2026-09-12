package com.unity3d.player;

import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.text.Normalizer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Bridges common Windows-created custom-map path assumptions to Android's case-sensitive FS.
 * It never rewrites the chart. Missing referenced assets get local aliases inside the private
 * mirror only, leaving the source SAF folder/archive untouched.
 */
final class V240MapCompatibility {
    private static final String TAG = "ADOFAI.V240MapCompat";
    private static final long MAX_CHART_SCAN_BYTES = 96L * 1024L * 1024L;
    private static final int MAX_FILES = 12000;
    private static final int MAX_ALIASES = 1024;
    private static final int BUFFER_BYTES = 256 * 1024;

    // Match quoted relative-looking asset strings anywhere in settings/actions. ADOFAI is JSON;
    // this intentionally avoids parsing the whole giant action graph just to repair filesystem
    // semantics. Escaped quotes are excluded and the extension allowlist keeps false positives low.
    private static final Pattern ASSET = Pattern.compile(
            "\\\"((?:[^\\\"\\\\]|\\\\.)+\\.(?:ogg|mp3|wav|flac|aac|m4a|png|jpe?g|gif|webp|bmp|mp4|webm|mov))\\\"",
            Pattern.CASE_INSENSITIVE);

    private V240MapCompatibility() {}

    static void repairMap(File chart) {
        if (chart == null || !chart.isFile()) return;
        File root = chart.getParentFile();
        if (root == null || !root.isDirectory()) return;
        if (chart.length() < 0L || chart.length() > MAX_CHART_SCAN_BYTES) {
            Log.w(TAG, "Skipping oversized chart compatibility scan: " + chart.length());
            return;
        }

        try {
            FileIndex index = new FileIndex(root);
            index.scan(root, "", 0);
            Set<String> references = readAssetReferences(chart);
            int aliases = 0;
            for (String raw : references) {
                if (aliases >= MAX_ALIASES) break;
                String relative = decodeJsonPath(raw).replace('\\', '/');
                relative = stripRelativePrefix(relative);
                if (relative.length() == 0 || isUnsafe(relative)) continue;

                File exact = safeResolve(root, relative);
                if (exact == null || exact.isFile()) continue;
                File source = index.resolve(relative);
                if (source == null || !source.isFile()) continue;
                if (sameCanonical(source, exact)) continue;

                File parent = exact.getParentFile();
                if (parent != null && !parent.mkdirs() && !parent.isDirectory()) continue;
                if (copyAlias(source, exact)) {
                    aliases++;
                    Log.d(TAG, "Created Android asset alias: " + relative + " <- " +
                            relativePath(root, source));
                }
            }
            Log.d(TAG, "Map compatibility pass refs=" + references.size() + " aliases=" + aliases);
        } catch (Throwable error) {
            // Compatibility repair must never prevent an otherwise valid chart from loading.
            Log.w(TAG, "Map compatibility pass failed open", error);
        }
    }

    private static final class FileIndex {
        final File root;
        final Map<String, File> normalized = new HashMap<String, File>();
        int count;

        FileIndex(File root) { this.root = root; }

        void scan(File dir, String prefix, int depth) throws IOException {
            if (depth > 64 || count > MAX_FILES) return;
            File[] children = dir.listFiles();
            if (children == null) return;
            for (File child : children) {
                if (++count > MAX_FILES) return;
                String rel = prefix.length() == 0 ? child.getName() : prefix + "/" + child.getName();
                if (child.isDirectory()) {
                    scan(child, rel, depth + 1);
                } else if (child.isFile()) {
                    String key = normalizedKey(rel);
                    // Ambiguous normalized names are deliberately disabled rather than guessed.
                    File previous = normalized.get(key);
                    if (previous == null) normalized.put(key, child);
                    else if (!sameCanonical(previous, child)) normalized.put(key, null);
                }
            }
        }

        File resolve(String relative) {
            String key = normalizedKey(relative);
            return normalized.containsKey(key) ? normalized.get(key) : null;
        }
    }

    private static Set<String> readAssetReferences(File chart) throws IOException {
        Set<String> out = new HashSet<String>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(chart), "UTF-8"), 64 * 1024);
        try {
            String line;
            while ((line = reader.readLine()) != null && out.size() < MAX_ALIASES * 4) {
                Matcher matcher = ASSET.matcher(line);
                while (matcher.find() && out.size() < MAX_ALIASES * 4) {
                    String value = matcher.group(1);
                    if (value != null && value.length() > 0) out.add(value);
                }
            }
        } finally {
            reader.close();
        }
        return out;
    }

    private static String normalizedKey(String value) {
        String path = value == null ? "" : value.replace('\\', '/');
        while (path.startsWith("./")) path = path.substring(2);
        path = Normalizer.normalize(path, Normalizer.Form.NFC);
        return path.toLowerCase(Locale.US);
    }

    /** Decode JSON string escapes without changing the chart itself. */
    private static String decodeJsonPath(String raw) {
        if (raw == null || raw.indexOf('\\') < 0) return raw == null ? "" : raw;
        StringBuilder out = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); ++i) {
            char ch = raw.charAt(i);
            if (ch != '\\' || i + 1 >= raw.length()) {
                out.append(ch);
                continue;
            }
            int slash = i;
            char next = raw.charAt(++i);
            if (next == '\\' || next == '/' || next == '"') {
                out.append(next);
            } else if (next == 'b') {
                out.append('\b');
            } else if (next == 'f') {
                out.append('\f');
            } else if (next == 'n') {
                out.append('\n');
            } else if (next == 'r') {
                out.append('\r');
            } else if (next == 't') {
                out.append('\t');
            } else if (next == 'u') {
                int first = parseHex4(raw, i + 1);
                if (first < 0) {
                    out.append(raw, slash, Math.min(raw.length(), i + 1));
                    continue;
                }
                i += 4;
                char firstChar = (char) first;
                // Preserve valid UTF-16 surrogate pairs as one code point. Lone surrogates are
                // kept as their literal JSON escape rather than corrupting a filename.
                if (Character.isHighSurrogate(firstChar) && i + 6 < raw.length() &&
                        raw.charAt(i + 1) == '\\' && raw.charAt(i + 2) == 'u') {
                    int second = parseHex4(raw, i + 3);
                    if (second >= 0 && Character.isLowSurrogate((char) second)) {
                        out.appendCodePoint(Character.toCodePoint(firstChar, (char) second));
                        i += 6;
                    } else {
                        out.append(raw, slash, i + 1);
                    }
                } else if (Character.isSurrogate(firstChar)) {
                    out.append(raw, slash, i + 1);
                } else {
                    out.append(firstChar);
                }
            } else {
                // Preserve unknown escapes rather than silently corrupting a filename.
                out.append('\\').append(next);
            }
        }
        return out.toString();
    }

    private static int parseHex4(String value, int offset) {
        if (offset < 0 || offset + 4 > value.length()) return -1;
        int result = 0;
        for (int i = 0; i < 4; ++i) {
            int digit = Character.digit(value.charAt(offset + i), 16);
            if (digit < 0) return -1;
            result = (result << 4) | digit;
        }
        return result;
    }

    private static String stripRelativePrefix(String value) {
        String out = value == null ? "" : value.trim();
        while (out.startsWith("./")) out = out.substring(2);
        return out;
    }

    private static boolean isUnsafe(String relative) {
        if (relative == null || relative.length() == 0) return true;
        String slash = relative.replace('\\', '/');
        if (slash.startsWith("/") || slash.indexOf('\u0000') >= 0) return true;
        // java.io.File on Android does not classify a Windows drive path as absolute.
        if (slash.length() >= 3 && Character.isLetter(slash.charAt(0)) &&
                slash.charAt(1) == ':' && slash.charAt(2) == '/') return true;
        String lower = slash.toLowerCase(Locale.US);
        if (lower.startsWith("file:") || lower.startsWith("content:") ||
                lower.startsWith("http:") || lower.startsWith("https:")) return true;
        String[] parts = slash.split("/");
        for (String part : parts) if ("..".equals(part)) return true;
        return false;
    }

    private static File safeResolve(File root, String relative) throws IOException {
        File target = new File(root, relative);
        String rootPath = root.getCanonicalPath();
        String targetPath = target.getCanonicalPath();
        if (targetPath.equals(rootPath) || targetPath.startsWith(rootPath + File.separator)) {
            return target;
        }
        return null;
    }

    private static boolean copyAlias(File source, File target) {
        if (target.exists()) return target.isFile();
        File temporary = new File(target.getParentFile(), target.getName() + ".v240tmp");
        byte[] buffer = new byte[BUFFER_BYTES];
        try (FileInputStream in = new FileInputStream(source);
             FileOutputStream out = new FileOutputStream(temporary, false)) {
            int count;
            while ((count = in.read(buffer)) != -1) {
                if (count > 0) out.write(buffer, 0, count);
            }
            out.flush();
            out.getFD().sync();
        } catch (Throwable error) {
            if (temporary.exists()) temporary.delete();
            return false;
        }
        if (target.exists()) {
            temporary.delete();
            return target.isFile();
        }
        if (temporary.renameTo(target)) return true;
        // renameTo can fail across odd app-private storage implementations; fall back to copy.
        try (FileInputStream in = new FileInputStream(temporary);
             FileOutputStream out = new FileOutputStream(target, false)) {
            int count;
            while ((count = in.read(buffer)) != -1) if (count > 0) out.write(buffer, 0, count);
            out.flush();
            out.getFD().sync();
            temporary.delete();
            return true;
        } catch (Throwable error) {
            temporary.delete();
            if (target.exists()) target.delete();
            return false;
        }
    }

    private static boolean sameCanonical(File a, File b) {
        try { return a.getCanonicalFile().equals(b.getCanonicalFile()); }
        catch (Throwable ignored) { return a.equals(b); }
    }

    private static String relativePath(File root, File file) {
        try {
            String base = root.getCanonicalPath();
            String full = file.getCanonicalPath();
            if (full.startsWith(base + File.separator)) {
                return full.substring(base.length() + 1).replace(File.separatorChar, '/');
            }
        } catch (Throwable ignored) {}
        return file.getName();
    }
}
