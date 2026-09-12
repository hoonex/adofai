package com.unity3d.player;

import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Conservative compatibility rewrite for charts saved by newer desktop ADOFAI builds.
 *
 * ADOFAI 2.6 changed ToggleBool serialization from the strings "Enabled"/"Disabled" to
 * JSON true/false. The 2.4 editor predates that serializer change. This class backports only
 * fields that are known to have been ToggleBool in 2.4, and only inside the app-private working
 * copy. Unknown fields and modern events are preserved byte-for-byte.
 */
final class V240ChartBackport {
    private static final String TAG = "ADOFAI.V240Backport";
    private static final long MAX_BYTES = 96L * 1024L * 1024L;

    // Settings that were serialized as ToggleBool strings in pre-2.6 charts.
    private static final Set<String> SETTINGS_TOGGLE_BOOL = new HashSet<String>(Arrays.asList(
            "separateCountdownTime",
            "seizureWarning",
            "showDefaultBGIfNoImage",
            "lockRot",
            "loopBG",
            "pulseOnFloor",
            "startCamLowVFX",
            "loopVideo",
            "floorIconOutlines",
            "stickToFloors"
    ));

    // Event fields whose old 2.4 representation was ToggleBool and whose names are specific
    // enough to convert without changing unrelated legacy JSON booleans.
    private static final Set<String> EVENT_TOGGLE_BOOL = new HashSet<String>(Arrays.asList(
            "disableOthers",
            "dontDisable",
            "minVfxOnly",
            "justThisTile",
            "editorOnly",
            "maxVfxOnly"
    ));

    private static final Pattern EVENT_TYPE = Pattern.compile(
            "\\\"eventType\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");

    private V240ChartBackport() {}

    static boolean backportForV240(File chart) {
        if (chart == null || !chart.isFile()) return false;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES || length > Integer.MAX_VALUE) {
            Log.w(TAG, "Skipping oversized chart backport: " + length);
            return false;
        }

        try {
            byte[] bytes = readFully(chart, (int) length);
            int bom = hasUtf8Bom(bytes) ? 3 : 0;
            String source = new String(bytes, bom, bytes.length - bom, StandardCharsets.UTF_8);
            String rewritten = rewriteDocument(source);
            if (source.equals(rewritten)) return false;

            File parent = chart.getParentFile();
            if (parent == null) throw new IOException("chart has no parent directory");
            File temp = new File(parent, chart.getName() + ".v240-backport.tmp");
            try (FileOutputStream out = new FileOutputStream(temp, false)) {
                if (bom != 0) out.write(new byte[] {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
                out.write(rewritten.getBytes(StandardCharsets.UTF_8));
                out.flush();
                out.getFD().sync();
            }
            if (chart.exists() && !chart.delete()) {
                temp.delete();
                throw new IOException("could not replace private chart copy");
            }
            if (!temp.renameTo(chart)) {
                copyFile(temp, chart);
                temp.delete();
            }
            Log.d(TAG, "Applied conservative 2.4 ToggleBool backport: " + chart.getName());
            return true;
        } catch (Throwable error) {
            // A compatibility pass must never make a chart that otherwise loaded become unusable.
            Log.w(TAG, "Chart backport failed open", error);
            return false;
        }
    }

    static String rewriteDocument(String source) {
        if (source == null || source.length() == 0) return source == null ? "" : source;
        StringBuilder out = new StringBuilder(source.length() + 128);
        int cursor = 0;
        while (cursor < source.length()) {
            int start = nextObjectStart(source, cursor);
            if (start < 0) {
                out.append(source, cursor, source.length());
                break;
            }
            out.append(source, cursor, start);
            int end = matchingObjectEnd(source, start);
            if (end < 0) {
                out.append(source, start, source.length());
                break;
            }
            String object = source.substring(start, end + 1);
            out.append(rewriteObject(object));
            cursor = end + 1;
        }
        return out.toString();
    }

    private static String rewriteObject(String object) {
        Matcher eventMatcher = EVENT_TYPE.matcher(object);
        if (eventMatcher.find()) {
            String eventType = eventMatcher.group(1);
            String result = object;
            for (String field : EVENT_TOGGLE_BOOL) result = rewriteBooleanField(result, field);
            if ("SetFilter".equals(eventType)) result = rewriteBooleanField(result, "enabled");
            return result;
        }

        // Only treat an object as settings when it contains several characteristic settings keys.
        // This prevents a same-named field inside arbitrary modern event payloads from being touched.
        int markers = 0;
        if (object.indexOf("\"songFilename\"") >= 0) markers++;
        if (object.indexOf("\"bpm\"") >= 0) markers++;
        if (object.indexOf("\"backgroundColor\"") >= 0) markers++;
        if (object.indexOf("\"trackStyle\"") >= 0) markers++;
        if (markers < 2) return object;

        String result = object;
        for (String field : SETTINGS_TOGGLE_BOOL) result = rewriteBooleanField(result, field);
        return result;
    }

    private static String rewriteBooleanField(String input, String field) {
        Pattern pattern = Pattern.compile(
                "(\\\"" + Pattern.quote(field) + "\\\"\\s*:\\s*)(true|false)(?=\\s*[,}])",
                Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(input);
        StringBuffer buffer = new StringBuffer(input.length() + 32);
        boolean changed = false;
        while (matcher.find()) {
            String legacy = "true".equalsIgnoreCase(matcher.group(2)) ? "\"Enabled\"" : "\"Disabled\"";
            matcher.appendReplacement(buffer, Matcher.quoteReplacement(matcher.group(1) + legacy));
            changed = true;
        }
        if (!changed) return input;
        matcher.appendTail(buffer);
        return buffer.toString();
    }

    private static int nextObjectStart(String text, int from) {
        boolean inString = false;
        boolean escaped = false;
        for (int i = Math.max(0, from); i < text.length(); ++i) {
            char ch = text.charAt(i);
            if (inString) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') inString = false;
            } else {
                if (ch == '"') inString = true;
                else if (ch == '{') return i;
            }
        }
        return -1;
    }

    private static int matchingObjectEnd(String text, int start) {
        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int i = start; i < text.length(); ++i) {
            char ch = text.charAt(i);
            if (inString) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') inString = false;
                continue;
            }
            if (ch == '"') inString = true;
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) return i;
        }
        return -1;
    }

    private static byte[] readFully(File file, int expected) throws IOException {
        byte[] data = new byte[expected];
        int offset = 0;
        try (FileInputStream in = new FileInputStream(file)) {
            while (offset < data.length) {
                int count = in.read(data, offset, data.length - offset);
                if (count < 0) break;
                if (count > 0) offset += count;
            }
            if (offset == data.length && in.read() < 0) return data;
        }
        if (offset == data.length) return data;
        return Arrays.copyOf(data, offset);
    }

    private static boolean hasUtf8Bom(byte[] bytes) {
        return bytes.length >= 3 && (bytes[0] & 0xFF) == 0xEF &&
                (bytes[1] & 0xFF) == 0xBB && (bytes[2] & 0xFF) == 0xBF;
    }

    private static void copyFile(File from, File to) throws IOException {
        byte[] buffer = new byte[256 * 1024];
        try (FileInputStream in = new FileInputStream(from);
             FileOutputStream out = new FileOutputStream(to, false)) {
            int count;
            while ((count = in.read(buffer)) != -1) if (count > 0) out.write(buffer, 0, count);
            out.flush();
            out.getFD().sync();
        }
    }
}
