package com.unity3d.player;

import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
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

    // Generic `enabled` is deliberately excluded and handled only for SetFilter below.
    private static final Set<String> EVENT_TOGGLE_BOOL = new HashSet<String>(Arrays.asList(
            "disableOthers",
            "dontDisable",
            "minVfxOnly",
            "justThisTile",
            "editorOnly",
            "maxVfxOnly"
    ));

    private static final Pattern SET_FILTER_OBJECT = Pattern.compile(
            "\\{(?=[^{}]*\\\"eventType\\\"\\s*:\\s*\\\"SetFilter\\\")[^{}]*\\}",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL);

    private V240ChartBackport() {}

    static boolean backportForV240(File chart) {
        if (chart == null || !chart.isFile()) return false;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES || length > Integer.MAX_VALUE) {
            Log.w(TAG, "Skipping oversized chart backport: " + length);
            return false;
        }

        File temp = null;
        File backup = null;
        try {
            byte[] bytes = readFully(chart, (int) length);
            int bom = hasUtf8Bom(bytes) ? 3 : 0;
            String source = new String(bytes, bom, bytes.length - bom, StandardCharsets.UTF_8);
            String rewritten = rewriteDocument(source);
            if (source.equals(rewritten)) return false;

            File parent = chart.getParentFile();
            if (parent == null) throw new IOException("chart has no parent directory");
            temp = new File(parent, chart.getName() + ".v240-backport.tmp");
            backup = new File(parent, chart.getName() + ".v240-backport.original");
            if (temp.exists() && !temp.delete()) throw new IOException("stale backport temp is locked");
            if (backup.exists() && !backup.delete()) throw new IOException("stale backport backup is locked");

            try (FileOutputStream out = new FileOutputStream(temp, false)) {
                if (bom != 0) out.write(new byte[] {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
                out.write(rewritten.getBytes(StandardCharsets.UTF_8));
                out.flush();
                out.getFD().sync();
            }

            // Same-directory renames keep the original intact until the replacement is complete.
            if (!chart.renameTo(backup)) throw new IOException("could not stage original chart");
            if (!temp.renameTo(chart)) {
                if (!backup.renameTo(chart)) {
                    Log.e(TAG, "Could not restore private chart after failed backport rename");
                }
                throw new IOException("could not install backported chart");
            }
            if (backup.exists() && !backup.delete()) backup.deleteOnExit();
            Log.d(TAG, "Applied conservative 2.4 ToggleBool backport: " + chart.getName());
            return true;
        } catch (Throwable error) {
            if (chart != null && !chart.exists() && backup != null && backup.exists()) {
                if (!backup.renameTo(chart)) Log.e(TAG, "Backport recovery failed for " + chart, error);
            }
            Log.w(TAG, "Chart backport failed open", error);
            return false;
        } finally {
            if (temp != null && temp.exists()) temp.delete();
        }
    }

    static String rewriteDocument(String source) {
        if (source == null || source.length() == 0) return source == null ? "" : source;
        String result = source;

        // These field names are evidence-backed pre-2.6 ToggleBool properties and are specific
        // enough not to collide with the legacy JSON booleans that must stay real booleans.
        for (String field : SETTINGS_TOGGLE_BOOL) result = rewriteBooleanField(result, field);
        for (String field : EVENT_TOGGLE_BOOL) result = rewriteBooleanField(result, field);

        // `enabled` is generic, so only convert it inside a flat SetFilter action object.
        Matcher matcher = SET_FILTER_OBJECT.matcher(result);
        StringBuffer out = new StringBuffer(result.length() + 64);
        boolean changed = false;
        while (matcher.find()) {
            String object = matcher.group();
            String rewritten = rewriteBooleanField(object, "enabled");
            if (!object.equals(rewritten)) changed = true;
            matcher.appendReplacement(out, Matcher.quoteReplacement(rewritten));
        }
        if (!changed) return result;
        matcher.appendTail(out);
        return out.toString();
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

    private static byte[] readFully(File file, int expected) throws IOException {
        byte[] data = new byte[expected];
        int offset = 0;
        try (FileInputStream in = new FileInputStream(file)) {
            while (offset < data.length) {
                int count = in.read(data, offset, data.length - offset);
                if (count < 0) break;
                if (count > 0) offset += count;
            }
        }
        return offset == data.length ? data : Arrays.copyOf(data, offset);
    }

    private static boolean hasUtf8Bom(byte[] bytes) {
        return bytes.length >= 3 && (bytes[0] & 0xFF) == 0xEF &&
                (bytes[1] & 0xFF) == 0xBB && (bytes[2] & 0xFF) == 0xBF;
    }
}
