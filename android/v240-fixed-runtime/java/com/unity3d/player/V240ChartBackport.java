package com.unity3d.player;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PushbackReader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

/**
 * Conservative compatibility rewrite for charts saved by newer desktop ADOFAI builds.
 *
 * ADOFAI 2.6 changed ToggleBool serialization from the strings "Enabled"/"Disabled" to
 * JSON true/false. The 2.4 editor predates that serializer change. This class backports only
 * fields that are known to have been ToggleBool in 2.4, and only inside the app-private working
 * copy. Unknown fields and modern events are preserved byte-for-byte.
 *
 * The transformer is streaming: even a very large decoration-heavy chart does not require a
 * second full in-memory copy of the JSON. Only one JSON string token is buffered at a time.
 */
final class V240ChartBackport {
    private static final String TAG = "ADOFAI.V240Backport";
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int IO_BUFFER_BYTES = 64 * 1024;
    private static final int MAX_STRING_TOKEN_CHARS = 16 * 1024 * 1024;

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

    private V240ChartBackport() {}

    static boolean backportForV240(File chart) {
        if (chart == null || !chart.isFile()) return false;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES) {
            Log.w(TAG, "Skipping oversized chart backport: " + length);
            return false;
        }

        File parent = chart.getParentFile();
        if (parent == null) return false;
        File temp = new File(parent, chart.getName() + ".v240-backport.tmp");
        File backup = new File(parent, chart.getName() + ".v240-backport.original");

        try {
            if (temp.exists() && !temp.delete()) throw new IOException("stale backport temp is locked");
            if (backup.exists() && !backup.delete()) throw new IOException("stale backport backup is locked");

            int changes = transformFile(chart, temp);
            if (changes == 0) {
                if (temp.exists()) temp.delete();
                return false;
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
            Log.d(TAG, "Applied streaming 2.4 ToggleBool backport changes=" + changes +
                    " file=" + chart.getName());
            return true;
        } catch (Throwable error) {
            if (!chart.exists() && backup.exists()) {
                if (!backup.renameTo(chart)) Log.e(TAG, "Backport recovery failed for " + chart, error);
            }
            Log.w(TAG, "Chart backport failed open", error);
            return false;
        } finally {
            if (temp.exists()) temp.delete();
        }
    }

    private static int transformFile(File source, File target) throws IOException {
        BufferedInputStream input = new BufferedInputStream(new FileInputStream(source), IO_BUFFER_BYTES);
        BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(target, false), IO_BUFFER_BYTES);
        boolean bom = false;
        try {
            input.mark(3);
            int b0 = input.read();
            int b1 = input.read();
            int b2 = input.read();
            bom = b0 == 0xEF && b1 == 0xBB && b2 == 0xBF;
            if (!bom) input.reset();
            else output.write(new byte[] {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});

            PushbackReader reader = new PushbackReader(
                    new InputStreamReader(input, StandardCharsets.UTF_8), 8);
            Writer writer = new OutputStreamWriter(output, StandardCharsets.UTF_8);
            int changes = transformJson(reader, writer);
            writer.flush();
            output.flush();
            outputFileSync(target);
            return changes;
        } finally {
            try { input.close(); } finally { output.close(); }
        }
    }

    /** Streaming JSON-token transform. Formatting and all non-target values are preserved. */
    private static int transformJson(PushbackReader reader, Writer writer) throws IOException {
        Deque<ObjectContext> objects = new ArrayDeque<ObjectContext>();
        int changes = 0;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            if (ch == '{') {
                objects.push(new ObjectContext());
                writer.write(ch);
                continue;
            }
            if (ch == '}') {
                writer.write(ch);
                if (!objects.isEmpty()) objects.pop();
                continue;
            }
            if (ch != '"') {
                writer.write(ch);
                continue;
            }

            String token = readStringToken(reader);
            writer.write(token);

            StringBuilder whitespace = new StringBuilder(8);
            int next = readAfterWhitespace(reader, whitespace);
            writer.write(whitespace.toString());
            if (next != ':') {
                if (next != -1) reader.unread(next);
                continue;
            }

            // This string token is a JSON object key.
            writer.write(':');
            String key = decodeJsonStringToken(token);
            whitespace.setLength(0);
            int first = readAfterWhitespace(reader, whitespace);
            writer.write(whitespace.toString());
            if (first == -1) break;

            ObjectContext context = objects.peek();
            if ("eventType".equals(key) && first == '"') {
                String eventToken = readStringToken(reader);
                writer.write(eventToken);
                if (context != null) context.eventType = decodeJsonStringToken(eventToken);
                continue;
            }

            boolean target = SETTINGS_TOGGLE_BOOL.contains(key) || EVENT_TOGGLE_BOOL.contains(key) ||
                    ("enabled".equals(key) && context != null && "SetFilter".equals(context.eventType));
            if (target && (first == 't' || first == 'f')) {
                BooleanLiteral literal = readBooleanLiteral(reader, first);
                if (literal.valid) {
                    writer.write(literal.value ? "\"Enabled\"" : "\"Disabled\"");
                    changes++;
                } else {
                    writer.write(literal.raw);
                }
                continue;
            }

            reader.unread(first);
        }
        return changes;
    }

    /** Read a raw JSON string token after its opening quote has already been consumed. */
    private static String readStringToken(PushbackReader reader) throws IOException {
        StringBuilder token = new StringBuilder(64);
        token.append('"');
        boolean escaped = false;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            token.append(ch);
            if (token.length() > MAX_STRING_TOKEN_CHARS) {
                throw new IOException("JSON string token exceeds compatibility safety limit");
            }
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                return token.toString();
            }
        }
        throw new IOException("unterminated JSON string");
    }

    private static int readAfterWhitespace(PushbackReader reader, StringBuilder whitespace) throws IOException {
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') whitespace.append(ch);
            else return value;
        }
        return -1;
    }

    private static final class ObjectContext {
        String eventType;
    }

    private static final class BooleanLiteral {
        final boolean valid;
        final boolean value;
        final String raw;
        BooleanLiteral(boolean valid, boolean value, String raw) {
            this.valid = valid;
            this.value = value;
            this.raw = raw;
        }
    }

    private static BooleanLiteral readBooleanLiteral(PushbackReader reader, int first) throws IOException {
        String expected = first == 't' ? "true" : "false";
        StringBuilder raw = new StringBuilder(expected.length());
        raw.append((char) first);
        for (int i = 1; i < expected.length(); ++i) {
            int value = reader.read();
            if (value == -1) return new BooleanLiteral(false, false, raw.toString());
            raw.append((char) value);
        }
        if (!expected.equals(raw.toString())) return new BooleanLiteral(false, false, raw.toString());

        int boundary = reader.read();
        if (boundary != -1) reader.unread(boundary);
        boolean validBoundary = boundary == -1 || boundary == ',' || boundary == '}' || boundary == ']' ||
                boundary == ' ' || boundary == '\t' || boundary == '\r' || boundary == '\n';
        return new BooleanLiteral(validBoundary, first == 't', raw.toString());
    }

    /** Decode only enough JSON escaping to compare property/event names. */
    private static String decodeJsonStringToken(String token) {
        if (token == null || token.length() < 2) return "";
        StringBuilder out = new StringBuilder(token.length() - 2);
        for (int i = 1; i < token.length() - 1; ++i) {
            char ch = token.charAt(i);
            if (ch != '\\' || i + 1 >= token.length() - 1) {
                out.append(ch);
                continue;
            }
            char next = token.charAt(++i);
            if (next == '"' || next == '\\' || next == '/') out.append(next);
            else if (next == 'b') out.append('\b');
            else if (next == 'f') out.append('\f');
            else if (next == 'n') out.append('\n');
            else if (next == 'r') out.append('\r');
            else if (next == 't') out.append('\t');
            else if (next == 'u' && i + 4 < token.length()) {
                int code = parseHex4(token, i + 1);
                if (code >= 0) {
                    out.append((char) code);
                    i += 4;
                } else {
                    out.append('\\').append(next);
                }
            } else {
                out.append(next);
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

    private static void outputFileSync(File file) throws IOException {
        // Re-open only for fsync after the Writer has flushed its encoded bytes. This avoids
        // retaining a second chart-sized buffer and keeps durability independent of Writer state.
        try (FileOutputStream sync = new FileOutputStream(file, true)) {
            sync.getFD().sync();
        }
    }
}
