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
import java.util.Deque;
import java.util.Locale;

/**
 * Narrow workaround for the historical HALL load bug present before ADOFAI 3.1.0.
 *
 * Public level documentation records that old builds can load HALL when tile hitsounds are
 * removed. We therefore touch only a private imported copy, only after identifying song=HALL and
 * artist=Frums, and only replace the initial tile hitsound plus SetHitsound event values with
 * None. Music, PlaySound events, chart timing and gameplay/custom-event data remain untouched.
 */
final class V240HallLegacyFix {
    private static final String TAG = "ADOFAI.V240HallFix";
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int IO_BUFFER_BYTES = 64 * 1024;
    private static final int IDENTITY_SCAN_CHARS = 8 * 1024 * 1024;
    private static final int MAX_STRING_TOKEN_CHARS = 16 * 1024 * 1024;

    private V240HallLegacyFix() {}

    static boolean applyIfNeeded(File chart) {
        if (chart == null || !chart.isFile()) return false;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES) return false;
        if (!isHallChart(chart)) return false;

        File parent = chart.getParentFile();
        if (parent == null) return false;
        File temp = new File(parent, chart.getName() + ".v240-hall.tmp");
        File backup = new File(parent, chart.getName() + ".v240-hall.original");
        try {
            if (temp.exists() && !temp.delete()) throw new IOException("stale HALL temp is locked");
            if (backup.exists() && !backup.delete()) throw new IOException("stale HALL backup is locked");
            int changes = suppressTileHitsounds(chart, temp);
            if (changes == 0) {
                temp.delete();
                return false;
            }
            if (!chart.renameTo(backup)) throw new IOException("could not stage HALL private chart");
            if (!temp.renameTo(chart)) {
                if (!backup.renameTo(chart)) Log.e(TAG, "Could not restore HALL private chart");
                throw new IOException("could not install HALL legacy workaround");
            }
            if (backup.exists() && !backup.delete()) backup.deleteOnExit();
            Log.i(TAG, "Applied pre-3.1 HALL hitsound workaround changes=" + changes);
            return true;
        } catch (Throwable error) {
            if (!chart.exists() && backup.exists() && !backup.renameTo(chart)) {
                Log.e(TAG, "HALL workaround recovery failed", error);
            }
            Log.w(TAG, "HALL workaround failed open", error);
            return false;
        } finally {
            if (temp.exists()) temp.delete();
        }
    }

    private static boolean isHallChart(File chart) {
        String artist = null;
        String song = null;
        BufferedInputStream input = null;
        PushbackReader reader = null;
        try {
            input = new BufferedInputStream(new FileInputStream(chart), IO_BUFFER_BYTES);
            skipUtf8Bom(input);
            reader = new PushbackReader(new InputStreamReader(input, StandardCharsets.UTF_8), 8);
            int scanned = 0;
            int value;
            while (scanned < IDENTITY_SCAN_CHARS && (value = reader.read()) != -1) {
                scanned++;
                if (value != '"') continue;
                String keyToken = readStringToken(reader);
                scanned += keyToken.length();
                StringBuilder whitespace = new StringBuilder(8);
                int next = readAfterWhitespace(reader, whitespace);
                scanned += whitespace.length() + 1;
                if (next != ':') {
                    if (next != -1) reader.unread(next);
                    continue;
                }
                whitespace.setLength(0);
                int first = readAfterWhitespace(reader, whitespace);
                scanned += whitespace.length() + 1;
                String key = decodeJsonStringToken(keyToken);
                if (("artist".equals(key) || "song".equals(key)) && first == '"') {
                    String token = readStringToken(reader);
                    scanned += token.length();
                    String decoded = cleanIdentity(decodeJsonStringToken(token));
                    if ("artist".equals(key) && artist == null) artist = decoded;
                    if ("song".equals(key) && song == null) song = decoded;
                    if (artist != null && song != null) break;
                } else if (first != -1) {
                    reader.unread(first);
                }
            }
        } catch (Throwable error) {
            Log.w(TAG, "Could not identify possible HALL chart", error);
            return false;
        } finally {
            try { if (reader != null) reader.close(); else if (input != null) input.close(); }
            catch (Throwable ignored) {}
        }
        return "hall".equals(song) && artist != null && artist.contains("frums");
    }

    private static int suppressTileHitsounds(File source, File target) throws IOException {
        FileInputStream rawInput = new FileInputStream(source);
        BufferedInputStream input = new BufferedInputStream(rawInput, IO_BUFFER_BYTES);
        FileOutputStream rawOutput = new FileOutputStream(target, false);
        BufferedOutputStream output = new BufferedOutputStream(rawOutput, IO_BUFFER_BYTES);
        try {
            boolean bom = skipUtf8Bom(input);
            if (bom) output.write(new byte[] {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
            PushbackReader reader = new PushbackReader(new InputStreamReader(input, StandardCharsets.UTF_8), 8);
            Writer writer = new OutputStreamWriter(output, StandardCharsets.UTF_8);
            int changes = transformHitsounds(reader, writer);
            writer.flush();
            output.flush();
            rawOutput.getFD().sync();
            return changes;
        } finally {
            try { input.close(); } finally { output.close(); }
        }
    }

    private static int transformHitsounds(PushbackReader reader, Writer writer) throws IOException {
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

            String keyToken = readStringToken(reader);
            writer.write(keyToken);
            StringBuilder whitespace = new StringBuilder(8);
            int next = readAfterWhitespace(reader, whitespace);
            writer.write(whitespace.toString());
            if (next != ':') {
                if (next != -1) reader.unread(next);
                continue;
            }
            writer.write(':');
            String key = decodeJsonStringToken(keyToken);
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

            if ("hitsound".equals(key) && first == '"' &&
                    (context == null || context.eventType == null ||
                            "SetHitsound".equals(context.eventType))) {
                String old = readStringToken(reader);
                if (!"none".equals(cleanIdentity(decodeJsonStringToken(old)))) changes++;
                writer.write("\"None\"");
                continue;
            }

            reader.unread(first);
        }
        return changes;
    }

    private static boolean skipUtf8Bom(BufferedInputStream input) throws IOException {
        input.mark(3);
        int b0 = input.read();
        int b1 = input.read();
        int b2 = input.read();
        if (b0 == 0xEF && b1 == 0xBB && b2 == 0xBF) return true;
        input.reset();
        return false;
    }

    private static String readStringToken(PushbackReader reader) throws IOException {
        StringBuilder token = new StringBuilder(64);
        token.append('"');
        boolean escaped = false;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            token.append(ch);
            if (token.length() > MAX_STRING_TOKEN_CHARS) throw new IOException("JSON string too large");
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') return token.toString();
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
                } else out.append('\\').append(next);
            } else out.append(next);
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

    private static String cleanIdentity(String value) {
        if (value == null) return "";
        StringBuilder plain = new StringBuilder(value.length());
        boolean inTag = false;
        for (int i = 0; i < value.length(); ++i) {
            char ch = value.charAt(i);
            if (ch == '<') inTag = true;
            else if (ch == '>' && inTag) inTag = false;
            else if (!inTag) plain.append(ch);
        }
        return plain.toString().trim().toLowerCase(Locale.US);
    }

    private static final class ObjectContext {
        String eventType;
    }
}
