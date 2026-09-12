package com.unity3d.player;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PushbackReader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.UUID;

/**
 * Lossless preserve-only bridge for chart events introduced after the v2.4 baseline.
 *
 * The legacy editor cannot decode event types it does not know. Rather than extending the
 * original IL2CPP enum/data tables, this bridge rewrites only the app-private working copy:
 *
 *   modern action      -> inactive EditorComment marker
 *   modern decoration  -> inactive AddDecoration marker
 *
 * The complete original JSON event object is stored beside the private chart. Save syncs never
 * copy the marker representation to the authoritative SAF document; writeRestoredCopy() replaces
 * every surviving marker with its exact original JSON object while streaming the file out.
 *
 * This is intentionally preserve-only. v2.4 cannot edit the opaque payload. If a marker survives,
 * the original event survives byte-for-byte. If a marker is deliberately removed, that original
 * event is omitted on export. No original game native library is modified.
 */
final class V240OpaqueEventBridge {
    private static final String TAG = "ADOFAI.V240Opaque";
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int IO_BUFFER_BYTES = 64 * 1024;
    private static final int MAX_EVENT_OBJECT_CHARS = 16 * 1024 * 1024;
    private static final String SESSION_SUFFIX = ".v240-opaque-events";
    private static final String READY = ".ready";
    private static final String MARKER_PREFIX = "__V240_OPAQUE__:";

    private enum Section {
        NONE,
        ACTIONS,
        DECORATIONS
    }

    private enum Mode {
        PREPARE,
        RESTORE
    }

    private static final class RewriteState {
        int replacements;
        int objectDepth;
        int arrayDepth;
        int targetArrayDepth = -1;
        Section section = Section.NONE;
    }

    private static final class StringToken {
        final String raw;
        final String decoded;
        final int end;

        StringToken(String raw, String decoded, int end) {
            this.raw = raw;
            this.decoded = decoded;
            this.end = end;
        }
    }

    private static final class Marker {
        final String token;
        final Section section;

        Marker(String token, Section section) {
            this.token = token;
            this.section = section;
        }
    }

    private V240OpaqueEventBridge() {}

    static boolean hasSession(File chart) {
        File dir = sessionDir(chart);
        return dir.isDirectory() && new File(dir, READY).isFile();
    }

    static boolean prepareForV240(File chart) {
        if (chart == null || !chart.isFile()) return false;
        if (hasSession(chart)) return true;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES) {
            Log.w(TAG, "Skipping oversized opaque-event bridge: " + length);
            return false;
        }

        V240ChartCompatibilityScanner.Result scan = V240ChartCompatibilityScanner.scanAndLog(chart);
        Set<String> targetTypes = scan.postV240;
        if (targetTypes.isEmpty()) return false;

        File parent = chart.getParentFile();
        if (parent == null) return false;
        File session = sessionDir(chart);
        File temp = new File(parent, chart.getName() + ".v240-opaque.tmp");
        File backup = new File(parent, chart.getName() + ".v240-opaque.original");

        try {
            deleteRecursively(session);
            if (!session.mkdirs() && !session.isDirectory()) {
                throw new IOException("opaque sidecar directory could not be created");
            }
            if (temp.exists() && !temp.delete()) throw new IOException("stale opaque temp is locked");
            if (backup.exists() && !backup.delete()) throw new IOException("stale opaque backup is locked");

            int replacements = rewriteFile(chart, temp, session, targetTypes, Mode.PREPARE);
            if (replacements <= 0) {
                deleteRecursively(session);
                if (temp.exists()) temp.delete();
                return false;
            }

            writeSmallFile(new File(session, READY), Integer.toString(replacements));
            if (!chart.renameTo(backup)) throw new IOException("could not stage private chart");
            if (!temp.renameTo(chart)) {
                if (!backup.renameTo(chart)) {
                    Log.e(TAG, "Could not restore private chart after opaque bridge failure");
                }
                throw new IOException("could not install opaque chart representation");
            }
            if (backup.exists() && !backup.delete()) backup.deleteOnExit();
            Log.d(TAG, "Preserved post-v2.4 events as opaque placeholders count=" + replacements
                    + " file=" + chart.getName());
            return true;
        } catch (Throwable error) {
            if (!chart.exists() && backup.exists()) {
                if (!backup.renameTo(chart)) Log.e(TAG, "Opaque bridge recovery failed", error);
            }
            deleteRecursively(session);
            Log.w(TAG, "Opaque-event bridge failed open", error);
            return false;
        } finally {
            if (temp.exists()) temp.delete();
        }
    }

    /**
     * Clone preserve-only sidecar state before Save As returns a new local working path.
     * The v2.4 serializer will copy the marker events into the new chart; the cloned sidecar lets
     * that chart restore the same original payloads on its first SAF sync.
     */
    static boolean cloneSession(File sourceChart, File destinationChart) {
        if (!hasSession(sourceChart)) return true;
        if (destinationChart == null) return false;
        File source = sessionDir(sourceChart);
        File destination = sessionDir(destinationChart);
        try {
            deleteRecursively(destination);
            if (!destination.mkdirs() && !destination.isDirectory()) {
                throw new IOException("destination opaque sidecar could not be created");
            }
            File[] children = source.listFiles();
            if (children == null) throw new IOException("source opaque sidecar could not be listed");
            byte[] buffer = new byte[IO_BUFFER_BYTES];
            for (File child : children) {
                if (child == null || !child.isFile()) continue;
                File target = new File(destination, child.getName());
                try (InputStream in = new BufferedInputStream(new FileInputStream(child), IO_BUFFER_BYTES);
                     OutputStream out = new BufferedOutputStream(new FileOutputStream(target), IO_BUFFER_BYTES)) {
                    int count;
                    while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                    out.flush();
                }
            }
            return new File(destination, READY).isFile();
        } catch (Throwable error) {
            deleteRecursively(destination);
            Log.e(TAG, "Could not clone opaque-event state for Save As", error);
            return false;
        }
    }

    /**
     * Writes an export view with opaque markers replaced by the exact original event objects.
     * Returns false when no opaque session exists, allowing the caller to use its normal byte copy.
     */
    static boolean writeRestoredCopy(File chart, OutputStream output) throws Exception {
        if (!hasSession(chart)) return false;
        if (chart == null || !chart.isFile()) throw new IOException("opaque chart is unavailable");
        if (output == null) throw new IOException("opaque export output is null");
        File session = sessionDir(chart);
        rewriteStream(chart, output, session, null, Mode.RESTORE);
        return true;
    }

    private static int rewriteFile(File source, File target, File session,
                                   Set<String> targetTypes, Mode mode) throws Exception {
        try (BufferedOutputStream output = new BufferedOutputStream(
                new FileOutputStream(target, false), IO_BUFFER_BYTES)) {
            int changes = rewriteStream(source, output, session, targetTypes, mode);
            output.flush();
            try (FileOutputStream sync = new FileOutputStream(target, true)) {
                sync.getFD().sync();
            }
            return changes;
        }
    }

    private static int rewriteStream(File source, OutputStream output, File session,
                                     Set<String> targetTypes, Mode mode) throws Exception {
        try (BufferedInputStream input = new BufferedInputStream(
                new FileInputStream(source), IO_BUFFER_BYTES)) {
            input.mark(3);
            int b0 = input.read();
            int b1 = input.read();
            int b2 = input.read();
            boolean bom = b0 == 0xEF && b1 == 0xBB && b2 == 0xBF;
            if (!bom) input.reset();
            else output.write(new byte[] {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});

            PushbackReader reader = new PushbackReader(
                    new InputStreamReader(input, StandardCharsets.UTF_8), 8);
            Writer writer = new OutputStreamWriter(output, StandardCharsets.UTF_8);
            RewriteState state = new RewriteState();
            rewriteJson(reader, writer, session, targetTypes, mode, state);
            writer.flush();
            return state.replacements;
        }
    }

    private static void rewriteJson(PushbackReader reader, Writer writer, File session,
                                    Set<String> targetTypes, Mode mode, RewriteState state)
            throws Exception {
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;

            if (state.section != Section.NONE && ch == '{'
                    && state.objectDepth == 1
                    && state.arrayDepth == state.targetArrayDepth) {
                String eventObject = readBalancedObject(reader);
                if (mode == Mode.PREPARE) {
                    String eventType = findTopLevelString(eventObject, "eventType");
                    if (eventType != null && targetTypes != null && targetTypes.contains(eventType)) {
                        String token = UUID.randomUUID().toString();
                        writeSmallFile(new File(session, token + ".json"), eventObject);
                        int floor = findTopLevelInt(eventObject, "floor", 0);
                        writer.write(placeholder(state.section, floor, token, eventType));
                        state.replacements++;
                    } else {
                        writer.write(eventObject);
                    }
                } else {
                    Marker marker = findMarker(eventObject, state.section);
                    if (marker == null) {
                        writer.write(eventObject);
                    } else {
                        File original = new File(session, marker.token + ".json");
                        if (!original.isFile()) {
                            throw new IOException("opaque event sidecar missing for token " + marker.token);
                        }
                        copyUtf8File(original, writer);
                        state.replacements++;
                    }
                }
                continue;
            }

            if (ch == '"') {
                String token = readStringToken(reader);
                writer.write(token);

                if (state.section == Section.NONE && state.objectDepth == 1 && state.arrayDepth == 0) {
                    StringBuilder whitespace = new StringBuilder(8);
                    int next = readAfterWhitespace(reader, whitespace);
                    writer.write(whitespace.toString());
                    if (next != ':') {
                        if (next != -1) reader.unread(next);
                        continue;
                    }
                    writer.write(':');
                    String key = decodeJsonStringToken(token);
                    whitespace.setLength(0);
                    int first = readAfterWhitespace(reader, whitespace);
                    writer.write(whitespace.toString());
                    if (first == -1) return;
                    Section section = "actions".equals(key) ? Section.ACTIONS
                            : ("decorations".equals(key) ? Section.DECORATIONS : Section.NONE);
                    if (section != Section.NONE && first == '[') {
                        writer.write('[');
                        state.arrayDepth++;
                        state.section = section;
                        state.targetArrayDepth = state.arrayDepth;
                    } else {
                        reader.unread(first);
                    }
                }
                continue;
            }

            writer.write(ch);
            if (ch == '{') {
                state.objectDepth++;
            } else if (ch == '}') {
                if (state.objectDepth > 0) state.objectDepth--;
            } else if (ch == '[') {
                state.arrayDepth++;
            } else if (ch == ']') {
                if (state.section != Section.NONE && state.arrayDepth == state.targetArrayDepth) {
                    state.section = Section.NONE;
                    state.targetArrayDepth = -1;
                }
                if (state.arrayDepth > 0) state.arrayDepth--;
            }
        }
    }

    /** Opening brace has already been consumed. */
    private static String readBalancedObject(PushbackReader reader) throws IOException {
        StringBuilder raw = new StringBuilder(512);
        raw.append('{');
        int depth = 1;
        boolean inString = false;
        boolean escaped = false;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            raw.append(ch);
            if (raw.length() > MAX_EVENT_OBJECT_CHARS) {
                throw new IOException("event object exceeds opaque bridge safety limit");
            }
            if (inString) {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') inString = false;
                continue;
            }
            if (ch == '"') {
                inString = true;
            } else if (ch == '{') {
                depth++;
            } else if (ch == '}') {
                depth--;
                if (depth == 0) return raw.toString();
            }
        }
        throw new IOException("unterminated event object");
    }

    private static String placeholder(Section section, int floor, String token, String eventType) {
        String marker = MARKER_PREFIX + token + ":"
                + (section == Section.ACTIONS ? "A" : "D") + ":" + eventType;
        if (section == Section.ACTIONS) {
            return "{\"floor\":" + Math.max(0, floor)
                    + ",\"eventType\":\"EditorComment\""
                    + ",\"comment\":" + jsonString(marker)
                    + ",\"active\":false,\"locked\":true}";
        }
        return "{\"floor\":" + Math.max(0, floor)
                + ",\"eventType\":\"AddDecoration\""
                + ",\"decorationImage\":\"\",\"opacity\":0"
                + ",\"tag\":" + jsonString(marker)
                + ",\"active\":false,\"locked\":true}";
    }

    private static Marker findMarker(String eventObject, Section section) {
        String eventType = findTopLevelString(eventObject, "eventType");
        String value;
        if (section == Section.ACTIONS && "EditorComment".equals(eventType)) {
            value = findTopLevelString(eventObject, "comment");
        } else if (section == Section.DECORATIONS && "AddDecoration".equals(eventType)) {
            value = findTopLevelString(eventObject, "tag");
        } else {
            return null;
        }
        if (value == null || !value.startsWith(MARKER_PREFIX)) return null;
        String rest = value.substring(MARKER_PREFIX.length());
        int first = rest.indexOf(':');
        if (first <= 0) return null;
        int second = rest.indexOf(':', first + 1);
        if (second <= first + 1) return null;
        String token = rest.substring(0, first);
        String sectionCode = rest.substring(first + 1, second);
        if (!safeToken(token)) return null;
        if (section == Section.ACTIONS && !"A".equals(sectionCode)) return null;
        if (section == Section.DECORATIONS && !"D".equals(sectionCode)) return null;
        return new Marker(token, section);
    }

    private static boolean safeToken(String token) {
        if (token == null || token.length() != 36) return false;
        for (int i = 0; i < token.length(); ++i) {
            char ch = token.charAt(i);
            if (ch == '-') continue;
            if ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) continue;
            return false;
        }
        return true;
    }

    private static String findTopLevelString(String object, String wantedKey) {
        if (object == null || wantedKey == null) return null;
        int objectDepth = 0;
        int arrayDepth = 0;
        for (int i = 0; i < object.length();) {
            char ch = object.charAt(i);
            if (ch == '{') {
                objectDepth++;
                i++;
                continue;
            }
            if (ch == '}') {
                objectDepth--;
                i++;
                continue;
            }
            if (ch == '[') {
                arrayDepth++;
                i++;
                continue;
            }
            if (ch == ']') {
                arrayDepth--;
                i++;
                continue;
            }
            if (ch != '"') {
                i++;
                continue;
            }
            StringToken key = parseString(object, i);
            if (key == null) return null;
            i = key.end;
            if (objectDepth != 1 || arrayDepth != 0) continue;
            int p = skipWhitespace(object, i);
            if (p >= object.length() || object.charAt(p) != ':') continue;
            p = skipWhitespace(object, p + 1);
            if (!wantedKey.equals(key.decoded) || p >= object.length() || object.charAt(p) != '"') continue;
            StringToken value = parseString(object, p);
            return value == null ? null : value.decoded;
        }
        return null;
    }

    private static int findTopLevelInt(String object, String wantedKey, int fallback) {
        if (object == null || wantedKey == null) return fallback;
        int objectDepth = 0;
        int arrayDepth = 0;
        for (int i = 0; i < object.length();) {
            char ch = object.charAt(i);
            if (ch == '{') { objectDepth++; i++; continue; }
            if (ch == '}') { objectDepth--; i++; continue; }
            if (ch == '[') { arrayDepth++; i++; continue; }
            if (ch == ']') { arrayDepth--; i++; continue; }
            if (ch != '"') { i++; continue; }
            StringToken key = parseString(object, i);
            if (key == null) return fallback;
            i = key.end;
            if (objectDepth != 1 || arrayDepth != 0) continue;
            int p = skipWhitespace(object, i);
            if (p >= object.length() || object.charAt(p) != ':') continue;
            p = skipWhitespace(object, p + 1);
            if (!wantedKey.equals(key.decoded) || p >= object.length()) continue;
            int start = p;
            if (object.charAt(p) == '-') p++;
            while (p < object.length() && Character.isDigit(object.charAt(p))) p++;
            if (p == start || (p == start + 1 && object.charAt(start) == '-')) return fallback;
            try {
                return Integer.parseInt(object.substring(start, p));
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static StringToken parseString(String value, int quoteIndex) {
        if (quoteIndex < 0 || quoteIndex >= value.length() || value.charAt(quoteIndex) != '"') return null;
        StringBuilder raw = new StringBuilder();
        raw.append('"');
        boolean escaped = false;
        for (int i = quoteIndex + 1; i < value.length(); ++i) {
            char ch = value.charAt(i);
            raw.append(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                String rawValue = raw.toString();
                return new StringToken(rawValue, decodeJsonStringToken(rawValue), i + 1);
            }
        }
        return null;
    }

    private static int skipWhitespace(String value, int offset) {
        int i = offset;
        while (i < value.length()) {
            char ch = value.charAt(i);
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
            i++;
        }
        return i;
    }

    /** Read a raw JSON string token after the opening quote was consumed. */
    private static String readStringToken(PushbackReader reader) throws IOException {
        StringBuilder token = new StringBuilder(64);
        token.append('"');
        boolean escaped = false;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            token.append(ch);
            if (token.length() > MAX_EVENT_OBJECT_CHARS) {
                throw new IOException("JSON string token exceeds opaque bridge safety limit");
            }
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') return token.toString();
        }
        throw new IOException("unterminated JSON string");
    }

    private static int readAfterWhitespace(PushbackReader reader, StringBuilder whitespace)
            throws IOException {
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

    private static String jsonString(String value) {
        StringBuilder out = new StringBuilder(value.length() + 2);
        out.append('"');
        for (int i = 0; i < value.length(); ++i) {
            char ch = value.charAt(i);
            if (ch == '"' || ch == '\\') out.append('\\').append(ch);
            else if (ch == '\b') out.append("\\b");
            else if (ch == '\f') out.append("\\f");
            else if (ch == '\n') out.append("\\n");
            else if (ch == '\r') out.append("\\r");
            else if (ch == '\t') out.append("\\t");
            else if (ch < 0x20) {
                String hex = Integer.toHexString(ch);
                out.append("\\u");
                for (int pad = hex.length(); pad < 4; ++pad) out.append('0');
                out.append(hex);
            } else out.append(ch);
        }
        out.append('"');
        return out.toString();
    }

    private static void copyUtf8File(File source, Writer writer) throws IOException {
        char[] buffer = new char[16 * 1024];
        try (InputStreamReader reader = new InputStreamReader(
                new FileInputStream(source), StandardCharsets.UTF_8)) {
            int count;
            while ((count = reader.read(buffer)) != -1) writer.write(buffer, 0, count);
        }
    }

    private static void writeSmallFile(File file, String value) throws IOException {
        try (FileOutputStream output = new FileOutputStream(file, false);
             Writer writer = new OutputStreamWriter(output, StandardCharsets.UTF_8)) {
            writer.write(value);
            writer.flush();
            output.getFD().sync();
        }
    }

    private static File sessionDir(File chart) {
        if (chart == null) return new File(".", "invalid" + SESSION_SUFFIX);
        File parent = chart.getParentFile();
        return new File(parent == null ? new File(".") : parent, chart.getName() + SESSION_SUFFIX);
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursively(child);
            }
        }
        if (!file.delete()) file.deleteOnExit();
    }
}
