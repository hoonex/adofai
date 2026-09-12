package com.unity3d.player;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PushbackReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

/**
 * Read-only streaming preflight for charts authored by later ADOFAI editors.
 *
 * This class intentionally does not normalize, delete, rename, or downgrade events. It only
 * reads JSON string tokens and records eventType values whose execution/semantics are known to
 * differ from the v2.4 baseline. A decoration-heavy chart therefore does not need to be loaded
 * into a JSONObject or duplicated in memory just to produce compatibility diagnostics.
 */
final class V240ChartCompatibilityScanner {
    private static final String TAG = "ADOFAI.V240Compat";
    private static final long MAX_BYTES = 512L * 1024L * 1024L;
    private static final int IO_BUFFER_BYTES = 64 * 1024;
    private static final int MAX_STRING_TOKEN_CHARS = 16 * 1024 * 1024;

    private static final Set<String> POST_V240 = new HashSet<String>(Arrays.asList(
            "SetFilterAdvanced",
            "SetFrameRate",
            "AddParticle",
            "SetParticle",
            "EmitParticle",
            "SetInputEvent",
            "TileDimensions"
    ));

    private static final Set<String> SEMANTIC_DRIFT = new HashSet<String>(Arrays.asList(
            "FreeRoam",
            "Pause",
            "SetConditionalEvents",
            "SetDefaultText",
            "RepeatEvents",
            "ColorTrack",
            "RecolorTrack",
            "MoveDecorations",
            "MoveTrack",
            "MoveCamera",
            "MultiPlanet"
    ));

    private V240ChartCompatibilityScanner() {}

    static final class Result {
        final Set<String> postV240 = new TreeSet<String>();
        final Set<String> semanticDrift = new TreeSet<String>();

        boolean requiresReview() {
            return !postV240.isEmpty() || !semanticDrift.isEmpty();
        }
    }

    static Result scan(File chart) {
        Result result = new Result();
        if (chart == null || !chart.isFile()) return result;
        long length = chart.length();
        if (length < 0L || length > MAX_BYTES) {
            Log.w(TAG, "Skipping oversized compatibility scan: " + length);
            return result;
        }

        try (BufferedInputStream input = new BufferedInputStream(
                     new FileInputStream(chart), IO_BUFFER_BYTES)) {
            input.mark(3);
            int b0 = input.read();
            int b1 = input.read();
            int b2 = input.read();
            boolean bom = b0 == 0xEF && b1 == 0xBB && b2 == 0xBF;
            if (!bom) input.reset();

            PushbackReader reader = new PushbackReader(
                    new InputStreamReader(input, StandardCharsets.UTF_8), 8);
            scanTokens(reader, result);
        } catch (Throwable error) {
            // Diagnostics must never stop the legacy editor from opening an otherwise usable map.
            Log.w(TAG, "Compatibility scan failed open: " + chart.getName(), error);
        }
        return result;
    }

    static Result scanAndLog(File chart) {
        Result result = scan(chart);
        if (result.requiresReview()) {
            Log.w(TAG, "v2.4 runtime review required file=" + safeName(chart)
                    + " post-v2.4=" + result.postV240
                    + " semantic-drift=" + result.semanticDrift);
        } else {
            Log.d(TAG, "No known post-v2.4 event risk detected file=" + safeName(chart));
        }
        return result;
    }

    private static void scanTokens(PushbackReader reader, Result result) throws Exception {
        int value;
        while ((value = reader.read()) != -1) {
            if (value != '"') continue;

            String token = readStringToken(reader);
            StringBuilder whitespace = new StringBuilder(8);
            int next = readAfterWhitespace(reader, whitespace);
            if (next != ':') {
                if (next != -1) reader.unread(next);
                continue;
            }

            String key = decodeJsonStringToken(token);
            whitespace.setLength(0);
            int first = readAfterWhitespace(reader, whitespace);
            if (first == -1) return;

            if ("eventType".equals(key) && first == '"') {
                String eventType = decodeJsonStringToken(readStringToken(reader));
                if (POST_V240.contains(eventType)) result.postV240.add(eventType);
                if (SEMANTIC_DRIFT.contains(eventType)) result.semanticDrift.add(eventType);
                continue;
            }

            reader.unread(first);
        }
    }

    /** Read a raw JSON string token after the opening quote was consumed. */
    private static String readStringToken(PushbackReader reader) throws Exception {
        StringBuilder token = new StringBuilder(64);
        token.append('"');
        boolean escaped = false;
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            token.append(ch);
            if (token.length() > MAX_STRING_TOKEN_CHARS) {
                throw new IllegalStateException("JSON string token exceeds compatibility safety limit");
            }
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                return token.toString();
            }
        }
        throw new IllegalStateException("unterminated JSON string");
    }

    private static int readAfterWhitespace(PushbackReader reader, StringBuilder whitespace)
            throws Exception {
        int value;
        while ((value = reader.read()) != -1) {
            char ch = (char) value;
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
                whitespace.append(ch);
            } else {
                return value;
            }
        }
        return -1;
    }

    /** Decode only enough JSON escaping to compare key and event names. */
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

    private static String safeName(File chart) {
        return chart == null ? "<null>" : chart.getName();
    }
}
