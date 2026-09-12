package com.unity3d.player;

/**
 * Converts the safe subset of post-v2.4 SetFrameRate actions into a v2.4 CallMethod scheduler
 * carrier. The native runtime intercepts the deliberately non-callable method marker before the
 * original reflection path. The UUID remains tied to V240OpaqueEventBridge's sidecar so saves can
 * restore the exact original event JSON.
 */
final class V240SetFrameRateBackport {
    static final String MARKER_PREFIX = "__V240_SET_FRAME_RATE__:";

    private V240SetFrameRateBackport() {}

    static String maybePlaceholder(String eventObject, int floor, String token) {
        return maybePlaceholder(eventObject, floor, token,
                V240EventCompat.isSetFrameRateBackportReady());
    }

    // Package-private deterministic overload used by the host regression contract.
    static String maybePlaceholder(String eventObject, int floor, String token, boolean runtimeReady) {
        if (!runtimeReady || eventObject == null || !safeToken(token)) return null;
        if (!"SetFrameRate".equals(findTopLevelString(eventObject, "eventType"))) return null;

        Boolean active = findTopLevelBoolean(eventObject, "active");
        if (Boolean.FALSE.equals(active)) return null;

        // v2.4's CallMethod carrier does not have the newer editor-only scheduling semantics.
        // Never turn an editor-only effect into one that can run during ordinary gameplay.
        String editorOnlyRaw = findTopLevelRawValue(eventObject, "editorOnly");
        Boolean editorOnly = findTopLevelBoolean(eventObject, "editorOnly");
        if ((editorOnlyRaw != null && editorOnly == null) || Boolean.TRUE.equals(editorOnly)) return null;

        // Tagged effects can participate in RepeatEvents / SetInputEvent scheduling. v2.4 cannot
        // reproduce every post-v2.4 manual-trigger interaction, so keep those preserve-only.
        String eventTag = findTopLevelString(eventObject, "eventTag");
        if (eventTag != null && !eventTag.isEmpty()) return null;

        // Require an explicit enabled state instead of guessing a newer-version default.
        Boolean enabled = findTopLevelBoolean(eventObject, "enabled");
        if (enabled == null) return null;

        String frameRate = findTopLevelNumber(eventObject, "frameRate");
        if (enabled.booleanValue() && frameRate == null) return null;
        if (frameRate == null) frameRate = "0"; // disabling ignores the target rate

        String angleOffset = findTopLevelNumber(eventObject, "angleOffset");
        if (angleOffset == null) angleOffset = "0";

        String method = MARKER_PREFIX + token + ":" + (enabled.booleanValue() ? "1" : "0")
                + ":" + frameRate;
        return "{\"floor\":" + Math.max(0, floor)
                + ",\"eventType\":\"CallMethod\""
                + ",\"method\":" + jsonString(method)
                + ",\"angleOffset\":" + angleOffset
                + ",\"active\":true,\"locked\":true}";
    }

    static String tokenFromPlaceholder(String eventObject) {
        if (eventObject == null || !"CallMethod".equals(
                findTopLevelString(eventObject, "eventType"))) return null;
        String method = findTopLevelString(eventObject, "method");
        if (method == null || !method.startsWith(MARKER_PREFIX)) return null;

        String rest = method.substring(MARKER_PREFIX.length());
        int first = rest.indexOf(':');
        if (first != 36) return null;
        String token = rest.substring(0, first);
        if (!safeToken(token)) return null;
        if (first + 2 >= rest.length()) return null;
        char enabled = rest.charAt(first + 1);
        if ((enabled != '0' && enabled != '1') || rest.charAt(first + 2) != ':') return null;
        String frameRate = rest.substring(first + 3);
        if (!validFiniteNumber(frameRate)) return null;
        return token;
    }

    private static Boolean findTopLevelBoolean(String object, String wantedKey) {
        String raw = findTopLevelRawValue(object, wantedKey);
        if (raw == null) return null;
        if ("true".equals(raw)) return Boolean.TRUE;
        if ("false".equals(raw)) return Boolean.FALSE;
        if (raw.length() >= 2 && raw.charAt(0) == '"' && raw.charAt(raw.length() - 1) == '"') {
            String value = decodeJsonStringToken(raw);
            if ("Enabled".equalsIgnoreCase(value) || "true".equalsIgnoreCase(value)) return Boolean.TRUE;
            if ("Disabled".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) return Boolean.FALSE;
        }
        return null;
    }

    private static String findTopLevelNumber(String object, String wantedKey) {
        String raw = findTopLevelRawValue(object, wantedKey);
        if (raw == null || !validFiniteNumber(raw)) return null;
        return raw;
    }

    private static boolean validFiniteNumber(String raw) {
        if (raw == null || raw.isEmpty() || raw.charAt(0) == '"') return false;
        try {
            float value = Float.parseFloat(raw);
            return !Float.isNaN(value) && !Float.isInfinite(value);
        } catch (NumberFormatException ignored) {
            return false;
        }
    }

    private static String findTopLevelRawValue(String object, String wantedKey) {
        if (object == null || wantedKey == null) return null;
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
            if (key == null) return null;
            i = key.end;
            if (objectDepth != 1 || arrayDepth != 0 || !wantedKey.equals(key.decoded)) continue;
            int p = skipWhitespace(object, i);
            if (p >= object.length() || object.charAt(p) != ':') continue;
            p = skipWhitespace(object, p + 1);
            if (p >= object.length()) return null;
            if (object.charAt(p) == '"') {
                StringToken value = parseString(object, p);
                return value == null ? null : value.raw;
            }
            int end = p;
            while (end < object.length()) {
                char c = object.charAt(end);
                if (c == ',' || c == '}' || c == ' ' || c == '\t' || c == '\r' || c == '\n') break;
                end++;
            }
            return end > p ? object.substring(p, end) : null;
        }
        return null;
    }

    private static String findTopLevelString(String object, String wantedKey) {
        String raw = findTopLevelRawValue(object, wantedKey);
        if (raw == null || raw.length() < 2 || raw.charAt(0) != '"'
                || raw.charAt(raw.length() - 1) != '"') return null;
        return decodeJsonStringToken(raw);
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

    private static StringToken parseString(String value, int quoteIndex) {
        if (quoteIndex < 0 || quoteIndex >= value.length() || value.charAt(quoteIndex) != '"') return null;
        StringBuilder raw = new StringBuilder();
        raw.append('"');
        boolean escaped = false;
        for (int i = quoteIndex + 1; i < value.length(); ++i) {
            char ch = value.charAt(i);
            raw.append(ch);
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') {
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

    private static boolean safeToken(String token) {
        if (token == null || token.length() != 36) return false;
        for (int i = 0; i < token.length(); ++i) {
            char ch = token.charAt(i);
            if (i == 8 || i == 13 || i == 18 || i == 23) {
                if (ch != '-') return false;
                continue;
            }
            if ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) continue;
            return false;
        }
        return true;
    }
}
