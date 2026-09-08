package com.unity3d.player;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Tiny loopback-only HTTP server used to reproduce ADOFAI's historical
 * "Open From URL" input shape without uploading the user's level anywhere.
 *
 * The server also records whether the published ZIP was actually requested.
 * That turns a vague "the game opened" result into useful runtime evidence:
 * HEAD means the URL was touched, while GET + served bytes means the ZIP body
 * was requested from the companion process.
 */
public final class LoopbackZipServer {
    private static final Map<String, File> FILES = new ConcurrentHashMap<String, File>();
    private static final Map<String, ProbeState> PROBES = new ConcurrentHashMap<String, ProbeState>();
    private static volatile ServerSocket server;
    private static volatile Throwable startError;

    private LoopbackZipServer() {}

    public static String publish(File zip) throws Exception {
        if (zip == null || !zip.isFile()) throw new IllegalArgumentException("ZIP does not exist");
        ensureStarted();
        String token = UUID.randomUUID().toString().replace("-", "");
        FILES.put(token, zip.getCanonicalFile());
        PROBES.put(token, new ProbeState());
        return "http://127.0.0.1:" + server.getLocalPort() + "/bundle/" + token + "/level.zip";
    }

    /** Returns a user-facing summary for the exact published URL, or null if unknown. */
    public static String diagnosticFor(String url) {
        String token = tokenFromUrl(url);
        if (token == null) return null;
        ProbeState probe = PROBES.get(token);
        if (probe == null) return null;
        return probe.describe();
    }

    private static synchronized void ensureStarted() throws Exception {
        if (server != null && !server.isClosed()) return;
        startError = null;
        final CountDownLatch latch = new CountDownLatch(1);
        Thread starter = new Thread(new Runnable() {
            @Override public void run() {
                try {
                    server = new ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"));
                    Thread accept = new Thread(new Runnable() {
                        @Override public void run() { acceptLoop(); }
                    }, "adofai-loopback-accept");
                    accept.setDaemon(true);
                    accept.start();
                } catch (Throwable error) {
                    startError = error;
                } finally {
                    latch.countDown();
                }
            }
        }, "adofai-loopback-start");
        starter.setDaemon(true);
        starter.start();
        if (!latch.await(3, TimeUnit.SECONDS)) throw new IllegalStateException("Loopback ZIP server start timed out");
        if (startError != null) throw new IllegalStateException("Loopback ZIP server failed", startError);
        if (server == null) throw new IllegalStateException("Loopback ZIP server did not start");
    }

    private static void acceptLoop() {
        while (true) {
            try {
                final Socket socket = server.accept();
                Thread worker = new Thread(new Runnable() {
                    @Override public void run() { handle(socket); }
                }, "adofai-loopback-client");
                worker.setDaemon(true);
                worker.start();
            } catch (Throwable ignored) {
                return;
            }
        }
    }

    private static void handle(Socket socket) {
        try {
            socket.setSoTimeout(5000);
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.US_ASCII));
            String request = reader.readLine();
            if (request == null) return;
            String[] parts = request.split(" ");
            String method = parts.length > 0 ? parts[0] : "";
            String path = parts.length > 1 ? parts[1] : "";
            String userAgent = "";
            String line;
            while ((line = reader.readLine()) != null && line.length() > 0) {
                int colon = line.indexOf(':');
                if (colon > 0 && "user-agent".equalsIgnoreCase(line.substring(0, colon).trim())) {
                    userAgent = line.substring(colon + 1).trim();
                }
            }

            String token = tokenFromPath(path);
            File file = token == null ? null : FILES.get(token);
            ProbeState probe = token == null ? null : PROBES.get(token);
            if (probe != null) probe.noteRequest(method, userAgent);

            if (file == null || !file.isFile() || !("GET".equals(method) || "HEAD".equals(method))) {
                writeStatus(socket.getOutputStream(), 404, "Not Found", 0L, null, false);
                return;
            }

            OutputStream output = socket.getOutputStream();
            writeStatus(output, 200, "OK", file.length(), "application/zip", true);
            if ("HEAD".equals(method)) return;
            FileInputStream input = new FileInputStream(file);
            try {
                byte[] buffer = new byte[64 * 1024];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    if (read <= 0) continue;
                    output.write(buffer, 0, read);
                    if (probe != null) probe.noteBytes(read);
                }
                output.flush();
            } finally {
                input.close();
            }
        } catch (Throwable ignored) {
        } finally {
            try { socket.close(); } catch (Throwable ignored) {}
        }
    }

    private static String tokenFromUrl(String url) {
        if (url == null || url.length() == 0) return null;
        try {
            return tokenFromPath(new URI(url).getPath());
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static String tokenFromPath(String path) {
        if (path == null || !path.startsWith("/bundle/") || !path.endsWith("/level.zip")) return null;
        String middle = path.substring("/bundle/".length(), path.length() - "/level.zip".length());
        if (middle.indexOf('/') >= 0 || middle.length() == 0) return null;
        return middle;
    }

    private static void writeStatus(OutputStream output, int code, String reason, long length,
                                    String type, boolean cacheControl) throws Exception {
        StringBuilder headers = new StringBuilder();
        headers.append("HTTP/1.1 ").append(code).append(' ').append(reason).append("\r\n");
        headers.append("Content-Length: ").append(length).append("\r\n");
        if (type != null) headers.append("Content-Type: ").append(type).append("\r\n");
        if (cacheControl) headers.append("Cache-Control: no-store\r\n");
        headers.append("Connection: close\r\n\r\n");
        output.write(headers.toString().getBytes(StandardCharsets.US_ASCII));
        output.flush();
    }

    private static final class ProbeState {
        private int requests;
        private int getRequests;
        private int headRequests;
        private long bytesServed;
        private long lastRequestAtMs;
        private String lastUserAgent = "";

        synchronized void noteRequest(String method, String userAgent) {
            requests++;
            if ("GET".equals(method)) getRequests++;
            if ("HEAD".equals(method)) headRequests++;
            lastRequestAtMs = System.currentTimeMillis();
            if (userAgent != null && userAgent.length() > 0) lastUserAgent = userAgent;
        }

        synchronized void noteBytes(long count) {
            if (count > 0) bytesServed += count;
        }

        synchronized String describe() {
            String ua = compactUserAgent(lastUserAgent);
            if (requests == 0) {
                return "ZIP URL 요청 0회: 공식 앱이 localhost ZIP을 요청한 흔적이 없습니다. "
                        + "Intent 미소비와 Android cleartext HTTP 차단은 아직 구분되지 않습니다.";
            }
            if (getRequests > 0) {
                return "ZIP URL GET " + getRequests + "회 / HEAD " + headRequests + "회, "
                        + bytesServed + " bytes 전송: ZIP URL 본문 요청이 실제로 감지됐습니다"
                        + (ua.length() == 0 ? "." : " (UA: " + ua + ").");
            }
            return "ZIP URL HEAD " + headRequests + "회 감지, GET 0회: URL 접근은 시작됐지만 ZIP 본문 다운로드는 확인되지 않았습니다"
                    + (ua.length() == 0 ? "." : " (UA: " + ua + ").");
        }
    }

    private static String compactUserAgent(String userAgent) {
        if (userAgent == null) return "";
        String value = userAgent.trim().replace('\n', ' ').replace('\r', ' ');
        return value.length() <= 96 ? value : value.substring(0, 96) + "…";
    }
}
