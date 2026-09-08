package com.unity3d.player;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Tiny loopback-only HTTP server used to reproduce ADOFAI's historical
 * "Open From URL" input shape without uploading the user's level anywhere.
 */
public final class LoopbackZipServer {
    private static final Map<String, File> FILES = new ConcurrentHashMap<String, File>();
    private static volatile ServerSocket server;
    private static volatile Throwable startError;

    private LoopbackZipServer() {}

    public static String publish(File zip) throws Exception {
        if (zip == null || !zip.isFile()) throw new IllegalArgumentException("ZIP does not exist");
        ensureStarted();
        String token = UUID.randomUUID().toString().replace("-", "");
        FILES.put(token, zip.getCanonicalFile());
        return "http://127.0.0.1:" + server.getLocalPort() + "/bundle/" + token + "/level.zip";
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
            String line;
            while ((line = reader.readLine()) != null && line.length() > 0) { /* consume headers */ }

            File file = resolve(path);
            if (file == null || !("GET".equals(method) || "HEAD".equals(method))) {
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
                while ((read = input.read(buffer)) >= 0) if (read > 0) output.write(buffer, 0, read);
                output.flush();
            } finally {
                input.close();
            }
        } catch (Throwable ignored) {
        } finally {
            try { socket.close(); } catch (Throwable ignored) {}
        }
    }

    private static File resolve(String path) {
        if (path == null || !path.startsWith("/bundle/") || !path.endsWith("/level.zip")) return null;
        String middle = path.substring("/bundle/".length(), path.length() - "/level.zip".length());
        if (middle.indexOf('/') >= 0 || middle.length() == 0) return null;
        File file = FILES.get(middle);
        return file != null && file.isFile() ? file : null;
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
}
