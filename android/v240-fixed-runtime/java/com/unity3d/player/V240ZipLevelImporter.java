package com.unity3d.player;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.Locale;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

/** Safe archive importer for downloaded TUF/legacy ADOFAI map bundles. */
final class V240ZipLevelImporter {
    private static final int MAX_FILES = 8192;
    private static final int MAX_DEPTH = 64;
    private static final long MAX_BYTES = 1024L * 1024L * 1024L;
    private static final int BUFFER_BYTES = 256 * 1024;

    private V240ZipLevelImporter() {}

    static File importArchive(File archive) throws Exception {
        if (archive == null || !archive.isFile()) {
            throw new IllegalArgumentException("archive does not exist");
        }
        File session = archive.getParentFile();
        if (session == null) throw new IllegalStateException("archive has no working directory");
        File root = new File(session, "unpacked-" + UUID.randomUUID().toString());
        if (!root.mkdirs() && !root.isDirectory()) {
            throw new IllegalStateException("could not create archive workspace");
        }

        Candidate best = null;
        int files = 0;
        long totalBytes = 0L;
        ZipFile zip = null;
        boolean success = false;
        try {
            zip = openZipWithFallback(archive);
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String rawName = entry.getName();
                if (rawName == null || rawName.length() == 0) continue;
                String normalized = rawName.replace('\\', '/');
                while (normalized.startsWith("/")) normalized = normalized.substring(1);
                if (normalized.length() == 0) continue;
                int depth = pathDepth(normalized);
                if (depth > MAX_DEPTH) throw new IllegalStateException("archive is nested too deeply");

                File target = safeTarget(root, normalized);
                if (entry.isDirectory() || normalized.endsWith("/")) {
                    if (!target.mkdirs() && !target.isDirectory()) {
                        throw new IllegalStateException("could not create archive directory");
                    }
                    continue;
                }

                if (++files > MAX_FILES) throw new IllegalStateException("archive has too many files");
                long declared = entry.getSize();
                long remaining = MAX_BYTES - totalBytes;
                if (remaining < 0L || (declared >= 0L && declared > remaining)) {
                    throw new IllegalStateException("archive expands beyond 1 GiB safety limit");
                }

                File parent = target.getParentFile();
                if (parent != null && !parent.mkdirs() && !parent.isDirectory()) {
                    throw new IllegalStateException("could not create archive parent directory");
                }
                long written;
                try (InputStream in = zip.getInputStream(entry);
                     FileOutputStream out = new FileOutputStream(target, false)) {
                    written = copyBounded(in, out, remaining);
                    out.getFD().sync();
                }
                totalBytes += written;

                Candidate candidate = Candidate.forFile(root, target, written);
                if (candidate != null && (best == null || candidate.betterThan(best))) best = candidate;
            }
            if (best == null) {
                throw new IllegalArgumentException("archive does not contain a usable .adofai level");
            }
            success = true;
            return best.file;
        } finally {
            if (zip != null) try { zip.close(); } catch (Throwable ignored) {}
            if (success) {
                // The archive is only a transport container after extraction. Removing the local
                // copy saves storage while preserving the complete extracted map tree.
                if (!archive.delete()) archive.deleteOnExit();
            } else {
                deleteRecursively(root);
            }
        }
    }

    private static ZipFile openZipWithFallback(File archive) throws IOException {
        Charset[] charsets = new Charset[] {
                StandardCharsets.UTF_8,
                charset("CP437"),
                charset("MS949"),
                charset("Shift_JIS")
        };
        IOException first = null;
        for (Charset charset : charsets) {
            if (charset == null) continue;
            try {
                return new ZipFile(archive, charset);
            } catch (ZipException error) {
                if (first == null) first = error;
            } catch (IOException error) {
                if (first == null) first = error;
            } catch (IllegalArgumentException error) {
                if (first == null) first = new IOException(error);
            }
        }
        if (first != null) throw first;
        throw new IOException("could not decode ZIP archive");
    }

    private static Charset charset(String name) {
        try { return Charset.forName(name); }
        catch (Throwable ignored) { return null; }
    }

    private static File safeTarget(File root, String relative) throws IOException {
        File target = new File(root, relative);
        String rootPath = root.getCanonicalPath();
        String targetPath = target.getCanonicalPath();
        if (!targetPath.equals(rootPath) && !targetPath.startsWith(rootPath + File.separator)) {
            throw new SecurityException("archive entry escapes workspace");
        }
        return target;
    }

    private static int pathDepth(String path) {
        int depth = 0;
        for (int i = 0; i < path.length(); ++i) if (path.charAt(i) == '/') depth++;
        return depth;
    }

    private static long copyBounded(InputStream in, FileOutputStream out, long maxBytes)
            throws IOException {
        byte[] buffer = new byte[BUFFER_BYTES];
        long total = 0L;
        int count;
        while ((count = in.read(buffer)) != -1) {
            if (count == 0) continue;
            total += count;
            if (total > maxBytes) throw new IOException("archive expands beyond safety limit");
            out.write(buffer, 0, count);
        }
        out.flush();
        return total;
    }

    private static final class Candidate {
        final File file;
        final int rank;
        final int depth;
        final long size;

        Candidate(File file, int rank, int depth, long size) {
            this.file = file;
            this.rank = rank;
            this.depth = depth;
            this.size = size;
        }

        static Candidate forFile(File root, File file, long size) {
            String name = file.getName();
            String lower = name.toLowerCase(Locale.US);
            if (!lower.endsWith(".adofai")) return null;
            int rank;
            if ("level.adofai".equals(lower)) rank = 0;
            else if ("main.adofai".equals(lower)) rank = 1;
            else if (isBackupLike(lower)) rank = 8;
            else rank = 2;
            int depth = 0;
            File cursor = file.getParentFile();
            while (cursor != null && !cursor.equals(root)) {
                depth++;
                cursor = cursor.getParentFile();
            }
            return new Candidate(file, rank, depth, size);
        }

        boolean betterThan(Candidate other) {
            if (rank != other.rank) return rank < other.rank;
            if (depth != other.depth) return depth < other.depth;
            if (size != other.size) return size > other.size;
            return file.getAbsolutePath().compareTo(other.file.getAbsolutePath()) < 0;
        }
    }

    private static boolean isBackupLike(String lower) {
        return lower.startsWith("backup") || lower.startsWith("autosave") ||
                lower.startsWith("recovery") || lower.startsWith("~") ||
                lower.contains(" backup") || lower.contains("autosave") ||
                lower.contains("recovery") || lower.contains("_backup") ||
                lower.endsWith(".old.adofai") || lower.endsWith(".bak.adofai");
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteRecursively(child);
        }
        if (!file.delete()) file.deleteOnExit();
    }
}
