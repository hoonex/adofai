package dev.hoonex.adofai.companion;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Read-only, grant-scoped bridge for handing one app-private working chart to
 * the unmodified official ADOFAI process. No official APK or app-private game
 * data is touched.
 */
public final class OfficialChartProvider extends ContentProvider {
    private static final Map<String, String> FILES = new ConcurrentHashMap<String, String>();

    public static Uri publish(android.content.Context context, File file) throws IOException {
        File root = new File(context.getFilesDir(), "working").getCanonicalFile();
        File target = file.getCanonicalFile();
        String rootPrefix = root.getPath() + File.separator;
        if (!target.isFile() || !target.getPath().startsWith(rootPrefix)) {
            throw new SecurityException("Chart is outside the companion working directory");
        }

        String token = UUID.randomUUID().toString();
        FILES.put(token, target.getAbsolutePath());
        return new Uri.Builder()
                .scheme("content")
                .authority(context.getPackageName() + ".charts")
                .appendPath(token)
                .appendPath(target.getName())
                .build();
    }

    @Override public boolean onCreate() {
        return true;
    }

    @Override public String getType(Uri uri) {
        return "application/json";
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (mode == null || (!"r".equals(mode) && !mode.startsWith("r"))) {
            throw new FileNotFoundException("Read-only provider");
        }
        File file = resolve(uri);
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override public Cursor query(Uri uri, String[] projection, String selection,
                                  String[] selectionArgs, String sortOrder) {
        File file;
        try {
            file = resolve(uri);
        } catch (FileNotFoundException error) {
            return null;
        }
        MatrixCursor cursor = new MatrixCursor(new String[] {
                OpenableColumns.DISPLAY_NAME,
                OpenableColumns.SIZE
        });
        cursor.addRow(new Object[] { file.getName(), file.length() });
        return cursor;
    }

    @Override public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    @Override public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    @Override public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("Read-only provider");
    }

    private static File resolve(Uri uri) throws FileNotFoundException {
        List<String> segments = uri == null ? null : uri.getPathSegments();
        if (segments == null || segments.size() < 1) throw new FileNotFoundException("Missing chart token");
        String encoded = FILES.get(segments.get(0));
        if (encoded == null) throw new FileNotFoundException("Expired chart grant");
        File file = new File(encoded);
        if (!file.isFile()) throw new FileNotFoundException("Chart no longer exists");
        return file;
    }
}
