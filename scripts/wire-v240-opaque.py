#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before_once(path: Path, marker: str, addition: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(marker)
    if count != 1:
        raise SystemExit(f"{path}: expected one insert marker, found {count}: {marker[:100]!r}")
    path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


# Build the restored chart completely in private storage before opening/truncating a SAF target.
opaque = JAVA / "V240OpaqueEventBridge.java"
insert_before_once(
    opaque,
    "    private static int rewriteFile(File source, File target, File session,\n",
    '''    static File buildRestoredExport(File chart) throws Exception {
        if (!hasSession(chart)) return null;
        if (chart == null || !chart.isFile()) throw new IOException("opaque chart is unavailable");
        File parent = chart.getParentFile();
        if (parent == null) throw new IOException("opaque chart has no parent directory");
        File export = new File(parent, chart.getName() + ".v240-export-"
                + UUID.randomUUID() + ".tmp");
        try {
            try (BufferedOutputStream output = new BufferedOutputStream(
                    new FileOutputStream(export, false), IO_BUFFER_BYTES)) {
                if (!writeRestoredCopy(chart, output)) {
                    throw new IOException("opaque session disappeared before export");
                }
                output.flush();
            }
            try (FileOutputStream sync = new FileOutputStream(export, true)) {
                sync.getFD().sync();
            }
            return export;
        } catch (Throwable error) {
            if (export.exists() && !export.delete()) export.deleteOnExit();
            if (error instanceof Exception) throw (Exception) error;
            throw new IOException("opaque export failed", error);
        }
    }

    static void releaseRestoredExport(File export) {
        if (export != null && export.exists() && !export.delete()) export.deleteOnExit();
    }

''',
)

# Ordinary SAF document bridge.
android = JAVA / "V240AndroidBridge.java"
replace_once(
    android,
    '    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();\n',
    '    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();\n'
    '    private static final Map<Integer, String> SAVE_OPAQUE_SOURCES = new ConcurrentHashMap<Integer, String>();\n',
)
replace_once(
    android,
    '''    public static int beginSave(String suggestedName, String mime) {
        return begin(MODE_SAVE, sanitizeName(emptyToDefault(suggestedName, "level.adofai")),
                emptyToDefault(mime, "application/octet-stream"), null, false);
    }
''',
    '''    public static int beginSave(String suggestedName, String mime) {
        return beginSave(suggestedName, mime, null);
    }

    public static int beginSave(String suggestedName, String mime, String opaqueSourcePath) {
        int id = begin(MODE_SAVE, sanitizeName(emptyToDefault(suggestedName, "level.adofai")),
                emptyToDefault(mime, "application/octet-stream"), null, false);
        if (id > 0 && opaqueSourcePath != null && opaqueSourcePath.length() > 0 && isPending(id)) {
            SAVE_OPAQUE_SOURCES.put(id, opaqueSourcePath);
        }
        return id;
    }
''',
)
replace_once(
    android,
    '''        result.done.countDown();
        return true;
''',
    '''        SAVE_OPAQUE_SOURCES.remove(id);
        result.done.countDown();
        return true;
''',
)
replace_once(
    android,
    '''            ensurePending(id);
            if (workingFiles.isEmpty()) throw new IllegalArgumentException("no readable document selected");

            // Only a single .adofai level document becomes the editor's writable Save target.
''',
    '''            ensurePending(id);
            if (workingFiles.isEmpty()) throw new IllegalArgumentException("no readable document selected");

            // Prepare unknown post-v2.4 events while this is still an app-private copy and before
            // any writable FileObserver is attached. External SAF content remains authoritative.
            for (File working : workingFiles) {
                if (isLevelDocument(working)) V240OpaqueEventBridge.prepareForV240(working);
            }

            // Only a single .adofai level document becomes the editor's writable Save target.
''',
)
replace_once(
    android,
    '''            ensurePending(id);
            bindSave(context, uri, working);
            bound = true;
''',
    '''            ensurePending(id);
            String opaqueSourcePath = SAVE_OPAQUE_SOURCES.remove(id);
            if (opaqueSourcePath != null && !V240OpaqueEventBridge.cloneSession(
                    new File(opaqueSourcePath), working)) {
                throw new IOException("opaque event state could not be cloned for Save As");
            }
            bindSave(context, uri, working);
            bound = true;
''',
)
replace_once(
    android,
    '''    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        ContentResolver resolver = binding.context.getContentResolver();
        try (InputStream in = new FileInputStream(binding.file);
             OutputStream out = requireOutput(resolver, binding.uri)) {
            copy(in, out);
        }
    }
''',
    '''    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        File restored = V240OpaqueEventBridge.buildRestoredExport(binding.file);
        File source = restored != null ? restored : binding.file;
        try {
            ContentResolver resolver = binding.context.getContentResolver();
            try (InputStream in = new FileInputStream(source);
                 OutputStream out = requireOutput(resolver, binding.uri)) {
                copy(in, out);
            }
        } finally {
            V240OpaqueEventBridge.releaseRestoredExport(restored);
        }
    }
''',
)

# Tree-backed map opener: same preserve-before-observer and restored-export rules.
level = JAVA / "V240LevelFolderBridge.java"
replace_once(
    level,
    '''            V240ChartBackport.backportForV240(state.chart);
            V240HallLegacyFix.applyIfNeeded(state.chart);
            ensurePending(id);

            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
''',
    '''            V240ChartBackport.backportForV240(state.chart);
            V240HallLegacyFix.applyIfNeeded(state.chart);
            V240OpaqueEventBridge.prepareForV240(state.chart);
            ensurePending(id);

            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
''',
)
replace_once(
    level,
    '''    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        ContentResolver resolver = binding.context.getContentResolver();
        try (InputStream in = new FileInputStream(binding.file);
             OutputStream out = requireOutput(resolver, binding.uri)) {
            copy(in, out, -1, Long.MAX_VALUE);
        }
    }
''',
    '''    private static void syncNow(SaveBinding binding) throws Exception {
        if (!binding.file.isFile()) return;
        File restored = V240OpaqueEventBridge.buildRestoredExport(binding.file);
        File source = restored != null ? restored : binding.file;
        try {
            ContentResolver resolver = binding.context.getContentResolver();
            try (InputStream in = new FileInputStream(source);
                 OutputStream out = requireOutput(resolver, binding.uri)) {
                copy(in, out, -1, Long.MAX_VALUE);
            }
        } finally {
            V240OpaqueEventBridge.releaseRestoredExport(restored);
        }
    }
''',
)

# File-selector orchestration: archives prepare here; direct/tree prepare inside their backends.
selector = JAVA / "FileSelector.java"
replace_once(
    selector,
    '''        start(V240AndroidBridge.beginSave(suggestedName, mimeForFilename(suggestedName)), false,
                BACKEND_DOCUMENT, true);
''',
    '''        start(V240AndroidBridge.beginSave(
                suggestedName, mimeForFilename(suggestedName), filePath), false,
                BACKEND_DOCUMENT, true);
''',
)
replace_once(
    selector,
    '''                    if (backend == BACKEND_ARCHIVE) {
                        V240ChartBackport.backportForV240(chart);
                        V240HallLegacyFix.applyIfNeeded(chart);
                    }
                    // Read-only streaming diagnostics run after any private-copy compatibility
                    // rewrite so they describe the exact chart the legacy parser will receive.
                    // This scan never mutates the chart and fails open on malformed/oversized input.
                    V240ChartCompatibilityScanner.scanAndLog(chart);
''',
    '''                    if (backend == BACKEND_ARCHIVE) {
                        V240ChartBackport.backportForV240(chart);
                        V240HallLegacyFix.applyIfNeeded(chart);
                        // Archive imports have no writable SAF binding yet. Direct documents and
                        // tree-backed levels prepare inside their backend before observer install.
                        V240OpaqueEventBridge.prepareForV240(chart);
                    }
''',
)

# DEX class presence.
build = ROOT / "scripts/build-v240-fixed-java.sh"
needle = "  'Lcom/unity3d/player/V240ChartBackport;' \\\n"
insert = needle + "  'Lcom/unity3d/player/V240ChartCompatibilityScanner;' \\\n  'Lcom/unity3d/player/V240OpaqueEventBridge;' \\\n"
replace_once(build, needle, insert)

# Runtime CI.
runtime = ROOT / ".github/workflows/v240-fixed-runtime.yml"
replace_once(
    runtime,
    "      - 'tests/test_v240_performance_runtime.py'\n",
    "      - 'tests/test_v240_performance_runtime.py'\n      - 'tests/test_v240_opaque_event_bridge.py'\n",
)
replace_once(
    runtime,
    '''      - name: Verify high-refresh safety contract
        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'
''',
    '''      - name: Verify high-refresh safety contract
        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'
      - name: Verify opaque event preservation contract
        run: python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v
''',
)

# Final release CI.
final = ROOT / ".github/workflows/v240-final-release.yml"
replace_once(
    final,
    "      - 'tests/test_v240_performance_runtime.py'\n",
    "      - 'tests/test_v240_performance_runtime.py'\n      - 'tests/test_v240_opaque_event_bridge.py'\n",
)
replace_once(
    final,
    '''      - name: Verify mobile performance and storage safety contract
        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'
''',
    '''      - name: Verify mobile performance and storage safety contract
        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'

      - name: Verify opaque event preservation contract
        run: python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v
''',
)
replace_once(
    final,
    '''          python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'
          bash scripts/build-v240-fixed-java.sh .work/v240-final-java
''',
    '''          python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'
          python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v
          bash scripts/build-v240-fixed-java.sh .work/v240-final-java
''',
)

print("opaque bridge wiring patch applied")
