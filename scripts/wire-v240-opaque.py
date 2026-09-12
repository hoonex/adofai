#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Build opaque exports completely in app-private storage before any SAF output is opened.
opaque = JAVA / "V240OpaqueEventBridge.java"
replace_once(
    opaque,
    "    private static int rewriteFile(File source, File target, File session,\n",
    """    static File buildRestoredExport(File chart) throws Exception {\n"
    "        if (!hasSession(chart)) return null;\n"
    "        if (chart == null || !chart.isFile()) throw new IOException(\"opaque chart is unavailable\");\n"
    "        File parent = chart.getParentFile();\n"
    "        if (parent == null) throw new IOException(\"opaque chart has no parent directory\");\n"
    "        File export = new File(parent, chart.getName() + \".v240-export-\"\n"
    "                + UUID.randomUUID() + \".tmp\");\n"
    "        try {\n"
    "            try (BufferedOutputStream output = new BufferedOutputStream(\n"
    "                    new FileOutputStream(export, false), IO_BUFFER_BYTES)) {\n"
    "                if (!writeRestoredCopy(chart, output)) {\n"
    "                    throw new IOException(\"opaque session disappeared before export\");\n"
    "                }\n"
    "                output.flush();\n"
    "            }\n"
    "            try (FileOutputStream sync = new FileOutputStream(export, true)) {\n"
    "                sync.getFD().sync();\n"
    "            }\n"
    "            return export;\n"
    "        } catch (Throwable error) {\n"
    "            if (export.exists() && !export.delete()) export.deleteOnExit();\n"
    "            if (error instanceof Exception) throw (Exception) error;\n"
    "            throw new IOException(\"opaque export failed\", error);\n"
    "        }\n"
    "    }\n\n"
    "    static void releaseRestoredExport(File export) {\n"
    "        if (export != null && export.exists() && !export.delete()) export.deleteOnExit();\n"
    "    }\n\n"
    "    private static int rewriteFile(File source, File target, File session,\n""",
)

# 2) Ordinary SAF document bridge: clone opaque state for Save As before binding, prepare direct
#    opened charts before observers start, and always export a restored temporary file.
android = JAVA / "V240AndroidBridge.java"
replace_once(
    android,
    "    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();\n",
    "    private static final Map<String, SaveBinding> SAVE_BINDINGS = new ConcurrentHashMap<String, SaveBinding>();\n"
    "    private static final Map<Integer, String> SAVE_OPAQUE_SOURCES = new ConcurrentHashMap<Integer, String>();\n",
)
replace_once(
    android,
    """    public static int beginSave(String suggestedName, String mime) {\n"
    "        return begin(MODE_SAVE, sanitizeName(emptyToDefault(suggestedName, \"level.adofai\")),\n"
    "                emptyToDefault(mime, \"application/octet-stream\"), null, false);\n"
    "    }\n""",
    """    public static int beginSave(String suggestedName, String mime) {\n"
    "        return beginSave(suggestedName, mime, null);\n"
    "    }\n\n"
    "    public static int beginSave(String suggestedName, String mime, String opaqueSourcePath) {\n"
    "        int id = begin(MODE_SAVE, sanitizeName(emptyToDefault(suggestedName, \"level.adofai\")),\n"
    "                emptyToDefault(mime, \"application/octet-stream\"), null, false);\n"
    "        if (id > 0 && opaqueSourcePath != null && opaqueSourcePath.length() > 0 && isPending(id)) {\n"
    "            SAVE_OPAQUE_SOURCES.put(id, opaqueSourcePath);\n"
    "        }\n"
    "        return id;\n"
    "    }\n""",
)
replace_once(
    android,
    """        result.done.countDown();\n"
    "        return true;\n""",
    """        SAVE_OPAQUE_SOURCES.remove(id);\n"
    "        result.done.countDown();\n"
    "        return true;\n""",
)
replace_once(
    android,
    """            ensurePending(id);\n"
    "            if (workingFiles.isEmpty()) throw new IllegalArgumentException(\"no readable document selected\");\n\n"
    "            // Only a single .adofai level document becomes the editor's writable Save target.\n""",
    """            ensurePending(id);\n"
    "            if (workingFiles.isEmpty()) throw new IllegalArgumentException(\"no readable document selected\");\n\n"
    "            // Hide post-v2.4 event types in the app-private working copy before a writable\n"
    "            // observer can ever see the marker representation. The authoritative SAF file\n"
    "            // remains untouched until syncNow builds a restored private export.\n"
    "            for (File working : workingFiles) {\n"
    "                if (isLevelDocument(working)) V240OpaqueEventBridge.prepareForV240(working);\n"
    "            }\n\n"
    "            // Only a single .adofai level document becomes the editor's writable Save target.\n""",
)
replace_once(
    android,
    """            ensurePending(id);\n"
    "            bindSave(context, uri, working);\n"
    "            bound = true;\n""",
    """            ensurePending(id);\n"
    "            String opaqueSourcePath = SAVE_OPAQUE_SOURCES.remove(id);\n"
    "            if (opaqueSourcePath != null && !V240OpaqueEventBridge.cloneSession(\n"
    "                    new File(opaqueSourcePath), working)) {\n"
    "                throw new IOException(\"opaque event state could not be cloned for Save As\");\n"
    "            }\n"
    "            bindSave(context, uri, working);\n"
    "            bound = true;\n""",
)
replace_once(
    android,
    """    private static void syncNow(SaveBinding binding) throws Exception {\n"
    "        if (!binding.file.isFile()) return;\n"
    "        ContentResolver resolver = binding.context.getContentResolver();\n"
    "        try (InputStream in = new FileInputStream(binding.file);\n"
    "             OutputStream out = requireOutput(resolver, binding.uri)) {\n"
    "            copy(in, out);\n"
    "        }\n"
    "    }\n""",
    """    private static void syncNow(SaveBinding binding) throws Exception {\n"
    "        if (!binding.file.isFile()) return;\n"
    "        File restored = V240OpaqueEventBridge.buildRestoredExport(binding.file);\n"
    "        File source = restored != null ? restored : binding.file;\n"
    "        try {\n"
    "            ContentResolver resolver = binding.context.getContentResolver();\n"
    "            try (InputStream in = new FileInputStream(source);\n"
    "                 OutputStream out = requireOutput(resolver, binding.uri)) {\n"
    "                copy(in, out);\n"
    "            }\n"
    "        } finally {\n"
    "            V240OpaqueEventBridge.releaseRestoredExport(restored);\n"
    "        }\n"
    "    }\n""",
)

# 3) Tree-backed level bridge: prepare while still detached from observer, restore before SAF write.
level = JAVA / "V240LevelFolderBridge.java"
replace_once(
    level,
    """            V240ChartBackport.backportForV240(state.chart);\n"
    "            V240HallLegacyFix.applyIfNeeded(state.chart);\n"
    "            ensurePending(id);\n\n"
    "            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {\n""",
    """            V240ChartBackport.backportForV240(state.chart);\n"
    "            V240HallLegacyFix.applyIfNeeded(state.chart);\n"
    "            V240OpaqueEventBridge.prepareForV240(state.chart);\n"
    "            ensurePending(id);\n\n"
    "            if ((grantFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {\n""",
)
replace_once(
    level,
    """    private static void syncNow(SaveBinding binding) throws Exception {\n"
    "        if (!binding.file.isFile()) return;\n"
    "        ContentResolver resolver = binding.context.getContentResolver();\n"
    "        try (InputStream in = new FileInputStream(binding.file);\n"
    "             OutputStream out = requireOutput(resolver, binding.uri)) {\n"
    "            copy(in, out, -1, Long.MAX_VALUE);\n"
    "        }\n"
    "    }\n""",
    """    private static void syncNow(SaveBinding binding) throws Exception {\n"
    "        if (!binding.file.isFile()) return;\n"
    "        File restored = V240OpaqueEventBridge.buildRestoredExport(binding.file);\n"
    "        File source = restored != null ? restored : binding.file;\n"
    "        try {\n"
    "            ContentResolver resolver = binding.context.getContentResolver();\n"
    "            try (InputStream in = new FileInputStream(source);\n"
    "                 OutputStream out = requireOutput(resolver, binding.uri)) {\n"
    "                copy(in, out, -1, Long.MAX_VALUE);\n"
    "            }\n"
    "        } finally {\n"
    "            V240OpaqueEventBridge.releaseRestoredExport(restored);\n"
    "        }\n"
    "    }\n""",
)

# 4) File-selector orchestration: archives prepare here; direct/tree prepare in their backend before
#    observer installation. Save As carries the current chart sidecar into the new working file.
selector = JAVA / "FileSelector.java"
replace_once(
    selector,
    """        start(V240AndroidBridge.beginSave(suggestedName, mimeForFilename(suggestedName)), false,\n"
    "                BACKEND_DOCUMENT, true);\n""",
    """        start(V240AndroidBridge.beginSave(\n"
    "                suggestedName, mimeForFilename(suggestedName), filePath), false,\n"
    "                BACKEND_DOCUMENT, true);\n""",
)
replace_once(
    selector,
    """                    if (backend == BACKEND_ARCHIVE) {\n"
    "                        V240ChartBackport.backportForV240(chart);\n"
    "                        V240HallLegacyFix.applyIfNeeded(chart);\n"
    "                    }\n"
    "                    // Read-only streaming diagnostics run after any private-copy compatibility\n"
    "                    // rewrite so they describe the exact chart the legacy parser will receive.\n"
    "                    // This scan never mutates the chart and fails open on malformed/oversized input.\n"
    "                    V240ChartCompatibilityScanner.scanAndLog(chart);\n""",
    """                    if (backend == BACKEND_ARCHIVE) {\n"
    "                        V240ChartBackport.backportForV240(chart);\n"
    "                        V240HallLegacyFix.applyIfNeeded(chart);\n"
    "                        // Archives have no writable SAF binding at this point, so prepare their\n"
    "                        // preserve-only placeholders here. Direct/tree backends already prepare\n"
    "                        // before installing their save observer.\n"
    "                        V240OpaqueEventBridge.prepareForV240(chart);\n"
    "                    }\n""",
)

# 5) DEX class-presence contract.
build = ROOT / "scripts/build-v240-fixed-java.sh"
replace_once(
    build,
    """  'Lcom/unity3d/player/V240ChartBackport;' \\\n  'Lcom/unity3d/player/V240HallLegacyFix;' \\\n""",
    """  'Lcom/unity3d/player/V240ChartBackport;' \\\n  'Lcom/unity3d/player/V240ChartCompatibilityScanner;' \\\n  'Lcom/unity3d/player/V240OpaqueEventBridge;' \\\n  'Lcom/unity3d/player/V240HallLegacyFix;' \\\n""",
)

# 6) CI gates: new preservation contract must run in both Java runtime and final-release paths.
runtime = ROOT / ".github/workflows/v240-fixed-runtime.yml"
replace_once(
    runtime,
    "      - 'tests/test_v240_performance_runtime.py'\n",
    "      - 'tests/test_v240_performance_runtime.py'\n      - 'tests/test_v240_opaque_event_bridge.py'\n",
)
replace_once(
    runtime,
    """      - name: Verify high-refresh safety contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n""",
    """      - name: Verify high-refresh safety contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n      - name: Verify opaque event preservation contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v\n""",
)

final = ROOT / ".github/workflows/v240-final-release.yml"
replace_once(
    final,
    "      - 'tests/test_v240_performance_runtime.py'\n",
    "      - 'tests/test_v240_performance_runtime.py'\n      - 'tests/test_v240_opaque_event_bridge.py'\n",
)
replace_once(
    final,
    """      - name: Verify mobile performance and storage safety contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n""",
    """      - name: Verify mobile performance and storage safety contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n\n      - name: Verify opaque event preservation contract\n        run: python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v\n""",
)
replace_once(
    final,
    """          python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n          bash scripts/build-v240-fixed-java.sh .work/v240-final-java\n""",
    """          python3 -m unittest discover -s tests -p 'test_v240_performance_runtime.py'\n          python3 -m unittest discover -s tests -p 'test_v240_opaque_event_bridge.py' -v\n          bash scripts/build-v240-fixed-java.sh .work/v240-final-java\n""",
)

print("opaque bridge wiring patch applied")
