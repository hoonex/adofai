import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240Bootstrap.java"
UPDATER = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240RuntimeUpdater.java"
REPORT = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240CompatibilityReport.java"
MANIFEST_PATCHER = ROOT / "android/game-patcher/app/src/main/java/dev/hoonex/adofai/gamepatcher/ManifestStoragePatcher.java"
DYNAMIC_ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
JAVA_BUILD = ROOT / "scripts/build-v240-fixed-java.sh"
DYNAMIC_BUILD = ROOT / "scripts/build-v240-dynamic-runtime.sh"


class V240RuntimeUpdaterContractTest(unittest.TestCase):
    def text(self, path):
        return path.read_text(encoding="utf-8")

    def test_bootstrap_never_hard_loads_embedded_native(self):
        source = self.text(BOOTSTRAP)
        self.assertIn("V240RuntimeUpdater.startIfReady();", source)
        self.assertNotIn('System.loadLibrary("v240fix")', source)
        self.assertNotIn("V240EventCompat.initialize();", source)
        self.assertIn("Java recovery mode", source)

    def test_updater_uses_only_app_private_code_cache(self):
        source = self.text(UPDATER)
        self.assertIn('getCodeCacheDir(), "v240-runtime"', source)
        self.assertNotIn("getExternalStorage", source)
        self.assertNotIn("Environment.", source)
        self.assertNotIn("/sdcard", source)

    def test_update_is_hash_gated_bounded_and_https_only(self):
        source = self.text(UPDATER)
        for marker in (
            "MAX_MANIFEST_BYTES",
            "MAX_BUNDLE_BYTES",
            "MAX_ENTRY_BYTES",
            "bundleSha256",
            "nativeSha256",
            "dexSha256",
            "MessageDigest.getInstance(\"SHA-256\")",
            '"https".equalsIgnoreCase(url.getProtocol())',
            "requireReleaseUrl(bundleUrl)",
            "too many redirects",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("setInstanceFollowRedirects(true)", source)

    def test_download_never_executes_in_same_process(self):
        source = self.text(UPDATER)
        check = source[source.index("private static void checkForUpdate"):]
        self.assertIn("downloaded-restart-required", check)
        self.assertNotIn("System.load(", check)
        self.assertNotIn("DexClassLoader(", check)

    def test_crash_loop_has_pending_marker_quarantine_and_rollback(self):
        source = self.text(UPDATER)
        for marker in (
            '"boot.pending"',
            "recoverInterruptedBoot()",
            "quarantineVersion(active, \"boot_crash\")",
            "restorePreviousPointer()",
            'new File(rootDir, "previous")',
            "HEALTH_DELAY_MS",
        ):
            self.assertIn(marker, source)

    def test_dynamic_code_is_read_only_before_loading(self):
        source = self.text(UPDATER)
        native_ro = source.index("nativeLib.setReadOnly()")
        dex_ro = source.index("runtimeDex.setReadOnly()")
        native_load = source.index("System.load(nativeLib.getAbsolutePath())")
        dex_load = source.index("new DexClassLoader(")
        self.assertLess(native_ro, native_load)
        self.assertLess(dex_ro, dex_load)

    def test_bundle_has_exact_allowlisted_entries(self):
        source = self.text(UPDATER)
        self.assertIn('if ("libv240fix.so".equals(name))', source)
        self.assertIn('else if ("runtime.dex".equals(name))', source)
        self.assertIn('throw new IllegalStateException("unexpected bundle entry: " + name)', source)

    def test_rollout_can_be_disabled_server_side(self):
        source = self.text(UPDATER)
        self.assertIn('manifest.optBoolean("rollout", false)', source)
        self.assertIn('channelState = "rollout-disabled"', source)

    def test_diagnostic_survives_without_native_library(self):
        source = self.text(REPORT)
        self.assertIn("V240RuntimeUpdater.diagnosticText()", source)
        self.assertIn('report.append("nativeProbe=not-loaded\\n")', source)

    def test_exact_v240_manifest_gets_only_ordinary_network_permission(self):
        source = self.text(MANIFEST_PATCHER)
        self.assertIn('INTERNET = "android.permission.INTERNET"', source)
        v240 = source[source.index("static void patchV240"):source.index("static void assertPermissions")]
        self.assertIn("manifest.addUsesPermission(INTERNET);", v240)
        self.assertNotIn("MANAGE_EXTERNAL_STORAGE", v240)
        self.assertIn("manifest.getUsesPermission(INTERNET)", source)

    def test_hot_swap_entry_is_separate_from_embedded_parent_dex(self):
        dynamic = self.text(DYNAMIC_ENTRY)
        java_build = self.text(JAVA_BUILD)
        dynamic_build = self.text(DYNAMIC_BUILD)
        self.assertIn("package dev.hoonex.adofai.v240.dynamic;", dynamic)
        self.assertIn("public static void install(Context context)", dynamic)
        self.assertIn("RuntimeEntry", dynamic_build)
        self.assertIn("hot-swappable entrypoint must never be embedded", java_build)
        self.assertIn("Ldev/hoonex/adofai/v240/dynamic/RuntimeEntry;", java_build)


if __name__ == "__main__":
    unittest.main()
