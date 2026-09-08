from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "android" / "native-inventory-probe" / "app" / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "nativeprobe" / "IntentCapabilityProbe.java"
MAIN = ROOT / "android" / "native-inventory-probe" / "app" / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "nativeprobe" / "MainActivity.java"
HANDOFF_FIXTURE = ROOT / "tests" / "fixtures" / "explicit-handoff-probe.adofai"


class NativeInventoryProbeTests(unittest.TestCase):
    def test_intent_probe_is_read_only_and_target_scoped(self):
        text = PROBE.read_text(encoding="utf-8")
        self.assertIn('intent.setPackage(targetPackage);', text)
        self.assertIn('Intent.ACTION_VIEW', text)
        self.assertIn('Intent.ACTION_SEND', text)
        self.assertIn('PackageManager.GET_ACTIVITIES', text)
        self.assertIn('pm.getLaunchIntentForPackage(targetPackage)', text)
        self.assertIn('exported_activities', text)
        self.assertIn('file:///sdcard/Download/adofai-intent-probe.adofai', text)
        self.assertIn('content://dev.hoonex.adofai.nativeprobe/adofai-intent-probe.adofai', text)
        self.assertIn('https://example.invalid/adofai-intent-probe.adofai', text)
        self.assertIn('view-https-adofai', text)
        self.assertIn('send-stream-application-json', text)
        self.assertIn('send-url-text-plain', text)
        self.assertIn('PackageManager.MATCH_DEFAULT_ONLY', text)
        self.assertNotIn('startActivity(', text)
        self.assertNotIn('startActivityForResult(', text)
        self.assertNotIn('deletePackage', text)
        self.assertNotIn('installPackage', text)

    def test_report_v2_contains_external_file_intent_evidence(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn('"adofai-native-inventory-v2"', text)
        self.assertIn('report.put("external_file_intents", IntentCapabilityProbe.build(pm, TARGET_PACKAGE));', text)
        self.assertIn('ACTION_VIEW/SEND resolution proves only Android routing capability', text)
        self.assertIn('Generating this inventory report does not launch ADOFAI', text)

    def test_explicit_handoff_is_opt_in_exact_target_and_non_mutating(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn('REQUEST_HANDOFF_FILE', text)
        self.assertIn('assertExactTarget();', text)
        self.assertIn('new ComponentName(TARGET_PACKAGE, TARGET_ACTIVITY)', text)
        self.assertIn('TARGET_ACTIVITY = "com.unity3d.player.UnityPlayerActivity"', text)
        self.assertIn('Intent.FLAG_GRANT_READ_URI_PERMISSION', text)
        self.assertIn('ClipData.newRawUri("ADOFAI chart", uri)', text)
        self.assertIn('Intent.EXTRA_STREAM', text)
        self.assertIn('REMOTE_PROBE_URL', text)
        self.assertIn('explicit-handoff-probe.adofai', text)
        self.assertNotIn('deletePackage', text)
        self.assertNotIn('installPackage', text)
        self.assertNotIn('pm clear', text)

    def test_remote_handoff_fixture_is_strict_json(self):
        parsed = json.loads(HANDOFF_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(parsed["settings"]["version"], 15)
        self.assertEqual(parsed["settings"]["songFilename"], "")
        self.assertGreaterEqual(len(parsed["angleData"]), 2)


if __name__ == "__main__":
    unittest.main()
