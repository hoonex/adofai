from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "android" / "native-inventory-probe" / "app" / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "nativeprobe" / "IntentCapabilityProbe.java"
MAIN = ROOT / "android" / "native-inventory-probe" / "app" / "src" / "main" / "java" / "dev" / "hoonex" / "adofai" / "nativeprobe" / "MainActivity.java"


class NativeInventoryProbeTests(unittest.TestCase):
    def test_intent_probe_is_read_only_and_target_scoped(self):
        text = PROBE.read_text(encoding="utf-8")
        self.assertIn('intent.setPackage(targetPackage);', text)
        self.assertIn('Intent.ACTION_VIEW', text)
        self.assertEqual(text.count('addViewProbe('), 6)  # helper declaration + five calls
        self.assertIn('file:///sdcard/Download/adofai-intent-probe.adofai', text)
        self.assertIn('content://dev.hoonex.adofai.nativeprobe/adofai-intent-probe.adofai', text)
        self.assertIn('PackageManager.MATCH_DEFAULT_ONLY', text)
        self.assertNotIn('startActivity(', text)
        self.assertNotIn('startActivityForResult(', text)
        self.assertNotIn('deletePackage', text)
        self.assertNotIn('installPackage', text)

    def test_report_v2_contains_external_file_intent_evidence(self):
        text = MAIN.read_text(encoding="utf-8")
        self.assertIn('"adofai-native-inventory-v2"', text)
        self.assertIn('report.put("external_file_intents", IntentCapabilityProbe.build(pm, TARGET_PACKAGE));', text)
        self.assertIn('ACTION_VIEW resolution proves only Android routing capability', text)
        self.assertIn('does not launch ADOFAI', text)


if __name__ == "__main__":
    unittest.main()
