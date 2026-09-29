from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "android/v240-dynamic-runtime/java/dev/hoonex/adofai/v240/dynamic/RuntimeEntry.java"
WINDOW = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240WindowCompat.java"


class RuntimeR25Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.window = WINDOW.read_text(encoding="utf-8")

    def test_hot_runtime_owns_full_width_policy_after_bootstrap(self):
        s = self.entry
        for marker in (
            "WINDOW_GUARD_REVISION = 25",
            "installWindowLifecycleGuard(app)",
            "Application.ActivityLifecycleCallbacks",
            "onActivityResumed(Activity activity)",
            "installViewportLayoutGuard(activity)",
            "View.OnLayoutChangeListener",
            "normalizeLegacyWindowViewport(current)",
            "LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES",
            "window viewport lifecycle guard r",
        ):
            self.assertIn(marker, s)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", s)

    def test_window_guard_is_event_driven_and_bounded(self):
        s = self.entry
        self.assertIn("scheduleLegacyWindowNormalization()", s)
        for delay in ("1750L", "3000L", "6500L"):
            self.assertIn(delay, s)
        self.assertNotIn("while (", s)
        self.assertNotIn("while(true)", s)
        self.assertNotIn("postDelayed(this", s)

    def test_parent_and_hot_runtime_converge_on_same_cutout_policy(self):
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.window)
        self.assertNotIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_NEVER", self.window)
        self.assertIn("LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES", self.entry)

    def test_layout_guard_does_not_hold_destroyed_activity(self):
        s = self.entry
        self.assertIn("clearViewportLayoutGuard(activity)", s)
        self.assertIn("removeOnLayoutChangeListener(viewportLayoutListener)", s)
        self.assertIn("guardedViewportDecor = null", s)
        self.assertIn("Activity current = currentActivity()", s)

    def test_graceful_stop_reuses_existing_lifecycle_guard_without_polling(self):
        s = self.entry
        self.assertIn("onActivityStopped(Activity activity)", s)
        self.assertIn("markParentRuntimeHealthyOnGracefulStop(activity)", s)
        self.assertIn('getDeclaredMethod("markHealthy", File.class)', s)
        self.assertIn('"com.unity3d.player.UnityPlayerActivity".equals(type.getName())', s)
        self.assertNotIn("boot health while", s)

    def test_layout_listener_cannot_self_trigger_forever(self):
        s = self.entry
        self.assertIn("boolean changed = false", s)
        self.assertIn("changed = true", s)
        self.assertIn("if (changed)", s)
        normalize = s[s.index("private static void normalizeLegacyWindowViewport(Activity activity)"):]
        normalize = normalize[:normalize.index("private static void scheduleLegacyGearRelocation")]
        self.assertLess(normalize.index("if (changed)"), normalize.index("decor.requestLayout()"))


if __name__ == "__main__":
    unittest.main()
