"""Bounded structural tests for the new Android 3.3 editor tile-map slice.

These tests intentionally do not claim gesture or selection success on a phone.
The Android CI build compiles the actual Java classes; device behavior remains
a separate acceptance gate.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "android/mobile-editor-shell/src/com/unity3d/player"
TILE_MAP = SOURCE / "MobileTileMapView.java"
SHELL = SOURCE / "MobileEditorShell.java"
COMPANION = ROOT / "scripts/prepare-editor-harness.sh"
IN_GAME = ROOT / "scripts/build-payload.sh"


class ModernTileSelectionTests(unittest.TestCase):
    def test_same_view_is_included_in_both_android_build_paths(self):
        self.assertTrue(TILE_MAP.is_file())
        self.assertIn('"${EDITOR_TILE_MAP}"', IN_GAME.read_text(encoding="utf-8"))
        self.assertIn(
            'cp "${TILE_MAP}" "${GEN}/MobileTileMapView.java"',
            COMPANION.read_text(encoding="utf-8"),
        )

    def test_selection_is_read_only_and_explicit_action_targets_selected_floor(self):
        view = TILE_MAP.read_text(encoding="utf-8")
        shell = SHELL.read_text(encoding="utf-8")
        self.assertIn("void onFloorSelected(int floor)", shell)
        self.assertIn("selectedFloor = floor;", shell)
        self.assertIn('event.put("floor", selectedFloor);', shell)
        self.assertIn('String[] names = new String[] {"Chart", "Settings", "Events", "Raw", "Tiles"}', shell)
        self.assertIn("selectedFloor = 0;", shell)
        self.assertNotIn("document.put(", view)
        self.assertNotIn("chart.put(", view)

    def test_pointer_arbitration_keeps_pan_and_pinch_distinct_from_taps(self):
        view = TILE_MAP.read_text(encoding="utf-8")
        for marker in (
            "scaleDetector.onTouchEvent(event)",
            "MotionEvent.ACTION_DOWN",
            "MotionEvent.ACTION_POINTER_DOWN",
            "MotionEvent.ACTION_POINTER_UP",
            "MotionEvent.ACTION_MOVE",
            "MotionEvent.ACTION_UP",
            "MotionEvent.ACTION_CANCEL",
            "!moved && !multiTouch",
            "if (listener != null) listener.onFloorSelected(nearest)",
            "ViewConfiguration.get(context).getScaledTouchSlop()",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, view)

    def test_schematic_geometry_rejects_untrusted_large_or_invalid_inputs(self):
        view = TILE_MAP.read_text(encoding="utf-8")
        for marker in (
            "MAX_DISPLAYED_FLOORS",
            "guardFloorCount(path.length())",
            "guardFloorCount(raw.length())",
            "positionOffsets(chart, directions.length)",
            "Invalid PositionTrack",
            "Non-finite PositionTrack",
            "Double.isFinite(x)",
            "double[] raw = angles.clone()",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, view)


if __name__ == "__main__":
    unittest.main()
