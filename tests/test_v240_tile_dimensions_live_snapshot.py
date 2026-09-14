import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "android/v240-fixed-runtime/native/V240CompatibilityReport.cpp"
EVENT_JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240EventCompat.java"
NATIVE_BUILD = ROOT / "scripts/build-v240-fixed-native.sh"
FIXTURE = ROOT / "fixtures/device/ADOFAI_2.4_TILEDIMENSIONS_OBSERVATION_TEST.adofai"


class V240TileDimensionsLiveSnapshotContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = NATIVE.read_text(encoding="utf-8")
        cls.event_java = EVENT_JAVA.read_text(encoding="utf-8")
        cls.native_build = NATIVE_BUILD.read_text(encoding="utf-8")
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_snapshot_path_is_evidence_only_and_non_mutating(self):
        for forbidden in (
            ".Call(",
            ".Set(",
            "BasicHook",
            "CreateNewObject",
            ".GetMethod(",
            ".GetProperty(",
            "ApplyEventsToFloors",
        ):
            self.assertNotIn(forbidden, self.native)
        self.assertIn("TileDimensions.liveSnapshot.mode=read-only", self.native)
        self.assertIn("TileDimensions.liveSnapshot.mutatesRuntime=0", self.native)
        self.assertIn("TileDimensions.liveSnapshot.conversionAssumption=none", self.native)

    def test_metadata_surface_is_prepared_only_from_bnm_loaded_callback(self):
        self.assertIn("void PrepareTileDimensionsSnapshotSurface()", self.native)
        self.assertIn(
            "Loading::AddOnLoadedEvent(PrepareTileDimensionsSnapshotSurface);",
            self.native,
        )
        self.assertIn("std::once_flag g_snapshotRegisterOnce;", self.native)
        self.assertIn("nativeRegisterTileDimensionsSnapshot", self.native)

        build_start = self.native.index("std::string BuildTileDimensionsLiveSnapshot()")
        build_end = self.native.index("std::size_t CopyText", build_start)
        build = self.native[build_start:build_end]
        for lazy_metadata_lookup in (
            'Class scnGame(',
            'Class scrLevelMaker(',
            'Class scrFloor(',
            '.GetField(',
            '.GetGeneric(',
            'AttachIl2Cpp',
        ):
            self.assertNotIn(lazy_metadata_lookup, build)

    def test_snapshot_surface_requires_exact_field_storage_and_types(self):
        expected = (
            'Class scnGame("", "scnGame")',
            'Class scrLevelMaker("", "scrLevelMaker")',
            'Class scrFloor("", "scrFloor")',
            'Class genericList("System.Collections.Generic", "List`1")',
            'scnGame.GetField("instance")',
            'scnGame.GetField("levelMaker")',
            'scrLevelMaker.GetField("listFloors")',
            'scrFloor.GetField("lengthMult")',
            'scrFloor.GetField("widthMult")',
            'scrFloor.GetField("seqID")',
            'IsReadableStaticField(scnGameInstance)',
            'IsReadableInstanceField(levelMaker)',
            'IsReadableInstanceField(listFloors)',
            'SameManagedType(listFloors.GetType(), listFloor)',
            'SameManagedType(lengthMult.GetType(), floatClass)',
            'SameManagedType(widthMult.GetType(), floatClass)',
        )
        for marker in expected:
            self.assertIn(marker, self.native)

    def test_live_read_validates_runtime_object_and_list_layout_before_sampling(self):
        expected = (
            'game->klass != g_scnGameClass',
            'levelMaker->klass != g_scrLevelMakerClass',
            'floors->klass != g_listFloorClass',
            'floor->klass != g_scrFloorClass',
            'const int floorCount = floors->GetSize();',
            'const int versionBefore = floors->GetVersion();',
            'auto* itemsBefore = floors->items;',
            'capacity < static_cast<std::size_t>(floorCount)',
            'floorCount > kMaxObservedFloorCount',
            'capacity > kMaxObservedListCapacity',
            'const int sampleCount = std::min(floorCount, kMaxSampleFloors);',
            'const int versionAfter = floors->GetVersion();',
            'itemsAfter != itemsBefore',
            'gameAfter != game',
            'levelMakerAfter != levelMaker',
            'floor_state_changed_during_snapshot',
        )
        for marker in expected:
            self.assertIn(marker, self.native)
        self.assertIn("constexpr int kMaxSampleFloors = 64;", self.native)

    def test_snapshot_reads_existing_multipliers_without_inventing_percentage_semantics(self):
        self.assertIn('Field<float> g_lengthMult;', self.native)
        self.assertIn('Field<float> g_widthMult;', self.native)
        self.assertIn('const float lengthMult = lengthMultField[floor].Get();', self.native)
        self.assertIn('const float widthMult = widthMultField[floor].Get();', self.native)
        self.assertNotIn("/ 100", self.native)
        self.assertNotIn("/100", self.native)
        self.assertNotIn("0.01f", self.native)
        self.assertNotIn("0.01F", self.native)

    def test_java_registers_diagnostic_independently_from_active_event_runtime(self):
        self.assertIn("nativeRegister();", self.event_java)
        self.assertIn("nativeRegisterTileDimensionsSnapshot();", self.event_java)
        self.assertIn("private static native void nativeRegisterTileDimensionsSnapshot();", self.event_java)
        self.assertLess(
            self.event_java.index("nativeRegister();"),
            self.event_java.index("nativeRegisterTileDimensionsSnapshot();"),
        )
        self.assertIn("diagnostic failure can never disable the proven SetFrameRate path", self.event_java)

    def test_clipboard_report_merges_cached_abi_and_live_floor_snapshot(self):
        self.assertIn("V240CopyPostV240CompatibilityReport", self.native)
        self.assertIn("V240CopyTileDimensionsLiveSnapshot", self.native)
        self.assertIn("ReadCopiedText(V240CopyPostV240CompatibilityReport)", self.native)
        self.assertIn("ReadCopiedText(V240CopyTileDimensionsLiveSnapshot)", self.native)
        self.assertIn("Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport", self.native)

    def test_native_build_requires_snapshot_exports_and_evidence_strings(self):
        self.assertIn("V240CompatibilityReport.cpp", self.native_build)
        self.assertIn(
            "Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport",
            self.native_build,
        )
        self.assertIn(
            "Java_com_unity3d_player_V240EventCompat_nativeRegisterTileDimensionsSnapshot",
            self.native_build,
        )
        self.assertIn("V240CopyTileDimensionsLiveSnapshot", self.native_build)
        self.assertIn("TileDimensions.liveSnapshot.status=ready", self.native_build)
        self.assertIn("TileDimensions.liveSnapshot.mutatesRuntime=0", self.native_build)
        self.assertIn("TileDimensions.liveSnapshot.conversionAssumption=none", self.native_build)

    def test_device_fixture_is_observational_and_keeps_serialized_values_verbatim(self):
        actions = [
            action for action in self.fixture["actions"]
            if action.get("eventType") == "TileDimensions"
        ]
        self.assertEqual(
            [
                {"floor": 1, "eventType": "TileDimensions", "width": 200, "length": 50},
                {"floor": 4, "eventType": "TileDimensions", "width": 125, "length": 175},
            ],
            actions,
        )
        self.assertNotIn("expectedLengthMult", self.fixture)
        self.assertNotIn("expectedWidthMult", self.fixture)


if __name__ == "__main__":
    unittest.main()
