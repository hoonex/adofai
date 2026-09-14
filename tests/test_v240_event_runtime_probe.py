from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOUCH_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240TouchAssist.cpp"
EVENT_NATIVE = ROOT / "android/v240-fixed-runtime/native/V240EventCompat.cpp"
BOOTSTRAP = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240Bootstrap.java"
EVENT_JAVA = ROOT / "android/v240-fixed-runtime/java/com/unity3d/player/V240EventCompat.java"
NATIVE_BUILD = ROOT / "scripts/build-v240-fixed-native.sh"
JAVA_BUILD = ROOT / "scripts/build-v240-fixed-java.sh"


class V240EventRuntimeProbeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.touch_source = TOUCH_NATIVE.read_text(encoding="utf-8")
        start = cls.touch_source.index("void LogPostV240CompatibilitySurface()")
        end = cls.touch_source.index("bool RaycastAt(", start)
        cls.probe = cls.touch_source[start:end]
        cls.event_source = EVENT_NATIVE.read_text(encoding="utf-8")
        cls.bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
        cls.event_java = EVENT_JAVA.read_text(encoding="utf-8")
        cls.native_build = NATIVE_BUILD.read_text(encoding="utf-8")
        cls.java_build = JAVA_BUILD.read_text(encoding="utf-8")

    def test_read_only_probe_covers_lossless_unknown_event_bridge_surfaces(self):
        expected = (
            'Class levelData("ADOFAI", "LevelData")',
            'Class levelEvent("ADOFAI", "LevelEvent")',
            'Class levelEventInfo("ADOFAI", "LevelEventInfo")',
            'Class gcs("", "GCS")',
            'levelData.GetMethod("Decode")',
            'levelEvent.GetMethod("Decode")',
            'levelEvent.GetMethod("Encode")',
            'gcs.GetField("levelEventsInfo")',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_read_only_probe_covers_frame_event_scheduler_candidates(self):
        self.assertIn('Class scnGame("", "scnGame")', self.probe)
        self.assertIn('scnGame.GetMethod("ApplyEvent")', self.probe)
        self.assertIn('Class scrCamera("", "scrCamera")', self.probe)
        self.assertIn('"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}', self.probe)
        self.assertIn('"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<float>()}', self.probe)
        self.assertIn('setCustomFrameRateTyped = setCustomFrameRateBoolInt || setCustomFrameRateBoolFloat', self.probe)

    def test_read_only_probe_requires_exact_call_method_abi(self):
        expected = (
            'Class scrPlanet("", "scrPlanet")',
            'Class ffxCallMethod("", "ffxCallMethod")',
            'ffxCallMethod.GetField("methodName").IsValid()',
            '"Decode", {levelEvent.GetCompileTimeClass()}',
            '"StartEffect", {scrPlanet.GetCompileTimeClass()}',
            'callMethodSchedulerSurface = ffxCallMethod && callMethodNameField',
            '&& callMethodDecode && callMethodStartEffect',
        )
        for marker in expected:
            self.assertIn(marker, self.probe)

    def test_read_only_probe_remains_non_mutating(self):
        self.assertNotIn("BasicHook", self.probe)
        self.assertNotIn("CreateNewObject", self.probe)
        self.assertNotIn(".Call(", self.probe)
        self.assertNotIn(".Set(", self.probe)
        self.assertNotIn("operator[]", self.probe)

    def test_execution_backport_revalidates_exact_managed_types(self):
        expected = (
            'Class levelEvent("ADOFAI", "LevelEvent")',
            'Class scrPlanet("", "scrPlanet")',
            'Class ffxCallMethod("", "ffxCallMethod")',
            'Class scrCamera("", "scrCamera")',
            'ffxCallMethod.GetField("methodName")',
            '"Decode", {levelEvent.GetCompileTimeClass()}',
            '"StartEffect", {scrPlanet.GetCompileTimeClass()}',
            'scrCamera.GetField("instance")',
            'scrCamera.GetProperty("instance")',
            '"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<int>()}',
            '"SetCustomFrameRate", {Defaults::Get<bool>(), Defaults::Get<float>()}',
            'SameManagedType(callMethodName.GetType(), stringClass)',
            'SameManagedType(cameraInstanceField.GetType(), scrCamera)',
            'SameManagedType(cameraInstanceProperty.GetType(), scrCamera)',
        )
        for marker in expected:
            self.assertIn(marker, self.event_source)

    def test_camera_singleton_accepts_field_or_property_but_not_untyped_surface(self):
        self.assertIn("cameraInstanceFieldTyped || cameraInstancePropertyTyped", self.event_source)
        self.assertIn("if (cameraInstanceFieldTyped) g_cameraInstanceField = cameraInstanceField;", self.event_source)
        self.assertIn("if (cameraInstancePropertyTyped) g_cameraInstanceProperty = cameraInstanceProperty;", self.event_source)
        self.assertIn("instanceField=%d instanceProperty=%d", self.event_source)
        self.assertNotIn("cameraInstance.IsValid()", self.event_source)

    def test_tile_dimensions_probe_requires_exact_float_floor_fields(self):
        expected = (
            'Class scrFloor("", "scrFloor")',
            'scrFloor.GetField("lengthMult")',
            'scrFloor.GetField("widthMult")',
            'const Class floatClass = Defaults::Get<float>().ToClass();',
            'SameManagedType(floorLengthMult.GetType(), floatClass)',
            'SameManagedType(floorWidthMult.GetType(), floatClass)',
            'tileDimensionsSurface = scrFloor',
            'V240: TileDimensions ABI scrFloor=%d lengthMultFloat=%d widthMultFloat=%d compatible=%d',
        )
        for marker in expected:
            self.assertIn(marker, self.event_source)

        start = self.event_source.index('auto floorLengthMult =')
        end = self.event_source.index('LOGD("V240: TileDimensions ABI', start)
        probe = self.event_source[start:end]
        self.assertNotIn(".Set(", probe)
        self.assertNotIn("BasicHook", probe)
        self.assertNotIn("CreateNewObject", probe)

    def test_tile_dimensions_probe_covers_floor_state_application_candidates(self):
        expected = (
            'Class levelData("ADOFAI", "LevelData")',
            'Class levelEvent("ADOFAI", "LevelEvent")',
            'Class scnGame("", "scnGame")',
            'Class scrLevelMaker("", "scrLevelMaker")',
            'Class genericList("System.Collections.Generic", "List`1")',
            'genericList.GetGeneric({scrFloor.GetCompileTimeClass()})',
            'genericList.GetGeneric({levelEvent.GetCompileTimeClass()})',
            '"ApplyEventsToFloors", {listFloor.GetCompileTimeClass()}',
            'listFloor.GetCompileTimeClass(), levelData.GetCompileTimeClass()',
            'scrLevelMaker.GetCompileTimeClass(), listEvent.GetCompileTimeClass()',
            'SameManagedType(applyEventsToFloorsLegacy.GetReturnType(), voidClass)',
            'SameManagedType(applyEventsToFloorsExtended.GetReturnType(), voidClass)',
            'tileDimensionsApplicationSurface = applyEventsToFloorsLegacyVoid',
            'V240: TileDimensions apply ABI scnGame=%d',
            'applicationSurface=%d active=0',
        )
        for marker in expected:
            self.assertIn(marker, self.event_source)

        start = self.event_source.index('auto applyEventsToFloorsLegacy =')
        end = self.event_source.index('LOGD("V240: SetText ABI', start)
        probe = self.event_source[start:end]
        self.assertNotIn(".Call(", probe)
        self.assertNotIn(".Set(", probe)
        self.assertNotIn("BasicHook", probe)
        self.assertNotIn("CreateNewObject", probe)

    def test_set_text_probe_requires_exact_enumerable_and_text_setter_abi(self):
        expected = (
            'Class scrDecorationManager("", "scrDecorationManager")',
            'Class scrDecoration("", "scrDecoration")',
            'Class scrTextDecoration("", "scrTextDecoration")',
            'Class genericEnumerable("System.Collections.Generic", "IEnumerable`1")',
            'genericEnumerable.GetGeneric({stringClass.GetCompileTimeClass()})',
            'genericEnumerable.GetGeneric({scrDecoration.GetCompileTimeClass()})',
            'scrDecorationManager.GetField("instance")',
            'scrDecorationManager.GetProperty("instance")',
            '"GetTaggedDecorations", {enumerableString.GetCompileTimeClass()}',
            'scrTextDecoration.GetMethod("SetText", {Defaults::Get<String*>()})',
            'const Class voidClass = Defaults::Get<void>().ToClass();',
            'SameManagedType(decorationManagerInstanceField.GetType(), scrDecorationManager)',
            'SameManagedType(decorationManagerInstanceProperty.GetType(), scrDecorationManager)',
            'SameManagedType(getTaggedDecorations.GetReturnType(), enumerableDecoration)',
            'SameManagedType(setTextString.GetReturnType(), voidClass)',
            'setTextSurface = scrDecorationManager',
            'V240: SetText ABI scrDecorationManager=%d',
        )
        for marker in expected:
            self.assertIn(marker, self.event_source)
        self.assertNotIn('GetMethod("GetTaggedDecorations", 1)', self.event_source)

    def test_set_text_probe_stays_read_only_until_collection_abi_is_proven(self):
        start = self.event_source.index('auto decorationManagerInstanceField =')
        end = self.event_source.index('LOGD("V240: SetText ABI', start)
        probe = self.event_source[start:end]
        self.assertNotIn(".Call(", probe)
        self.assertNotIn(".Set(", probe)
        self.assertNotIn("BasicHook", probe)
        self.assertNotIn("CreateNewObject", probe)

    def test_marker_is_deliberately_not_a_valid_call_method_expression(self):
        prefix = '__V240_SET_FRAME_RATE__:'
        self.assertIn(prefix, self.event_source)
        self.assertNotIn('(__V240_SET_FRAME_RATE__', self.event_source)
        self.assertNotIn('__V240_SET_FRAME_RATE__(', self.event_source)

    def test_marker_path_is_fail_closed_and_never_falls_into_reflection(self):
        start = self.event_source.index("void HookCallMethodStartEffect(")
        end = self.event_source.index("void ProbeAndInstallEventCompat()", start)
        hook = self.event_source[start:end]
        marker = hook.index("if (ParseSetFrameRateMarker")
        marker_return = hook.index("return;", marker)
        original = hook.index("g_oldCallMethodStartEffect(self, planet)")
        self.assertLess(marker, marker_return, original)
        self.assertIn("SetFrameRate marker suppressed because camera runtime is unavailable", hook)

    def test_non_marker_call_method_preserves_original_behavior(self):
        self.assertIn(
            "if (g_oldCallMethodStartEffect) g_oldCallMethodStartEffect(self, planet);",
            self.event_source,
        )

    def test_capability_stays_false_until_scheduler_hook_is_proven(self):
        hook_call = self.event_source.index(
            "BasicHook(callMethodStartEffect, HookCallMethodStartEffect, g_oldCallMethodStartEffect);"
        )
        old_pointer_check = self.event_source.index(
            "const bool hooked = g_oldCallMethodStartEffect != nullptr;", hook_call
        )
        ready_store = self.event_source.index(
            "g_setFrameRateBackportReady.store(hooked", old_pointer_check
        )
        self.assertLess(hook_call, old_pointer_check, ready_store)
        self.assertIn("std::atomic<bool> g_setFrameRateBackportReady{false};", self.event_source)

    def test_camera_call_uses_existing_v240_camera_singleton(self):
        self.assertIn("IL2CPP::Il2CppObject* camera = GetCameraInstance();", self.event_source)
        self.assertIn("property.Get();", self.event_source)
        self.assertIn("field.Get();", self.event_source)
        self.assertIn("method[camera].Call(enabled, frameRate);", self.event_source)
        self.assertIn("method[camera].Call(enabled, static_cast<int>(std::lround(frameRate)));", self.event_source)
        self.assertNotIn("Application.set_targetFrameRate", self.event_source)
        self.assertNotIn("Time.timeScale", self.event_source)

    def test_bootstrap_registers_event_compat_after_native_library_load(self):
        load = self.bootstrap.index('System.loadLibrary("v240fix");')
        init = self.bootstrap.index("V240EventCompat.initialize();")
        self.assertLess(load, init)
        self.assertIn("private static native void nativeRegister();", self.event_java)
        self.assertIn("nativeIsSetFrameRateBackportReady", self.event_java)
        self.assertIn("return false;", self.event_java)

    def test_event_registration_preregisters_touch_before_activity_is_ready(self):
        register = self.event_source.index(
            "Java_com_unity3d_player_V240EventCompat_nativeRegister")
        touch = self.event_source.index(
            "Java_com_unity3d_player_V240SettingsOverlay_nativeApplyTouchAssist(", register)
        event = self.event_source.index("RegisterV240EventCompat();", touch)
        self.assertLess(register, touch, event)
        early = self.event_source[touch:event]
        self.assertIn("JNI_TRUE, 0.0f", early)

        init = self.bootstrap.index("V240EventCompat.initialize();")
        mobile = self.bootstrap.index("installMobileRuntimeWhenActivityIsReady();", init)
        self.assertLess(init, mobile)

    def test_native_registration_is_idempotent_and_bnm_loaded_gated(self):
        self.assertIn("std::once_flag g_registerOnce;", self.event_source)
        self.assertIn("std::call_once(g_registerOnce", self.event_source)
        self.assertIn("Loading::AddOnLoadedEvent", self.event_source)
        self.assertIn("ProbeAndInstallEventCompat();", self.event_source)
        self.assertIn("Java_com_unity3d_player_V240EventCompat_nativeRegister", self.event_source)

    def test_payload_builds_cannot_drop_event_compat(self):
        self.assertIn(
            'EVENT_COMPAT_SOURCE="${ROOT}/android/v240-fixed-runtime/native/V240EventCompat.cpp"',
            self.native_build,
        )
        self.assertIn('cp "${EVENT_COMPAT_SOURCE}" "${JNI}/V240EventCompat.cpp"', self.native_build)
        self.assertIn("V240EventCompat.cpp", self.native_build)
        self.assertIn("Java_com_unity3d_player_V240EventCompat_nativeIsSetFrameRateBackportReady", self.native_build)
        self.assertIn("V240EventCompat.initialize();", self.java_build)
        self.assertIn("Lcom/unity3d/player/V240EventCompat;", self.java_build)

    def test_runtime_log_has_machine_searchable_capability_markers(self):
        self.assertIn("V240: compatibility surface LevelData=%d", self.probe)
        self.assertIn("scrCamera.SetCustomFrameRate=%d", self.probe)
        self.assertIn("V240: event compat probe ffxCallMethod=%d", self.event_source)
        self.assertIn("V240: TileDimensions ABI scrFloor=%d", self.event_source)
        self.assertIn("V240: TileDimensions apply ABI scnGame=%d", self.event_source)
        self.assertIn("V240: SetText ABI scrDecorationManager=%d", self.event_source)
        self.assertIn("SetFrameRate execution backport ready", self.event_source)
        self.assertIn("opaque preserve-only mode retained", self.event_source)


if __name__ == "__main__":
    unittest.main()
