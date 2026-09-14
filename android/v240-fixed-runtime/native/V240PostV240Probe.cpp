#include <jni.h>
#include <mutex>
#include <sstream>
#include <string>

#include "universe.h"
#include "Logger.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

namespace {
std::once_flag g_postV240ProbeOnce;
std::mutex g_postV240ReportMutex;
std::string g_postV240Report =
        "V240 compatibility report\n"
        "probe=pending\n"
        "mode=evidence-only\n"
        "activeBackports=SetFrameRate-only\n";

bool SameManagedType(const Class& left, const Class& right) {
    return left && right && left.GetClass() == right.GetClass();
}

void ProbePostV240Feasibility() {
    // Evidence-only inventory. This file must stay non-mutating: it deliberately performs no
    // managed calls, field/property writes, object creation, hooks, or load callbacks. The caller
    // must only invoke V240RunPostV240FeasibilityProbe after BNM reports IL2CPP fully loaded.
    // Active event backports remain forbidden until the required v2.4 ABI and lifecycle semantics
    // are independently proven.
    Class levelData("ADOFAI", "LevelData");
    Class levelEvent("ADOFAI", "LevelEvent");
    Class scnGame("", "scnGame");
    Class scrLevelMaker("", "scrLevelMaker");
    Class scrFloor("", "scrFloor");
    Class genericList("System.Collections.Generic", "List`1");

    Class scrPlanet("", "scrPlanet");
    Class ffxPlusBase("", "ffxPlusBase");
    Class scrController("", "scrController");
    Class inputEventTarget("", "InputEventTarget");
    Class inputEventState("", "InputEventState");
    Class ffxSetInputEventPlus("", "ffxSetInputEventPlus");

    Class scrDecorationManager("", "scrDecorationManager");
    Class scrDecoration("", "scrDecoration");
    Class scrParticleDecoration("", "scrParticleDecoration");
    Class particleSystem("UnityEngine", "ParticleSystem");
    Class genericEnumerable("System.Collections.Generic", "IEnumerable`1");
    Class linqEnumerable("System.Linq", "Enumerable");

    Class ffxSetFilterPlus("", "ffxSetFilterPlus");
    Class ffxSetFilterAdvancedPlus("", "ffxSetFilterAdvancedPlus");
    Class systemType("System", "Type");
    Class gameObject("UnityEngine", "GameObject");
    Class component("UnityEngine", "Component");
    Class behaviour("UnityEngine", "Behaviour");

    const Class stringClass = Defaults::Get<String*>().ToClass();
    const Class boolClass = Defaults::Get<bool>().ToClass();
    const Class floatClass = Defaults::Get<float>().ToClass();
    const Class voidClass = Defaults::Get<void>().ToClass();

    const Class listFloor = (genericList && scrFloor)
            ? genericList.GetGeneric({scrFloor.GetCompileTimeClass()}) : Class{};
    const Class listEvent = (genericList && levelEvent)
            ? genericList.GetGeneric({levelEvent.GetCompileTimeClass()}) : Class{};
    auto floorLengthMult = scrFloor ? scrFloor.GetField("lengthMult") : FieldBase{};
    auto floorWidthMult = scrFloor ? scrFloor.GetField("widthMult") : FieldBase{};
    auto applyEventsToFloorsLegacy = (scnGame && listFloor)
            ? scnGame.GetMethod(
                    "ApplyEventsToFloors", {listFloor.GetCompileTimeClass()})
            : MethodBase{};
    auto applyEventsToFloorsExtended = (scnGame && listFloor && levelData && scrLevelMaker && listEvent)
            ? scnGame.GetMethod(
                    "ApplyEventsToFloors",
                    {listFloor.GetCompileTimeClass(), levelData.GetCompileTimeClass(),
                     scrLevelMaker.GetCompileTimeClass(), listEvent.GetCompileTimeClass()})
            : MethodBase{};
    const bool floorLengthMultFloat = floorLengthMult.IsValid()
            && SameManagedType(floorLengthMult.GetType(), floatClass);
    const bool floorWidthMultFloat = floorWidthMult.IsValid()
            && SameManagedType(floorWidthMult.GetType(), floatClass);
    const bool tileDimensionsSurface = scrFloor
            && floorLengthMultFloat
            && floorWidthMultFloat;
    const bool applyEventsToFloorsLegacyVoid = applyEventsToFloorsLegacy.IsValid()
            && SameManagedType(applyEventsToFloorsLegacy.GetReturnType(), voidClass);
    const bool applyEventsToFloorsExtendedVoid = applyEventsToFloorsExtended.IsValid()
            && SameManagedType(applyEventsToFloorsExtended.GetReturnType(), voidClass);
    const bool tileDimensionsApplicationSurface = applyEventsToFloorsLegacyVoid
            || applyEventsToFloorsExtendedVoid;

    auto startEffectWithOffset = (ffxPlusBase && scrPlanet)
            ? ffxPlusBase.GetMethod("StartEffectWithOffset", {scrPlanet.GetCompileTimeClass()})
            : MethodBase{};
    auto resetInputEventFfx = scrController
            ? scrController.GetMethod("ResetInputEventFfx", 0) : MethodBase{};
    auto inputEventFfx = scrController
            ? scrController.GetField("inputEventFfx") : FieldBase{};
    const bool startEffectWithOffsetVoid = startEffectWithOffset.IsValid()
            && SameManagedType(startEffectWithOffset.GetReturnType(), voidClass);
    const bool resetInputEventFfxVoid = resetInputEventFfx.IsValid()
            && SameManagedType(resetInputEventFfx.GetReturnType(), voidClass);
    const bool inputSchedulerBase = ffxPlusBase
            && scrController
            && startEffectWithOffsetVoid
            && resetInputEventFfxVoid
            && inputEventFfx.IsValid();

    const Class enumerableString = (genericEnumerable && stringClass)
            ? genericEnumerable.GetGeneric({stringClass.GetCompileTimeClass()}) : Class{};
    const Class enumerableDecoration = (genericEnumerable && scrDecoration)
            ? genericEnumerable.GetGeneric({scrDecoration.GetCompileTimeClass()}) : Class{};
    auto decorationManagerInstanceField = scrDecorationManager
            ? scrDecorationManager.GetField("instance") : FieldBase{};
    auto decorationManagerInstanceProperty = scrDecorationManager
            ? scrDecorationManager.GetProperty("instance") : PropertyBase{};
    auto getTaggedDecorations = (scrDecorationManager && enumerableString)
            ? scrDecorationManager.GetMethod(
                    "GetTaggedDecorations", {enumerableString.GetCompileTimeClass()})
            : MethodBase{};
    auto toArrayDefinition = linqEnumerable
            ? linqEnumerable.GetMethod("ToArray", 1) : MethodBase{};
    auto toDecorationArray = (toArrayDefinition.IsValid() && scrDecoration)
            ? toArrayDefinition.GetGeneric({scrDecoration.GetCompileTimeClass()}) : MethodBase{};
    auto particleSystemField = scrParticleDecoration
            ? scrParticleDecoration.GetField("particleSystem") : FieldBase{};
    auto emitCount = particleSystem
            ? particleSystem.GetMethod("Emit", {Defaults::Get<int>()}) : MethodBase{};

    const bool decorationManagerFieldTyped = decorationManagerInstanceField.IsValid()
            && SameManagedType(decorationManagerInstanceField.GetType(), scrDecorationManager);
    const bool decorationManagerPropertyTyped = decorationManagerInstanceProperty.IsValid()
            && SameManagedType(decorationManagerInstanceProperty.GetType(), scrDecorationManager);
    const bool decorationManagerSingletonTyped = decorationManagerFieldTyped
            || decorationManagerPropertyTyped;
    const bool getTaggedDecorationsTyped = getTaggedDecorations.IsValid()
            && enumerableDecoration
            && SameManagedType(getTaggedDecorations.GetReturnType(), enumerableDecoration);
    const bool decorationMaterializerTyped = toDecorationArray.IsValid();
    const bool particleSystemFieldTyped = particleSystemField.IsValid()
            && SameManagedType(particleSystemField.GetType(), particleSystem);
    const bool emitCountVoid = emitCount.IsValid()
            && SameManagedType(emitCount.GetReturnType(), voidClass);
    const bool emitParticleSubstrate = scrParticleDecoration
            && particleSystem
            && decorationManagerSingletonTyped
            && getTaggedDecorationsTyped
            && decorationMaterializerTyped
            && particleSystemFieldTyped
            && emitCountVoid;

    auto typeGetType = (systemType && stringClass)
            ? systemType.GetMethod("GetType", {stringClass.GetCompileTimeClass()}) : MethodBase{};
    auto getComponentByType = (gameObject && systemType)
            ? gameObject.GetMethod("GetComponent", {systemType.GetCompileTimeClass()}) : MethodBase{};
    auto addComponentByType = (gameObject && systemType)
            ? gameObject.GetMethod("AddComponent", {systemType.GetCompileTimeClass()}) : MethodBase{};
    auto behaviourEnabled = behaviour ? behaviour.GetProperty("enabled") : PropertyBase{};
    const bool typeGetTypeTyped = typeGetType.IsValid()
            && SameManagedType(typeGetType.GetReturnType(), systemType);
    const bool getComponentTyped = getComponentByType.IsValid()
            && SameManagedType(getComponentByType.GetReturnType(), component);
    const bool addComponentTyped = addComponentByType.IsValid()
            && SameManagedType(addComponentByType.GetReturnType(), component);
    const bool behaviourEnabledBool = behaviourEnabled.IsValid()
            && SameManagedType(behaviourEnabled.GetType(), boolClass);
    const bool advancedFilterReflectionBase = systemType
            && gameObject
            && component
            && behaviour
            && typeGetTypeTyped
            && getComponentTyped
            && addComponentTyped
            && behaviourEnabledBool;

    LOGD("V240: SetInputEvent feasibility ffxPlusBase=%d StartEffectWithOffsetVoid=%d scrController=%d ResetInputEventFfxVoid=%d inputEventFfx=%d targetEnum=%d stateEnum=%d exactPlusClass=%d schedulerBase=%d active=0",
         ffxPlusBase ? 1 : 0,
         startEffectWithOffsetVoid ? 1 : 0,
         scrController ? 1 : 0,
         resetInputEventFfxVoid ? 1 : 0,
         inputEventFfx.IsValid() ? 1 : 0,
         inputEventTarget ? 1 : 0,
         inputEventState ? 1 : 0,
         ffxSetInputEventPlus ? 1 : 0,
         inputSchedulerBase ? 1 : 0);

    LOGD("V240: EmitParticle ABI decorationManager=%d singletonTyped=%d GetTaggedDecorationsTyped=%d ToArrayDecoration=%d scrParticleDecoration=%d particleSystemFieldTyped=%d ParticleSystem=%d EmitIntVoid=%d substrate=%d active=0",
         scrDecorationManager ? 1 : 0,
         decorationManagerSingletonTyped ? 1 : 0,
         getTaggedDecorationsTyped ? 1 : 0,
         decorationMaterializerTyped ? 1 : 0,
         scrParticleDecoration ? 1 : 0,
         particleSystemFieldTyped ? 1 : 0,
         particleSystem ? 1 : 0,
         emitCountVoid ? 1 : 0,
         emitParticleSubstrate ? 1 : 0);

    LOGD("V240: SetParticle feasibility scrParticleDecoration=%d ParticleSystem=%d exactSetParticlePlus=%d preserveOnly=1",
         scrParticleDecoration ? 1 : 0,
         particleSystem ? 1 : 0,
         Class("ADOFAI.FloorFX", "ffxSetParticlePlus") ? 1 : 0);

    LOGD("V240: SetFilterAdvanced feasibility legacySetFilterPlus=%d exactAdvancedPlus=%d TypeGetType=%d GetComponentType=%d AddComponentType=%d BehaviourEnabledBool=%d reflectionBase=%d active=0",
         ffxSetFilterPlus ? 1 : 0,
         ffxSetFilterAdvancedPlus ? 1 : 0,
         typeGetTypeTyped ? 1 : 0,
         getComponentTyped ? 1 : 0,
         addComponentTyped ? 1 : 0,
         behaviourEnabledBool ? 1 : 0,
         advancedFilterReflectionBase ? 1 : 0);

    std::ostringstream report;
    report << "V240 compatibility report\n"
           << "probe=complete\n"
           << "mode=evidence-only\n"
           << "activeBackports=SetFrameRate-only\n"
           << "TileDimensions.scrFloor=" << (scrFloor ? 1 : 0) << '\n'
           << "TileDimensions.lengthMultFloat=" << (floorLengthMultFloat ? 1 : 0) << '\n'
           << "TileDimensions.widthMultFloat=" << (floorWidthMultFloat ? 1 : 0) << '\n'
           << "TileDimensions.fieldSurface=" << (tileDimensionsSurface ? 1 : 0) << '\n'
           << "TileDimensions.scnGame=" << (scnGame ? 1 : 0) << '\n'
           << "TileDimensions.ListFloor=" << (listFloor ? 1 : 0) << '\n'
           << "TileDimensions.ListEvent=" << (listEvent ? 1 : 0) << '\n'
           << "TileDimensions.LevelData=" << (levelData ? 1 : 0) << '\n'
           << "TileDimensions.scrLevelMaker=" << (scrLevelMaker ? 1 : 0) << '\n'
           << "TileDimensions.ApplyEventsToFloorsLegacyVoid="
           << (applyEventsToFloorsLegacyVoid ? 1 : 0) << '\n'
           << "TileDimensions.ApplyEventsToFloorsExtendedVoid="
           << (applyEventsToFloorsExtendedVoid ? 1 : 0) << '\n'
           << "TileDimensions.applicationSurface="
           << (tileDimensionsApplicationSurface ? 1 : 0) << '\n'
           << "TileDimensions.active=0\n"
           << "SetInputEvent.schedulerBase=" << (inputSchedulerBase ? 1 : 0) << '\n'
           << "SetInputEvent.active=0\n"
           << "EmitParticle.substrate=" << (emitParticleSubstrate ? 1 : 0) << '\n'
           << "EmitParticle.active=0\n"
           << "SetParticle.scrParticleDecoration=" << (scrParticleDecoration ? 1 : 0) << '\n'
           << "SetParticle.preserveOnly=1\n"
           << "SetFilterAdvanced.reflectionBase=" << (advancedFilterReflectionBase ? 1 : 0) << '\n'
           << "SetFilterAdvanced.active=0\n";
    {
        std::lock_guard<std::mutex> lock(g_postV240ReportMutex);
        g_postV240Report = report.str();
    }
}
} // namespace

extern "C" void V240RunPostV240FeasibilityProbe() {
    // Called only from V240EventCompat's existing post-BNM loaded callback, before event compat
    // installation. The probe stays evidence-only; std::call_once prevents duplicate scans.
    std::call_once(g_postV240ProbeOnce, []() { ProbePostV240Feasibility(); });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240SettingsOverlay_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (!env) return nullptr;
    std::string report;
    {
        std::lock_guard<std::mutex> lock(g_postV240ReportMutex);
        report = g_postV240Report;
    }
    return env->NewStringUTF(report.c_str());
}
