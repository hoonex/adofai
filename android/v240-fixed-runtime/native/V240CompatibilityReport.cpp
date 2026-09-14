#include <jni.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

#include "universe.h"

using namespace BNM;
using namespace BNM::Structures::Mono;

extern "C" std::size_t V240CopyPostV240CompatibilityReport(
        char* buffer, std::size_t bufferSize);

namespace {
constexpr int kMaxObservedFloorCount = 250000;
constexpr std::size_t kMaxObservedListCapacity = 500000;
constexpr int kMaxSampleFloors = 64;

std::once_flag g_snapshotRegisterOnce;
std::atomic<bool> g_snapshotSurfacePrepared{false};
std::atomic<bool> g_snapshotSurfaceReady{false};

Field<IL2CPP::Il2CppObject*> g_scnGameInstance;
Field<IL2CPP::Il2CppObject*> g_levelMaker;
Field<List<IL2CPP::Il2CppObject*>*> g_listFloors;
Field<float> g_lengthMult;
Field<float> g_widthMult;
Field<int> g_seqId;
bool g_hasSeqId = false;

IL2CPP::Il2CppClass* g_scnGameClass = nullptr;
IL2CPP::Il2CppClass* g_scrLevelMakerClass = nullptr;
IL2CPP::Il2CppClass* g_scrFloorClass = nullptr;
IL2CPP::Il2CppClass* g_listFloorClass = nullptr;
std::string g_snapshotSurfaceReport;

bool SameManagedType(const Class& left, const Class& right) {
    return left && right && left.GetClass() == right.GetClass();
}

bool IsReadableStaticField(const FieldBase& field) {
    return field.IsValid() && field._isStatic && !field._isThreadStatic && !field._isConst;
}

bool IsReadableInstanceField(const FieldBase& field) {
    return field.IsValid() && !field._isStatic && !field._isThreadStatic && !field._isConst;
}

void PrepareTileDimensionsSnapshotSurface() {
    Class scnGame("", "scnGame");
    Class scrLevelMaker("", "scrLevelMaker");
    Class scrFloor("", "scrFloor");
    Class genericList("System.Collections.Generic", "List`1");

    const Class floatClass = Defaults::Get<float>().ToClass();
    const Class intClass = Defaults::Get<int>().ToClass();
    const Class listFloor = (genericList && scrFloor)
            ? genericList.GetGeneric({scrFloor.GetCompileTimeClass()}) : Class{};

    auto scnGameInstance = scnGame ? scnGame.GetField("instance") : FieldBase{};
    auto levelMaker = scnGame ? scnGame.GetField("levelMaker") : FieldBase{};
    auto listFloors = scrLevelMaker ? scrLevelMaker.GetField("listFloors") : FieldBase{};
    auto lengthMult = scrFloor ? scrFloor.GetField("lengthMult") : FieldBase{};
    auto widthMult = scrFloor ? scrFloor.GetField("widthMult") : FieldBase{};
    auto seqId = scrFloor ? scrFloor.GetField("seqID") : FieldBase{};

    const bool scnGameInstanceStaticTyped = IsReadableStaticField(scnGameInstance)
            && SameManagedType(scnGameInstance.GetType(), scnGame);
    const bool levelMakerInstanceTyped = IsReadableInstanceField(levelMaker)
            && SameManagedType(levelMaker.GetType(), scrLevelMaker);
    const bool listFloorsTyped = IsReadableInstanceField(listFloors)
            && listFloor
            && SameManagedType(listFloors.GetType(), listFloor);
    const bool lengthMultFloat = IsReadableInstanceField(lengthMult)
            && SameManagedType(lengthMult.GetType(), floatClass);
    const bool widthMultFloat = IsReadableInstanceField(widthMult)
            && SameManagedType(widthMult.GetType(), floatClass);
    const bool seqIdInt = IsReadableInstanceField(seqId)
            && SameManagedType(seqId.GetType(), intClass);

    const bool ready = scnGame
            && scrLevelMaker
            && scrFloor
            && listFloor
            && scnGameInstanceStaticTyped
            && levelMakerInstanceTyped
            && listFloorsTyped
            && lengthMultFloat
            && widthMultFloat;

    char surface[1024];
    std::snprintf(
            surface,
            sizeof(surface),
            "TileDimensions.liveSnapshot.surface.scnGame=%d\n"
            "TileDimensions.liveSnapshot.surface.scrLevelMaker=%d\n"
            "TileDimensions.liveSnapshot.surface.scrFloor=%d\n"
            "TileDimensions.liveSnapshot.surface.ListFloor=%d\n"
            "TileDimensions.liveSnapshot.surface.scnGameInstanceStaticTyped=%d\n"
            "TileDimensions.liveSnapshot.surface.levelMakerInstanceTyped=%d\n"
            "TileDimensions.liveSnapshot.surface.listFloorsTyped=%d\n"
            "TileDimensions.liveSnapshot.surface.lengthMultFloat=%d\n"
            "TileDimensions.liveSnapshot.surface.widthMultFloat=%d\n"
            "TileDimensions.liveSnapshot.surface.seqIdInt=%d\n"
            "TileDimensions.liveSnapshot.surface.ready=%d\n",
            scnGame ? 1 : 0,
            scrLevelMaker ? 1 : 0,
            scrFloor ? 1 : 0,
            listFloor ? 1 : 0,
            scnGameInstanceStaticTyped ? 1 : 0,
            levelMakerInstanceTyped ? 1 : 0,
            listFloorsTyped ? 1 : 0,
            lengthMultFloat ? 1 : 0,
            widthMultFloat ? 1 : 0,
            seqIdInt ? 1 : 0,
            ready ? 1 : 0);
    g_snapshotSurfaceReport = surface;

    if (ready) {
        g_scnGameInstance = scnGameInstance;
        g_levelMaker = levelMaker;
        g_listFloors = listFloors;
        g_lengthMult = lengthMult;
        g_widthMult = widthMult;
        if (seqIdInt) g_seqId = seqId;
        g_hasSeqId = seqIdInt;
        g_scnGameClass = scnGame.GetClass();
        g_scrLevelMakerClass = scrLevelMaker.GetClass();
        g_scrFloorClass = scrFloor.GetClass();
        g_listFloorClass = listFloor.GetClass();
    }

    g_snapshotSurfaceReady.store(ready, std::memory_order_release);
    g_snapshotSurfacePrepared.store(true, std::memory_order_release);
}

std::string SnapshotUnavailable(std::string report, const char* reason) {
    report += "TileDimensions.liveSnapshot.status=unavailable\n";
    report += "TileDimensions.liveSnapshot.reason=";
    report += reason ? reason : "unknown";
    report += '\n';
    return report;
}

std::string BuildTileDimensionsLiveSnapshot() {
    std::string report = "TileDimensions.liveSnapshot.mode=read-only\n";
    report += "TileDimensions.liveSnapshot.mutatesRuntime=0\n";
    report += "TileDimensions.liveSnapshot.conversionAssumption=none\n";

    if (!g_snapshotSurfacePrepared.load(std::memory_order_acquire)) {
        report += "TileDimensions.liveSnapshot.status=pending\n";
        report += "TileDimensions.liveSnapshot.reason=bnm_surface_not_prepared\n";
        return report;
    }

    report += g_snapshotSurfaceReport;
    if (!g_snapshotSurfaceReady.load(std::memory_order_acquire)) {
        return SnapshotUnavailable(std::move(report), "exact_field_surface_mismatch");
    }

    Field<IL2CPP::Il2CppObject*> scnGameInstance = g_scnGameInstance;
    IL2CPP::Il2CppObject* game = scnGameInstance.Get();
    if (!game) return SnapshotUnavailable(std::move(report), "no_scnGame_instance");
    if (game->klass != g_scnGameClass) {
        return SnapshotUnavailable(std::move(report), "scnGame_instance_type_mismatch");
    }

    Field<IL2CPP::Il2CppObject*> levelMakerField = g_levelMaker;
    IL2CPP::Il2CppObject* levelMaker = levelMakerField[game].Get();
    if (!levelMaker) return SnapshotUnavailable(std::move(report), "no_levelMaker");
    if (levelMaker->klass != g_scrLevelMakerClass) {
        return SnapshotUnavailable(std::move(report), "levelMaker_type_mismatch");
    }

    Field<List<IL2CPP::Il2CppObject*>*> listFloorsField = g_listFloors;
    List<IL2CPP::Il2CppObject*>* floors = listFloorsField[levelMaker].Get();
    if (!floors) return SnapshotUnavailable(std::move(report), "no_listFloors");
    if (floors->klass != g_listFloorClass) {
        return SnapshotUnavailable(std::move(report), "listFloors_runtime_type_mismatch");
    }

    const int floorCount = floors->GetSize();
    const int versionBefore = floors->GetVersion();
    auto* itemsBefore = floors->items;
    const std::size_t capacity = itemsBefore
            ? static_cast<std::size_t>(itemsBefore->GetCapacity()) : 0;

    if (floorCount < 0 || floorCount > kMaxObservedFloorCount) {
        return SnapshotUnavailable(std::move(report), "floor_count_out_of_bounds");
    }
    if (capacity > kMaxObservedListCapacity || capacity < static_cast<std::size_t>(floorCount)) {
        return SnapshotUnavailable(std::move(report), "list_capacity_out_of_bounds");
    }
    if (floorCount > 0 && !itemsBefore) {
        return SnapshotUnavailable(std::move(report), "list_items_missing");
    }

    IL2CPP::Il2CppObject** floorData = floorCount > 0 ? itemsBefore->GetData() : nullptr;
    if (floorCount > 0 && !floorData) {
        return SnapshotUnavailable(std::move(report), "list_data_missing");
    }

    const int sampleCount = std::min(floorCount, kMaxSampleFloors);
    std::string samples;
    samples.reserve(static_cast<std::size_t>(sampleCount) * 96U);

    Field<float> lengthMultField = g_lengthMult;
    Field<float> widthMultField = g_widthMult;
    Field<int> seqIdField = g_seqId;

    for (int index = 0; index < sampleCount; ++index) {
        IL2CPP::Il2CppObject* floor = floorData[index];
        if (!floor) {
            return SnapshotUnavailable(std::move(report), "sample_floor_null");
        }
        if (floor->klass != g_scrFloorClass) {
            return SnapshotUnavailable(std::move(report), "sample_floor_type_mismatch");
        }

        const float lengthMult = lengthMultField[floor].Get();
        const float widthMult = widthMultField[floor].Get();
        if (!std::isfinite(lengthMult) || !std::isfinite(widthMult)) {
            return SnapshotUnavailable(std::move(report), "sample_multiplier_nonfinite");
        }

        char line[256];
        if (g_hasSeqId) {
            const int seqId = seqIdField[floor].Get();
            std::snprintf(
                    line,
                    sizeof(line),
                    "TileDimensions.liveSnapshot.floor[%d].seqID=%d lengthMult=%.9g widthMult=%.9g\n",
                    index,
                    seqId,
                    static_cast<double>(lengthMult),
                    static_cast<double>(widthMult));
        } else {
            std::snprintf(
                    line,
                    sizeof(line),
                    "TileDimensions.liveSnapshot.floor[%d].lengthMult=%.9g widthMult=%.9g\n",
                    index,
                    static_cast<double>(lengthMult),
                    static_cast<double>(widthMult));
        }
        samples += line;
    }

    const int versionAfter = floors->GetVersion();
    const int floorCountAfter = floors->GetSize();
    auto* itemsAfter = floors->items;
    Field<IL2CPP::Il2CppObject*> scnGameInstanceAfterField = g_scnGameInstance;
    IL2CPP::Il2CppObject* gameAfter = scnGameInstanceAfterField.Get();
    Field<IL2CPP::Il2CppObject*> levelMakerAfterField = g_levelMaker;
    IL2CPP::Il2CppObject* levelMakerAfter = gameAfter
            ? levelMakerAfterField[gameAfter].Get() : nullptr;

    if (versionAfter != versionBefore
            || floorCountAfter != floorCount
            || itemsAfter != itemsBefore
            || gameAfter != game
            || levelMakerAfter != levelMaker) {
        return SnapshotUnavailable(std::move(report), "floor_state_changed_during_snapshot");
    }

    char header[256];
    std::snprintf(
            header,
            sizeof(header),
            "TileDimensions.liveSnapshot.status=ready\n"
            "TileDimensions.liveSnapshot.floorCount=%d\n"
            "TileDimensions.liveSnapshot.listCapacity=%zu\n"
            "TileDimensions.liveSnapshot.listVersion=%d\n"
            "TileDimensions.liveSnapshot.sampleCount=%d\n",
            floorCount,
            capacity,
            versionBefore,
            sampleCount);
    report += header;
    report += samples;
    return report;
}

std::size_t CopyText(const std::string& text, char* buffer, std::size_t bufferSize) {
    const std::size_t required = text.size();
    if (!buffer || bufferSize == 0) return required;
    const std::size_t copied = std::min(required, bufferSize - 1);
    if (copied > 0) std::memcpy(buffer, text.data(), copied);
    buffer[copied] = '\0';
    return required;
}

std::string ReadCopiedText(std::size_t (*copyFn)(char*, std::size_t)) {
    if (!copyFn) return {};
    const std::size_t length = copyFn(nullptr, 0);
    std::string text(length + 1, '\0');
    copyFn(text.data(), text.size());
    text.resize(std::strlen(text.c_str()));
    return text;
}
} // namespace

extern "C" std::size_t V240CopyTileDimensionsLiveSnapshot(
        char* buffer, std::size_t bufferSize) {
    return CopyText(BuildTileDimensionsLiveSnapshot(), buffer, bufferSize);
}

extern "C" JNIEXPORT void JNICALL
Java_com_unity3d_player_V240EventCompat_nativeRegisterTileDimensionsSnapshot(
        JNIEnv*, jclass) {
    std::call_once(g_snapshotRegisterOnce, []() {
        Loading::AddOnLoadedEvent(PrepareTileDimensionsSnapshotSurface);
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (!env) return nullptr;

    std::string report = ReadCopiedText(V240CopyPostV240CompatibilityReport);
    const std::string tileDimensions = ReadCopiedText(V240CopyTileDimensionsLiveSnapshot);
    if (!report.empty() && report.back() != '\n') report.push_back('\n');
    report += tileDimensions;
    return env->NewStringUTF(report.c_str());
}
