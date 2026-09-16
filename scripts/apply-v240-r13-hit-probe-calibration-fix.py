#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r13-hit-probe-calibration-fix.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor, found {count}: {old[:180]!r}")
    s = s.replace(old, new, 1)


# r14 follows the real-device r13 evidence while deliberately keeping the existing
# r13 build/channel markers so no unrelated release plumbing has to change:
# - Android touch -> legacy mouse edges are device-proven.
# - the guessed RDUtils.GetFloorAtPosition(Vector2) resolver is absent/different.
# - Persistence.inputOffsetNotSet exists, but Persistence get/set_inputOffset do not.
# Therefore this layer disables both guessed execution paths and performs only a
# bounded metadata inventory. No new hook, managed invocation, field value read,
# calibration write, or synthesized selection is introduced here.

replace_once(
    "std::string g_calibrationMarker;\n",
    "std::string g_calibrationMarker;\n\n"
    "std::atomic<bool> g_metadataInventoryReady{false};\n"
    "std::atomic<int> g_editorHitAbiGuard{0};\n"
    "std::atomic<int> g_editorHitCameraFieldGuard{0};\n"
    "std::atomic<int> g_editorHitMainCameraGuard{0};\n"
    "std::atomic<int> g_editorHitScreenToWorldGuard{0};\n"
    "std::atomic<int> g_editorHitFloorMethodGuard{0};\n"
    "std::atomic<int> g_calibrationSentinelValid{0};\n"
    "std::atomic<int> g_calibrationSentinelConst{0};\n"
    "std::atomic<int> g_calibrationSentinelStatic{0};\n"
    "std::atomic<int> g_calibrationSentinelTypeFloat{0};\n"
    "std::atomic<int> g_calibrationGetterResolved{0};\n"
    "std::atomic<int> g_calibrationSetterResolved{0};\n"
    "std::atomic<int> g_calibrationSaveResolved{0};\n"
    "std::string g_metadataInventory = \"metadataInventoryReady=0\\n\";\n"
    "std::string g_editorHitCallMarker = \"editor-r13-hit.pending\";\n"
)

metadata_code = r'''
constexpr size_t kMetadataMaxMethodsPerClass = 12;
constexpr size_t kMetadataMaxFieldsPerClass = 12;
constexpr size_t kMetadataMaxNameChars = 48;
constexpr size_t kMetadataMaxTypeChars = 56;

std::string ClipMetadataText(const char* value, size_t maxChars) {
    if (value == nullptr) return "?";
    std::string out(value);
    if (out.size() > maxChars) out.resize(maxChars);
    for (char& ch : out) {
        if (ch == '\n' || ch == '\r' || ch == ';' || ch == '|') ch = '_';
    }
    return out;
}

std::string LowerMetadataName(const char* value) {
    std::string out = value == nullptr ? "" : std::string(value);
    for (char& ch : out) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return out;
}

bool MetadataNameMatches(const char* value, bool calibrationSide) {
    const std::string name = LowerMetadataName(value);
    if (name.empty()) return false;
    static constexpr const char* kEditorTokens[] = {
        "floor", "position", "point", "mouse", "tile", "hit", "select", "ray", "camera"
    };
    static constexpr const char* kCalibrationTokens[] = {
        "offset", "calibr", "preset", "latency", "input", "audio", "timing"
    };
    if (calibrationSide) {
        for (const char* token : kCalibrationTokens) {
            if (name.find(token) != std::string::npos) return true;
        }
        return false;
    }
    for (const char* token : kEditorTokens) {
        if (name.find(token) != std::string::npos) return true;
    }
    return false;
}

std::string MetadataTypeName(const IL2CPP::Il2CppType* type) {
    if (type == nullptr) return "?";
    Class cls(type);
    IL2CPP::Il2CppClass* klass = cls ? cls.GetClass() : nullptr;
    if (klass == nullptr || klass->name == nullptr) return "?";
    std::string out;
    if (klass->namespaze != nullptr && klass->namespaze[0] != '\0') {
        out += ClipMetadataText(klass->namespaze, 24);
        out += '.';
    }
    out += ClipMetadataText(klass->name, kMetadataMaxTypeChars);
    if (TypeByRef(type) != 0) out += '&';
    if (out.size() > kMetadataMaxTypeChars) out.resize(kMetadataMaxTypeChars);
    return out;
}

void AppendMetadataClass(std::ostringstream& out, const char* key, const Class& cls,
                         bool calibrationSide) {
    out << "meta." << key << ".present=" << (cls ? 1 : 0) << '\n';
    if (!cls) return;

    const auto methods = cls.GetMethods(false);
    const auto fields = cls.GetFields(false);
    size_t matchingMethods = 0;
    size_t emittedMethods = 0;
    std::ostringstream methodLine;
    for (const MethodBase& method : methods) {
        IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
        if (info == nullptr || info->name == nullptr || !MetadataNameMatches(info->name, calibrationSide)) continue;
        ++matchingMethods;
        if (emittedMethods >= kMetadataMaxMethodsPerClass) continue;
        if (emittedMethods != 0) methodLine << ';';
        methodLine << ClipMetadataText(info->name, kMetadataMaxNameChars)
                   << '(';
        const uint8_t count = info->parameters_count;
        for (uint8_t i = 0; i < count; ++i) {
            if (i != 0) methodLine << ',';
            const IL2CPP::Il2CppType* p = info->parameters != nullptr ? info->parameters[i] : nullptr;
            methodLine << MetadataTypeName(p);
        }
        methodLine << ")->" << MetadataTypeName(info->return_type)
                   << "[S" << (method._isStatic ? 1 : 0)
                   << "P" << (info->methodPointer != nullptr ? 1 : 0) << ']';
        ++emittedMethods;
    }
    out << "meta." << key << ".methodTotal=" << methods.size() << '\n'
        << "meta." << key << ".methodMatches=" << matchingMethods << '\n'
        << "meta." << key << ".methods="
        << (emittedMethods == 0 ? "<none>" : methodLine.str()) << '\n';

    size_t matchingFields = 0;
    size_t emittedFields = 0;
    std::ostringstream fieldLine;
    for (const FieldBase& field : fields) {
        IL2CPP::FieldInfo* info = field.IsValid() ? field.GetInfo() : nullptr;
        if (info == nullptr || info->name == nullptr || !MetadataNameMatches(info->name, calibrationSide)) continue;
        ++matchingFields;
        if (emittedFields >= kMetadataMaxFieldsPerClass) continue;
        if (emittedFields != 0) fieldLine << ';';
        fieldLine << ClipMetadataText(info->name, kMetadataMaxNameChars)
                  << ':' << MetadataTypeName(info->type)
                  << "[S" << (field._isStatic ? 1 : 0)
                  << "C" << (field._isConst ? 1 : 0)
                  << "T" << (field._isThreadStatic ? 1 : 0)
                  << "O" << field.GetOffset() << ']';
        ++emittedFields;
    }
    out << "meta." << key << ".fieldTotal=" << fields.size() << '\n'
        << "meta." << key << ".fieldMatches=" << matchingFields << '\n'
        << "meta." << key << ".fields="
        << (emittedFields == 0 ? "<none>" : fieldLine.str()) << '\n';
}

void CollectV240MetadataInventory() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_metadataInventoryReady.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_installMutex);
    if (g_metadataInventoryReady.load(std::memory_order_relaxed)) return;

    Class rdUtils("", "RDUtils");
    Class editor("", "scnEditor");
    Class floor("", "scrFloor");
    Class persistence("", "Persistence");
    Class conductor("", "scrConductor");
    Class controller("", "scrController");
    Class calibrationPreset("", "CalibrationPreset");
    Class camera("UnityEngine", "Camera");
    Class vector2 = Defaults::Get<Vector2>();
    Class vector3 = Defaults::Get<Vector3>();
    Class floatClass = Defaults::Get<float>();
    Class playerPrefs("UnityEngine", "PlayerPrefs");

    // Retain the r13 exact guesses as metadata facts only. Nothing below invokes them.
    FieldBase cameraField = editor ? editor.GetField("camera") : FieldBase{};
    const bool cameraFieldAbi = cameraField.IsValid() && !cameraField._isStatic &&
            !cameraField._isThreadStatic && !cameraField._isConst && camera &&
            SameClass(cameraField.GetType(), camera);
    g_editorHitCameraFieldGuard.store(cameraFieldAbi ? 1 : 0);

    MethodBase mainBase = camera ? camera.GetMethod("get_main", 0) : MethodBase{};
    IL2CPP::MethodInfo* mainInfo = mainBase.IsValid() ? mainBase.GetInfo() : nullptr;
    const bool mainAbi = mainInfo != nullptr && mainInfo->methodPointer != nullptr &&
            mainBase._isStatic && mainInfo->parameters_count == 0 && camera &&
            SameClass(Class(mainInfo->return_type), camera);
    g_editorHitMainCameraGuard.store(mainAbi ? 1 : 0);

    MethodBase screenBase = camera ? camera.GetMethod("ScreenToWorldPoint", 1) : MethodBase{};
    IL2CPP::MethodInfo* screenInfo = screenBase.IsValid() ? screenBase.GetInfo() : nullptr;
    const bool screenParam = screenInfo != nullptr && screenInfo->parameters_count == 1 &&
            screenInfo->parameters != nullptr && screenInfo->parameters[0] != nullptr;
    const bool screenAbi = screenInfo != nullptr && screenInfo->methodPointer != nullptr &&
            !screenBase._isStatic && screenParam && vector3 &&
            SameClass(Class(screenInfo->parameters[0]), vector3) &&
            TypeByRef(screenInfo->parameters[0]) == 0 &&
            SameClass(Class(screenInfo->return_type), vector3);
    g_editorHitScreenToWorldGuard.store(screenAbi ? 1 : 0);

    MethodBase floorBase = rdUtils ? rdUtils.GetMethod("GetFloorAtPosition", 1) : MethodBase{};
    IL2CPP::MethodInfo* floorInfo = floorBase.IsValid() ? floorBase.GetInfo() : nullptr;
    const bool floorParam = floorInfo != nullptr && floorInfo->parameters_count == 1 &&
            floorInfo->parameters != nullptr && floorInfo->parameters[0] != nullptr;
    const bool floorAbi = floorInfo != nullptr && floorInfo->methodPointer != nullptr &&
            floorBase._isStatic && floorParam && vector2 && floor &&
            SameClass(Class(floorInfo->parameters[0]), vector2) &&
            TypeByRef(floorInfo->parameters[0]) == 0 &&
            SameClass(Class(floorInfo->return_type), floor);
    g_editorHitFloorMethodGuard.store(floorAbi ? 1 : 0);
    g_editorHitAbiGuard.store(((cameraFieldAbi || mainAbi) && screenAbi && floorAbi) ? 1 : 0);

    FieldBase sentinelBase = persistence ? persistence.GetField("inputOffsetNotSet") : FieldBase{};
    MethodBase getterBase = persistence ? persistence.GetMethod("get_inputOffset", 0) : MethodBase{};
    MethodBase setterBase = persistence ? persistence.GetMethod("set_inputOffset", 1) : MethodBase{};
    MethodBase saveBase = playerPrefs ? playerPrefs.GetMethod("Save", 0) : MethodBase{};
    IL2CPP::MethodInfo* getterInfo = getterBase.IsValid() ? getterBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* setterInfo = setterBase.IsValid() ? setterBase.GetInfo() : nullptr;
    IL2CPP::MethodInfo* saveInfo = saveBase.IsValid() ? saveBase.GetInfo() : nullptr;
    const bool setterParam = setterInfo != nullptr && setterInfo->parameters_count == 1 &&
            setterInfo->parameters != nullptr && setterInfo->parameters[0] != nullptr;
    g_calibrationSentinelValid.store(sentinelBase.IsValid() ? 1 : 0);
    g_calibrationSentinelConst.store(sentinelBase.IsValid() && sentinelBase._isConst ? 1 : 0);
    g_calibrationSentinelStatic.store(sentinelBase.IsValid() && sentinelBase._isStatic ? 1 : 0);
    g_calibrationSentinelTypeFloat.store(
            sentinelBase.IsValid() && floatClass && SameClass(sentinelBase.GetType(), floatClass) ? 1 : 0);
    g_calibrationGetterResolved.store(getterInfo != nullptr && getterInfo->methodPointer != nullptr ? 1 : 0);
    g_calibrationSetterResolved.store(setterInfo != nullptr && setterInfo->methodPointer != nullptr ? 1 : 0);
    g_calibrationSaveResolved.store(saveInfo != nullptr && saveInfo->methodPointer != nullptr ? 1 : 0);
    const bool calibrationAbi = sentinelBase.IsValid() && sentinelBase._isConst &&
            !sentinelBase._isThreadStatic && floatClass && SameClass(sentinelBase.GetType(), floatClass) &&
            getterInfo != nullptr && getterInfo->methodPointer != nullptr && getterBase._isStatic &&
            getterInfo->parameters_count == 0 && SameClass(Class(getterInfo->return_type), floatClass) &&
            setterInfo != nullptr && setterInfo->methodPointer != nullptr && setterBase._isStatic &&
            setterParam && SameClass(Class(setterInfo->parameters[0]), floatClass) &&
            saveInfo != nullptr && saveInfo->methodPointer != nullptr && saveBase._isStatic &&
            saveInfo->parameters_count == 0;
    g_calibrationAbiGuard.store(calibrationAbi ? 1 : 0);

    std::ostringstream out;
    out << "metadataInventoryReady=1\n"
        << "metadataInventoryRevision=1\n"
        << "metadataInventoryPolicy=bounded-read-only-no-invoke\n"
        << "metadataInventoryMethodCalls=0\n"
        << "metadataInventoryFieldValueReads=0\n"
        << "metadataInventoryWrites=0\n"
        << "metadataInventoryMaxMethodsPerClass=" << kMetadataMaxMethodsPerClass << '\n'
        << "metadataInventoryMaxFieldsPerClass=" << kMetadataMaxFieldsPerClass << '\n';
    AppendMetadataClass(out, "RDUtils", rdUtils, false);
    AppendMetadataClass(out, "scnEditor", editor, false);
    AppendMetadataClass(out, "scrFloor", floor, false);
    AppendMetadataClass(out, "Persistence", persistence, true);
    AppendMetadataClass(out, "scrConductor", conductor, true);
    AppendMetadataClass(out, "scrController", controller, true);
    AppendMetadataClass(out, "CalibrationPreset", calibrationPreset, true);
    g_metadataInventory = out.str();
    g_metadataInventoryReady.store(true, std::memory_order_release);
}

'''
replace_once("bool PrepareCalibrationFuse() {\n", metadata_code + "bool PrepareCalibrationFuse() {\n")

# Fix r12's impossible BNM literal-constant guard for diagnostics only. The write routine
# remains compiled for historical comparison but is no longer called by ReconcileInstallState.
replace_once(
    "    const bool abi = sentinelBase.IsValid() && sentinelBase._isStatic && sentinelBase._isConst &&\n",
    "    const bool abi = sentinelBase.IsValid() && sentinelBase._isConst &&\n"
    "            !sentinelBase._isThreadStatic &&\n"
)

# Disable both unproven active paths. Metadata collection runs before the existing pass-through
# editor hooks are installed; no guessed hit resolver or calibration writer is executed.
replace_once("    MaybeNeutralizeUnsetCalibration();\n", "    CollectV240MetadataInventory();\n")

replace_once(
    "nativeProbe=cache-post-bnm-scneditor-input-edge-calibration-v1\\n",
    "nativeProbe=cache-post-bnm-scneditor-hitprobe-calibration-v2\\n"
)
replace_once(
    "nativeStage=post-bnm-scneditor-input-edge-and-calibration\\n",
    "nativeStage=post-bnm-scneditor-world-hit-and-calibration\\n"
)
replace_once("abiProbeRevision=12\\n", "abiProbeRevision=13\\n")

replace_once(
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n',
    '        << "editorInputEdgePolicy=observe-only" << \'\\n\'\n'
    '        << "editorHitPolicy=screen-to-world-rdutils-observe-only" << \'\\n\'\n'
    '        << "editorHitMutation=0" << \'\\n\'\n'
    '        << "editorHitExecution=disabled-r14-metadata-inventory" << \'\\n\'\n'
    '        << "editorHitAbiGuard=" << g_editorHitAbiGuard.load() << \'\\n\'\n'
    '        << "editorHitCameraFieldGuard=" << g_editorHitCameraFieldGuard.load() << \'\\n\'\n'
    '        << "editorHitMainCameraGuard=" << g_editorHitMainCameraGuard.load() << \'\\n\'\n'
    '        << "editorHitScreenToWorldGuard=" << g_editorHitScreenToWorldGuard.load() << \'\\n\'\n'
    '        << "editorHitFloorMethodGuard=" << g_editorHitFloorMethodGuard.load() << \'\\n\'\n'
    '        << "editorHitMarkerReady=0" << \'\\n\'\n'
    '        << "editorHitRecoveryState=0" << \'\\n\'\n'
    '        << "editorHitFirstCallProven=0" << \'\\n\'\n'
    '        << "editorHitProbeCalls=0" << \'\\n\'\n'
    '        << "editorHitProbeDownCalls=0" << \'\\n\'\n'
    '        << "editorHitFloorFound=0" << \'\\n\'\n'
    '        << "editorHitDownFloorFound=0" << \'\\n\'\n'
    '        << "editorHitHeldFloorFound=0" << \'\\n\'\n'
    '        << "editorHitCameraSource=0" << \'\\n\'\n'
    '        << "editorHitLastWorldX100=0" << \'\\n\'\n'
    '        << "editorHitLastWorldY100=0" << \'\\n\'\n'
    '        << "editorHitLastFloorNonNull=0" << \'\\n\'\n'
    '        << "editorSelectWhileHitFound=0" << \'\\n\'\n'
    '        << "editorSelectMatchesHit=0" << \'\\n\'\n'
)

replace_once(
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel" << \'\\n\'\n',
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel-bnm-const-v2" << \'\\n\'\n'
    '        << "calibrationExecution=disabled-r14-metadata-inventory" << \'\\n\'\n'
)
replace_once(
    '        << "calibrationAfterX100=" << g_calibrationAfterX100.load() << \'\\n\'\n',
    '        << "calibrationAfterX100=" << g_calibrationAfterX100.load() << \'\\n\'\n'
    '        << "calibrationSentinelValid=" << g_calibrationSentinelValid.load() << \'\\n\'\n'
    '        << "calibrationSentinelConst=" << g_calibrationSentinelConst.load() << \'\\n\'\n'
    '        << "calibrationSentinelStatic=" << g_calibrationSentinelStatic.load() << \'\\n\'\n'
    '        << "calibrationSentinelTypeFloat=" << g_calibrationSentinelTypeFloat.load() << \'\\n\'\n'
    '        << "calibrationGetterResolved=" << g_calibrationGetterResolved.load() << \'\\n\'\n'
    '        << "calibrationSetterResolved=" << g_calibrationSetterResolved.load() << \'\\n\'\n'
    '        << "calibrationSaveResolved=" << g_calibrationSaveResolved.load() << \'\\n\'\n'
    '        << g_metadataInventory'
)

for marker in (
    "abiProbeRevision=13",
    "nativeProbe=cache-post-bnm-scneditor-hitprobe-calibration-v2",
    "editorInputEdgePolicy=observe-only",
    "editorHitPolicy=screen-to-world-rdutils-observe-only",
    "editorHitMutation=0",
    'Class rdUtils("", "RDUtils")',
    'rdUtils.GetMethod("GetFloorAtPosition", 1)',
    'camera.GetMethod("ScreenToWorldPoint", 1)',
    "editor-r13-hit.pending",
    "calibrationPolicy=neutralize-only-exact-game-sentinel-bnm-const-v2",
    "sentinelBase.IsValid() && sentinelBase._isConst",
    "metadataInventoryRevision=1",
    "metadataInventoryPolicy=bounded-read-only-no-invoke",
    "metadataInventoryMethodCalls=0",
    "metadataInventoryFieldValueReads=0",
    "metadataInventoryWrites=0",
    "CollectV240MetadataInventory();",
    'AppendMetadataClass(out, "RDUtils"',
    'AppendMetadataClass(out, "Persistence"',
    'AppendMetadataClass(out, "CalibrationPreset"',
):
    if marker not in s:
        raise SystemExit(f"r14 metadata marker missing after transform: {marker}")

if "sentinelBase.IsValid() && sentinelBase._isStatic && sentinelBase._isConst" in s:
    raise SystemExit("r12 impossible const/static calibration guard survived")
if "    MaybeNeutralizeUnsetCalibration();\n" in s:
    raise SystemExit("unproven calibration writer must not execute in r14 metadata inventory")
if s.count("BasicHook(") != 5:
    raise SystemExit(f"r14 must not add hooks; expected five compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
