#!/usr/bin/env python3
"""R26: exact synchronous SFB SaveFilePanel/OpenFolderPanel support through the stable parent SAF bridge.

The authoritative v2.4 metadata exposes:
- SaveFilePanel(string title, string directory, string defaultName, string extension) -> string
- SaveFilePanel(string title, string directory, string defaultName, SFB.ExtensionFilter[] extensions) -> string
- OpenFolderPanel(string title, string directory, bool multiselect) -> string[]

R26 does not duplicate SAF storage code. The child hot DEX reflects into the stable parent
V240AndroidBridge for ACTION_CREATE_DOCUMENT / ACTION_OPEN_DOCUMENT_TREE and save binding.
The three managed hooks are independently self-fused from the already device-proven OpenFilePanel hook.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r26-sfb-save-folder.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r26 anchor must occur once, got {n}: {old[:180]!r}")
    s = s.replace(old, new, 1)

# JNI method ids belong to the same global child bridge reference created by r11.
once(
    "jmethodID g_dynamicDiagnosticsMethod = nullptr;\n",
    "jmethodID g_dynamicDiagnosticsMethod = nullptr;\n"
    "jmethodID g_dynamicSaveMethod = nullptr;\n"
    "jmethodID g_dynamicFolderMethod = nullptr;\n"
)

once(
    "std::string g_calibrationR22CallMarker;\n",
    "std::string g_calibrationR22CallMarker;\n"
    "using SfbSaveStringFn = String* (*)(String*, String*, String*, String*, IL2CPP::MethodInfo*);\n"
    "using SfbSaveFiltersFn = String* (*)(String*, String*, String*, void*, IL2CPP::MethodInfo*);\n"
    "using SfbFolderFn = Array<String*>* (*)(String*, String*, bool, IL2CPP::MethodInfo*);\n"
    "SfbSaveStringFn g_oldSfbSaveString = nullptr;\n"
    "SfbSaveFiltersFn g_oldSfbSaveFilters = nullptr;\n"
    "SfbFolderFn g_oldSfbFolder = nullptr;\n"
    "std::atomic<int> g_sfbExtraAbiGuard{0};\n"
    "std::atomic<int> g_sfbExtraHookInstalled{0};\n"
    "std::atomic<int> g_sfbExtraInstallAttempted{0};\n"
    "std::atomic<int> g_sfbExtraMarkerReady{0};\n"
    "std::atomic<int> g_sfbExtraRecoveryState{0};\n"
    "std::atomic<int> g_sfbExtraCallsInFlight{0};\n"
    "std::atomic<int> g_sfbSaveCalls{0};\n"
    "std::atomic<int> g_sfbSaveReturns{0};\n"
    "std::atomic<int> g_sfbFolderCalls{0};\n"
    "std::atomic<int> g_sfbFolderReturns{0};\n"
    "std::atomic<int> g_sfbExtraBridgeReady{0};\n"
    "std::string g_sfbExtraInstallMarker;\n"
    "std::string g_sfbExtraCallMarker;\n"
)

# Extend r11 bridge registration atomically. A bundle always ships native+DEX from one commit.
once(
    """    if (g_dynamicBridgeClass != nullptr && g_dynamicBegin != nullptr &&
        g_dynamicAwait != nullptr && g_dynamicDiagnosticsMethod != nullptr) {
""",
    """    if (g_dynamicBridgeClass != nullptr && g_dynamicBegin != nullptr &&
        g_dynamicAwait != nullptr && g_dynamicDiagnosticsMethod != nullptr &&
        g_dynamicSaveMethod != nullptr && g_dynamicFolderMethod != nullptr) {
"""
)
once(
    """    jmethodID diagnostics = env->GetStaticMethodID(
            bridgeClass, "diagnostics", "()Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (begin == nullptr || await == nullptr || diagnostics == nullptr) return false;
""",
    """    jmethodID diagnostics = env->GetStaticMethodID(
            bridgeClass, "diagnostics", "()Ljava/lang/String;");
    jmethodID save = env->GetStaticMethodID(
            bridgeClass, "save",
            "(Ljava/lang/String;Ljava/lang/String;J)Ljava/lang/String;");
    jmethodID folder = env->GetStaticMethodID(
            bridgeClass, "folder", "(J)Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (begin == nullptr || await == nullptr || diagnostics == nullptr ||
        save == nullptr || folder == nullptr) return false;
"""
)
once(
    """    g_dynamicBridgeClass = global;
    g_dynamicBegin = begin;
    g_dynamicAwait = await;
    g_dynamicDiagnosticsMethod = diagnostics;
    g_dynamicBridgeReady.store(true, std::memory_order_release);
    g_dynamicBridgeMethodResolution.store(1);
    return true;
}
""",
    """    g_dynamicBridgeClass = global;
    g_dynamicBegin = begin;
    g_dynamicAwait = await;
    g_dynamicDiagnosticsMethod = diagnostics;
    g_dynamicSaveMethod = save;
    g_dynamicFolderMethod = folder;
    g_sfbExtraBridgeReady.store(1);
    g_dynamicBridgeReady.store(true, std::memory_order_release);
    g_dynamicBridgeMethodResolution.store(1);
    return true;
}
"""
)

r26_code = r'''
constexpr jlong kSfbExtraPickerTimeoutMs = 600000;

bool PrepareSfbExtraFuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) { g_sfbExtraRecoveryState.store(4); return false; }
    g_sfbExtraInstallMarker = dir + "/sfb-r26-extra-install.pending";
    g_sfbExtraCallMarker = dir + "/sfb-r26-extra-call.pending";
    g_sfbExtraMarkerReady.store(1);
    if (MarkerExists(g_sfbExtraInstallMarker)) { g_sfbExtraRecoveryState.store(1); return false; }
    if (MarkerExists(g_sfbExtraCallMarker)) { g_sfbExtraRecoveryState.store(2); return false; }
    const std::string probe = dir + "/sfb-r26-extra-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_sfbExtraMarkerReady.store(0);
        g_sfbExtraRecoveryState.store(4);
        return false;
    }
    ClearMarker(probe);
    return true;
}

std::string ReadDynamicPickerState(JNIEnv* env, jstring state) {
    if (env == nullptr || state == nullptr) return "";
    std::string encoded;
    const char* chars = env->GetStringUTFChars(state, nullptr);
    if (chars != nullptr) {
        const std::string value(chars);
        env->ReleaseStringUTFChars(state, chars);
        if (value.rfind("O:", 0) == 0) {
            encoded = value.substr(2);
            g_safLastState.store(encoded.empty() ? 3 : 4);
        } else if (value.rfind("C:", 0) == 0) {
            g_safLastState.store(3);
        } else {
            g_safLastState.store(2);
        }
    }
    return encoded;
}

std::string RunDynamicSavePicker(String* defaultName, const std::string& extension) {
    std::lock_guard<std::mutex> serialized(g_pickerMutex);
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!GetEnv(&env, &attached) || !DynamicBridgeReady()) {
        g_safLastState.store(1);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    jclass bridgeClass = nullptr;
    jmethodID method = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
        bridgeClass = g_dynamicBridgeClass;
        method = g_dynamicSaveMethod;
    }
    if (!bridgeClass || !method) {
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    std::string suggested = defaultName ? defaultName->str() : "level.adofai";
    if (suggested.empty()) suggested = "level.adofai";
    if (suggested.size() > 512) suggested.resize(512);
    const std::string mime = MimeForExtension(extension).empty()
            ? "application/octet-stream" : MimeForExtension(extension);
    jstring jname = env->NewStringUTF(suggested.c_str());
    jstring jmime = env->NewStringUTF(mime.c_str());
    if (!jname || !jmime || env->ExceptionCheck()) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (jname) env->DeleteLocalRef(jname);
        if (jmime) env->DeleteLocalRef(jmime);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    jstring state = reinterpret_cast<jstring>(env->CallStaticObjectMethod(
            bridgeClass, method, jname, jmime, kSfbExtraPickerTimeoutMs));
    env->DeleteLocalRef(jname);
    env->DeleteLocalRef(jmime);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        state = nullptr;
        g_safLastState.store(2);
    }
    std::string encoded = ReadDynamicPickerState(env, state);
    if (state) env->DeleteLocalRef(state);
    RefreshDynamicDiagnostics(env);
    if (attached) g_vm->DetachCurrentThread();
    return encoded;
}

std::string RunDynamicFolderPicker() {
    std::lock_guard<std::mutex> serialized(g_pickerMutex);
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!GetEnv(&env, &attached) || !DynamicBridgeReady()) {
        g_safLastState.store(1);
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }

    jclass bridgeClass = nullptr;
    jmethodID method = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_dynamicBridgeMutex);
        bridgeClass = g_dynamicBridgeClass;
        method = g_dynamicFolderMethod;
    }
    if (!bridgeClass || !method) {
        if (attached) g_vm->DetachCurrentThread();
        return "";
    }
    jstring state = reinterpret_cast<jstring>(env->CallStaticObjectMethod(
            bridgeClass, method, kSfbExtraPickerTimeoutMs));
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        state = nullptr;
        g_safLastState.store(2);
    }
    std::string encoded = ReadDynamicPickerState(env, state);
    if (state) env->DeleteLocalRef(state);
    RefreshDynamicDiagnostics(env);
    if (attached) g_vm->DetachCurrentThread();
    return encoded;
}

std::string FirstSfbExtension(Array<ExtensionFilterValue>* filters) {
    std::vector<std::string> values;
    if (ReadFilterExtensions(filters, &values) && !values.empty()) return values.front();
    return "adofai";
}

bool BeginSfbExtraCall() {
    const int previous = g_sfbExtraCallsInFlight.fetch_add(1);
    if (previous == 0 && !WriteMarker(g_sfbExtraCallMarker)) {
        g_markerWriteFailures.fetch_add(1);
        g_sfbExtraCallsInFlight.fetch_sub(1);
        return false;
    }
    return true;
}

void EndSfbExtraCall() {
    if (g_sfbExtraCallsInFlight.fetch_sub(1) == 1) ClearMarker(g_sfbExtraCallMarker);
}

String* HookSfbSaveString(String* title, String* directory, String* defaultName,
                          String* extension, IL2CPP::MethodInfo* methodInfo) {
    g_sfbSaveCalls.fetch_add(1);
    if (!BeginSfbExtraCall()) {
        return g_oldSfbSaveString
                ? g_oldSfbSaveString(title, directory, defaultName, extension, methodInfo)
                : nullptr;
    }
    std::string normalized;
    if (!NormalizeExtension(extension, &normalized)) normalized = "adofai";
    String* result = CreateMonoString(RunDynamicSavePicker(defaultName, normalized));
    g_sfbSaveReturns.fetch_add(1);
    EndSfbExtraCall();
    return result;
}

String* HookSfbSaveFilters(String* title, String* directory, String* defaultName,
                           Array<ExtensionFilterValue>* filters,
                           IL2CPP::MethodInfo* methodInfo) {
    g_sfbSaveCalls.fetch_add(1);
    if (!BeginSfbExtraCall()) {
        return g_oldSfbSaveFilters
                ? g_oldSfbSaveFilters(title, directory, defaultName, filters, methodInfo)
                : nullptr;
    }
    String* result = CreateMonoString(
            RunDynamicSavePicker(defaultName, FirstSfbExtension(filters)));
    g_sfbSaveReturns.fetch_add(1);
    EndSfbExtraCall();
    return result;
}

Array<String*>* HookSfbFolder(String* title, String* directory, bool multiselect,
                              IL2CPP::MethodInfo* methodInfo) {
    g_sfbFolderCalls.fetch_add(1);
    if (!BeginSfbExtraCall()) {
        return g_oldSfbFolder
                ? g_oldSfbFolder(title, directory, multiselect, methodInfo)
                : nullptr;
    }
    Array<String*>* result = ToManagedStringArray(RunDynamicFolderPicker());
    g_sfbFolderReturns.fetch_add(1);
    EndSfbExtraCall();
    return result;
}

bool SfbMethodMatches(const MethodBase& method, const char* name,
                      const std::vector<std::string>& params,
                      const char* returnType) {
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    if (!info || !info->methodPointer || !method._isStatic || !info->name ||
        std::string(info->name) != name || info->parameters_count != params.size() ||
        !info->return_type || MetadataTypeName(info->return_type) != returnType) return false;
    for (size_t i = 0; i < params.size(); ++i) {
        const IL2CPP::Il2CppType* p = info->parameters ? info->parameters[i] : nullptr;
        if (!p || TypeByRef(p) != 0 || MetadataTypeName(p) != params[i]) return false;
    }
    return true;
}

MethodBase FindSfbMethod(const Class& browser, const char* name,
                         const std::vector<std::string>& params,
                         const char* returnType) {
    if (!browser) return {};
    for (const MethodBase& method : browser.GetMethods(false)) {
        if (SfbMethodMatches(method, name, params, returnType)) return method;
    }
    return {};
}

void MaybeInstallSfbExtraHooks() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_sfbExtraHookInstalled.load() || g_sfbExtraInstallAttempted.load() ||
        g_sfbExtraRecoveryState.load() != 0 || !DynamicBridgeReady()) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_sfbExtraHookInstalled.load() || g_sfbExtraInstallAttempted.load()) return;

    Class browser("SFB", "StandaloneFileBrowser");
    MethodBase saveString = FindSfbMethod(browser, "SaveFilePanel",
            {"System.String", "System.String", "System.String", "System.String"},
            "System.String");
    MethodBase saveFilters = FindSfbMethod(browser, "SaveFilePanel",
            {"System.String", "System.String", "System.String", "SFB.ExtensionFilter[]"},
            "System.String");
    MethodBase folder = FindSfbMethod(browser, "OpenFolderPanel",
            {"System.String", "System.String", "System.Boolean"},
            "System.String[]");
    const bool abi = saveString.IsValid() && saveFilters.IsValid() && folder.IsValid() &&
            g_dynamicSaveMethod != nullptr && g_dynamicFolderMethod != nullptr;
    g_sfbExtraAbiGuard.store(abi ? 1 : 0);
    if (!abi || !PrepareSfbExtraFuse()) return;

    if (!WriteMarker(g_sfbExtraInstallMarker)) {
        g_sfbExtraMarkerReady.store(0);
        g_sfbExtraRecoveryState.store(4);
        return;
    }
    g_sfbExtraInstallAttempted.store(1);
    BasicHook(saveString, HookSfbSaveString, g_oldSfbSaveString);
    BasicHook(saveFilters, HookSfbSaveFilters, g_oldSfbSaveFilters);
    BasicHook(folder, HookSfbFolder, g_oldSfbFolder);
    const bool installed = g_oldSfbSaveString && g_oldSfbSaveFilters && g_oldSfbFolder;
    g_sfbExtraHookInstalled.store(installed ? 1 : 0);
    if (installed) ClearMarker(g_sfbExtraInstallMarker);
    else g_sfbExtraRecoveryState.store(5);
}

'''

once(
    "void ReconcileInstallState() {\n",
    r26_code + "void ReconcileInstallState() {\n"
)
once(
    """    MaybeInstallTileR21();
    MaybeInstallSfbHook();
""",
    """    MaybeInstallTileR21();
    MaybeInstallSfbExtraHooks();
    MaybeInstallSfbHook();
"""
)

# R24's top-level count describes active installed hook sites.
once(
    r"""                                        (g_calibrationR22HookInstalled.load() ? 1 : 0) +
                                        (g_tileR21HookInstalled.load() ? 2 : 0)) << '\n'
""",
    r"""                                        (g_calibrationR22HookInstalled.load() ? 1 : 0) +
                                        (g_tileR21HookInstalled.load() ? 2 : 0) +
                                        (g_sfbExtraHookInstalled.load() ? 3 : 0)) << '\n'
"""
)
once(
    'activeHookPolicy=sfb-1-calibrationR22-1-tileR21-2',
    'activeHookPolicy=sfb-open-1-sfb-save-folder-3-calibrationR22-1-tileR21-2'
)
once(
    '        << "historicalInactiveHookSitesExcluded=1\\n"\n',
    '        << "historicalInactiveHookSitesExcluded=1\\n"\n'
    '        << "sfbExtraRevision=26\\n"\n'
    '        << "sfbExtraPolicy=exact-v240-sync-SaveFilePanel-OpenFolderPanel-parent-SAF" << \'\\n\'\n'
    '        << "sfbExtraAbiGuard=" << g_sfbExtraAbiGuard.load() << \'\\n\'\n'
    '        << "sfbExtraBridgeReady=" << g_sfbExtraBridgeReady.load() << \'\\n\'\n'
    '        << "sfbExtraHookInstalled=" << g_sfbExtraHookInstalled.load() << \'\\n\'\n'
    '        << "sfbExtraInstallAttempted=" << g_sfbExtraInstallAttempted.load() << \'\\n\'\n'
    '        << "sfbExtraMarkerReady=" << g_sfbExtraMarkerReady.load() << \'\\n\'\n'
    '        << "sfbExtraRecoveryState=" << g_sfbExtraRecoveryState.load() << \'\\n\'\n'
    '        << "sfbSaveCalls=" << g_sfbSaveCalls.load() << \'\\n\'\n'
    '        << "sfbSaveReturns=" << g_sfbSaveReturns.load() << \'\\n\'\n'
    '        << "sfbFolderCalls=" << g_sfbFolderCalls.load() << \'\\n\'\n'
    '        << "sfbFolderReturns=" << g_sfbFolderReturns.load() << \'\\n\'\n'
)

for marker in (
    "sfbExtraRevision=26",
    "sfbExtraPolicy=exact-v240-sync-SaveFilePanel-OpenFolderPanel-parent-SAF",
    'bridgeClass, "save",',
    'bridgeClass, "folder", "(J)Ljava/lang/String;"',
    'FindSfbMethod(browser, "SaveFilePanel"',
    'FindSfbMethod(browser, "OpenFolderPanel"',
    'SFB.ExtensionFilter[]',
    'BasicHook(saveString, HookSfbSaveString, g_oldSfbSaveString)',
    'BasicHook(saveFilters, HookSfbSaveFilters, g_oldSfbSaveFilters)',
    'BasicHook(folder, HookSfbFolder, g_oldSfbFolder)',
    'MaybeInstallSfbExtraHooks();',
    'sfb-r26-extra-install.pending',
    'sfb-r26-extra-call.pending',
):
    if marker not in s:
        raise SystemExit(f"r26 marker missing: {marker}")

if s.count("BasicHook(") != 16:
    raise SystemExit(f"r26 expected sixteen compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
r27 = Path(__file__).with_name("apply-v240-r27-hot-fps-bridge.py")
if not r27.is_file():
    raise SystemExit(f"missing r27 overlay: {r27}")
__import__("subprocess").run([sys.executable, str(r27), str(path)], check=True)
