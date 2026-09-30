#!/usr/bin/env python3
"""R29: make r28's exact Physics2D.SyncTransforms resolver robust across Android namespaces.

Real-device evidence from bootstrap v3 + runtime 02e6d998 showed:
- the hot runtime was healthy and hash-verified;
- the r28 tile policy was present;
- but tileR21HookInstalled=0, tileR21AbiGuard=0, tileR21IcallReady=0,
  tileR21InstallAttempted=0.

The exact v2.4 libil2cpp.so exports il2cpp_resolve_icall and exact libunity.so contains
UnityEngine.Physics2D::SyncTransforms. R29 therefore preserves r28's ownership boundary,
hook set, raycast arguments/results, and fail-open semantics while resolving the exported
icall resolver through RTLD_DEFAULT first and the loaded libil2cpp handle as an Android
namespace fallback. It also records each exact gate so a future device report can distinguish
class/ABI failure from resolver visibility without adding probes or mutations.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r29-il2cpp-resolver-fallback.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r29 anchor must occur once, got {n}: {old[:220]!r}")
    s = s.replace(old, new, 1)


once(
    'std::atomic<int> g_tileR28SyncWithoutTouchCalls{0};\n',
    'std::atomic<int> g_tileR28SyncWithoutTouchCalls{0};\n'
    'std::atomic<int> g_tileR29EditorClassReady{0};\n'
    'std::atomic<int> g_tileR29PhysicsClassReady{0};\n'
    'std::atomic<int> g_tileR29PrimitiveTypesReady{0};\n'
    'std::atomic<int> g_tileR29ObjectsAbiReady{0};\n'
    'std::atomic<int> g_tileR29RaycastAbiReady{0};\n'
    'std::atomic<int> g_tileR29ResolveSymbolPath{0};\n'
    'std::atomic<int> g_tileR29ResolveIcallResult{0};\n'
)

resolver = r'''
ResolveIcallFn ResolveIcallR29(void* managedMethodPointer) {
    auto direct = reinterpret_cast<ResolveIcallFn>(
            dlsym(RTLD_DEFAULT, "il2cpp_resolve_icall"));
    if (direct) {
        g_tileR29ResolveSymbolPath.store(1);
        return direct;
    }

    // Resolve the already-loaded owner from an exact v2.4 managed method pointer.
    // RTLD_NOLOAD guarantees this fallback can never create a second IL2CPP runtime.
    Dl_info owner{};
    if (managedMethodPointer &&
        dladdr(managedMethodPointer, &owner) != 0 &&
        owner.dli_fname != nullptr) {
        void* handle = dlopen(owner.dli_fname, RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            auto viaOwner = reinterpret_cast<ResolveIcallFn>(
                    dlsym(handle, "il2cpp_resolve_icall"));
            dlclose(handle);
            if (viaOwner) {
                g_tileR29ResolveSymbolPath.store(2);
                return viaOwner;
            }
        }
    }

    // Some Android linker namespaces accept the SONAME even when RTLD_DEFAULT does not
    // expose its globals. Still use NOLOAD so this remains lookup-only.
    void* handle = dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD);
    if (handle) {
        auto viaSoname = reinterpret_cast<ResolveIcallFn>(
                dlsym(handle, "il2cpp_resolve_icall"));
        dlclose(handle);
        if (viaSoname) {
            g_tileR29ResolveSymbolPath.store(3);
            return viaSoname;
        }
    }

    g_tileR29ResolveSymbolPath.store(0);
    return nullptr;
}

'''
once('void MaybeInstallTileR21() {\n', resolver + 'void MaybeInstallTileR21() {\n')

once(
    '    Class intClass = Defaults::Get<int>();\n'
    '    if (!editor || !physics || !vector2 || !floatClass || !intClass) return;\n',
    '    Class intClass = Defaults::Get<int>();\n'
    '    g_tileR29EditorClassReady.store(editor ? 1 : 0);\n'
    '    g_tileR29PhysicsClassReady.store(physics ? 1 : 0);\n'
    '    g_tileR29PrimitiveTypesReady.store((vector2 && floatClass && intClass) ? 1 : 0);\n'
    '    if (!editor || !physics || !vector2 || !floatClass || !intClass) return;\n'
)

once(
    '    const bool raycastParams = raycastInfo && raycastInfo->parameters_count == 4 &&\n',
    '    g_tileR29ObjectsAbiReady.store(objectsAbi ? 1 : 0);\n\n'
    '    const bool raycastParams = raycastInfo && raycastInfo->parameters_count == 4 &&\n'
)

once(
    '    if (input) g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);\n',
    '    g_tileR29RaycastAbiReady.store(raycastAbi ? 1 : 0);\n\n'
    '    if (input) g_tileR21GetTouchCount = input.GetMethod("get_touchCount", 0);\n'
)

once(
    '    auto resolve = reinterpret_cast<ResolveIcallFn>(\n'
    '            dlsym(RTLD_DEFAULT, "il2cpp_resolve_icall"));\n'
    '    PhysicsSyncFn sync = nullptr;\n'
    '    if (resolve) {\n'
    '        sync = reinterpret_cast<PhysicsSyncFn>(\n'
    '                resolve("UnityEngine.Physics2D::SyncTransforms()"));\n'
    '        if (!sync) {\n'
    '            sync = reinterpret_cast<PhysicsSyncFn>(\n'
    '                    resolve("UnityEngine.Physics2D::SyncTransforms"));\n'
    '        }\n'
    '    }\n'
    '    g_tileR21IcallReady.store(sync ? 1 : 0);\n',
    '    auto resolve = ResolveIcallR29(\n'
    '            raycastInfo ? reinterpret_cast<void*>(raycastInfo->methodPointer) : nullptr);\n'
    '    PhysicsSyncFn sync = nullptr;\n'
    '    if (resolve) {\n'
    '        sync = reinterpret_cast<PhysicsSyncFn>(\n'
    '                resolve("UnityEngine.Physics2D::SyncTransforms()"));\n'
    '        if (!sync) {\n'
    '            sync = reinterpret_cast<PhysicsSyncFn>(\n'
    '                    resolve("UnityEngine.Physics2D::SyncTransforms"));\n'
    '        }\n'
    '    }\n'
    '    g_tileR29ResolveIcallResult.store(sync ? 1 : 0);\n'
    '    g_tileR21IcallReady.store(sync ? 1 : 0);\n'
)

once(
    '        << "tileR28SyncWithoutTouchCalls=" << g_tileR28SyncWithoutTouchCalls.load() << \'\\n\'\n',
    '        << "tileR28SyncWithoutTouchCalls=" << g_tileR28SyncWithoutTouchCalls.load() << \'\\n\'\n'
    '        << "tileResolverRevision=29" << \'\\n\'\n'
    '        << "tileR29ResolverPolicy=RTLD_DEFAULT-then-method-owner-NOLOAD-then-SONAME-NOLOAD" << \'\\n\'\n'
    '        << "tileR29EditorClassReady=" << g_tileR29EditorClassReady.load() << \'\\n\'\n'
    '        << "tileR29PhysicsClassReady=" << g_tileR29PhysicsClassReady.load() << \'\\n\'\n'
    '        << "tileR29PrimitiveTypesReady=" << g_tileR29PrimitiveTypesReady.load() << \'\\n\'\n'
    '        << "tileR29ObjectsAbiReady=" << g_tileR29ObjectsAbiReady.load() << \'\\n\'\n'
    '        << "tileR29RaycastAbiReady=" << g_tileR29RaycastAbiReady.load() << \'\\n\'\n'
    '        << "tileR29ResolveSymbolPath=" << g_tileR29ResolveSymbolPath.load() << \'\\n\'\n'
    '        << "tileR29ResolveIcallResult=" << g_tileR29ResolveIcallResult.load() << \'\\n\'\n'
)

for marker in (
    "tileResolverRevision=29",
    "tileR29ResolverPolicy=RTLD_DEFAULT-then-method-owner-NOLOAD-then-SONAME-NOLOAD",
    "ResolveIcallR29(void* managedMethodPointer)",
    'dlopen("libil2cpp.so", RTLD_NOW | RTLD_NOLOAD)',
    'dlsym(handle, "il2cpp_resolve_icall")',
    "tileR29EditorClassReady=",
    "tileR29PhysicsClassReady=",
    "tileR29PrimitiveTypesReady=",
    "tileR29ObjectsAbiReady=",
    "tileR29RaycastAbiReady=",
    "tileR29ResolveSymbolPath=",
    "tileR29ResolveIcallResult=",
):
    if marker not in s:
        raise SystemExit(f"r29 marker missing: {marker}")

if s.count("BasicHook(") != 18:
    raise SystemExit(f"r29 adds no hooks; expected 18 compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
