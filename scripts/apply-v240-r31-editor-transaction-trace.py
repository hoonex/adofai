#!/usr/bin/env python3
"""R31: bounded read-only editor pointer-transaction trace.

R30 proved that isolated raycast counters are insufficient to explain the phone-only
selection failure. R31 correlates one legacy left-pointer transaction across the exact
v2.4 editor state-machine boundaries without changing coordinates, selection, dragging,
raycasts, or editor fields.

Active R31 hooks are exact-ABI guarded and pass-through only:
- scnEditor.HandleMouseActions()
- scnEditor.SelectFloor(scrFloor, bool)
- scnEditor.SmartObjectSelect(bool)
- scnEditor.GizmoAtMouse()
- scnEditor.DragCamera(Vector3)
- scnEditor.DragTilesStart()
- scnEditor.DragTiles(Vector3)

The already-active R30 ObjectsAtMouse/RaycastAll wrappers feed the same transaction.
Only the latest eight transactions are retained. There is no per-frame logging.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r31-editor-transaction-trace.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"r31 anchor must occur once, got {n}: {old[:220]!r}")
    s = s.replace(old, new, 1)


state = r'''
constexpr int kEditorTraceR31Slots = 8;
constexpr int kEditorTraceR31HookCount = 7;
constexpr int kEditorTraceR31HandleBit = 1 << 0;
constexpr int kEditorTraceR31SelectBit = 1 << 1;
constexpr int kEditorTraceR31SmartBit = 1 << 2;
constexpr int kEditorTraceR31GizmoBit = 1 << 3;
constexpr int kEditorTraceR31DragCameraBit = 1 << 4;
constexpr int kEditorTraceR31DragTilesStartBit = 1 << 5;
constexpr int kEditorTraceR31DragTilesBit = 1 << 6;

struct EditorTraceR31Txn {
    std::atomic<int> seq{0};
    std::atomic<int> state{0}; // 0 empty, 1 active, 2 completed, 3 superseded.
    std::atomic<int> startX100{0};
    std::atomic<int> startY100{0};
    std::atomic<int> endX100{0};
    std::atomic<int> endY100{0};
    std::atomic<int> screenW{0};
    std::atomic<int> screenH{0};
    std::atomic<int> touchStart{0};
    std::atomic<int> handleFrames{0};
    std::atomic<int> heldFrames{0};
    std::atomic<int> releaseSeen{0};
    std::atomic<int> postReleaseFrames{0};
    std::atomic<int> maxDx100{0};
    std::atomic<int> maxDy100{0};
    std::atomic<int> objectsCalls{0};
    std::atomic<int> objectsLastCount{-3};
    std::atomic<int> raycastCalls{0};
    std::atomic<int> ray1Count{-3};
    std::atomic<int> ray2Count{-3};
    std::atomic<int> ray1X1000{0};
    std::atomic<int> ray1Y1000{0};
    std::atomic<int> ray2X1000{0};
    std::atomic<int> ray2Y1000{0};
    std::atomic<int> ray1Mask{0};
    std::atomic<int> ray2Mask{0};
    std::atomic<int> smartCalls{0};
    std::atomic<int> smartNonNull{0};
    std::atomic<int> gizmoCalls{0};
    std::atomic<int> gizmoNonNull{0};
    std::atomic<int> selectCalls{0};
    std::atomic<int> selectNonNull{0};
    std::atomic<int> selectCameraJump{0};
    std::atomic<int> dragCameraCalls{0};
    std::atomic<int> dragTilesStartCalls{0};
    std::atomic<int> dragTilesCalls{0};
};

EditorTraceR31Txn g_editorTraceR31Txns[kEditorTraceR31Slots];
std::atomic<int> g_editorTraceR31NextSeq{0};
std::atomic<int> g_editorTraceR31AbiMask{0};
std::atomic<int> g_editorTraceR31InputAbiGuard{0};
std::atomic<int> g_editorTraceR31InstalledMask{0};
std::atomic<int> g_editorTraceR31InstallAttempted{0};
std::atomic<int> g_editorTraceR31MarkerReady{0};
std::atomic<int> g_editorTraceR31RecoveryState{0};
std::atomic<int> g_editorTraceR31CallProven[kEditorTraceR31HookCount]{};
std::mutex g_editorTraceR31CallMutex[kEditorTraceR31HookCount];
std::string g_editorTraceR31InstallMarker;
std::string g_editorTraceR31CallMarkers[kEditorTraceR31HookCount];

Method<Vector3> g_editorTraceR31GetMousePosition;
Method<int> g_editorTraceR31GetTouchCount;
Method<int> g_editorTraceR31GetScreenWidth;
Method<int> g_editorTraceR31GetScreenHeight;
Method<bool> g_editorTraceR31GetMouseButton;
Method<bool> g_editorTraceR31GetMouseButtonDown;
Method<bool> g_editorTraceR31GetMouseButtonUp;

thread_local int g_editorTraceR31ActiveSlot = -1;
thread_local int g_editorTraceR31ActiveSeq = 0;
thread_local bool g_editorTraceR31FinishOnNextHandle = false;

using EditorTraceR31VoidSelf = void (*)(IL2CPP::Il2CppObject*, IL2CPP::MethodInfo*);
using EditorTraceR31Select = void (*)(IL2CPP::Il2CppObject*, IL2CPP::Il2CppObject*, bool,
                                     IL2CPP::MethodInfo*);
using EditorTraceR31BoolObject = IL2CPP::Il2CppObject* (*)(IL2CPP::Il2CppObject*, bool,
                                                          IL2CPP::MethodInfo*);
using EditorTraceR31ObjectSelf = IL2CPP::Il2CppObject* (*)(IL2CPP::Il2CppObject*,
                                                          IL2CPP::MethodInfo*);
using EditorTraceR31Vector3 = void (*)(IL2CPP::Il2CppObject*, Vector3, IL2CPP::MethodInfo*);

EditorTraceR31VoidSelf g_oldEditorTraceR31Handle = nullptr;
EditorTraceR31Select g_oldEditorTraceR31Select = nullptr;
EditorTraceR31BoolObject g_oldEditorTraceR31Smart = nullptr;
EditorTraceR31ObjectSelf g_oldEditorTraceR31Gizmo = nullptr;
EditorTraceR31Vector3 g_oldEditorTraceR31DragCamera = nullptr;
EditorTraceR31VoidSelf g_oldEditorTraceR31DragTilesStart = nullptr;
EditorTraceR31Vector3 g_oldEditorTraceR31DragTiles = nullptr;

void EditorTraceR31AtomicMax(std::atomic<int>& target, int value) {
    int current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
}

EditorTraceR31Txn* EditorTraceR31ActiveTxn() {
    if (g_editorTraceR31ActiveSlot < 0 ||
        g_editorTraceR31ActiveSlot >= kEditorTraceR31Slots ||
        g_editorTraceR31ActiveSeq <= 0) return nullptr;
    EditorTraceR31Txn& txn = g_editorTraceR31Txns[g_editorTraceR31ActiveSlot];
    if (txn.seq.load(std::memory_order_acquire) != g_editorTraceR31ActiveSeq ||
        txn.state.load(std::memory_order_relaxed) != 1) return nullptr;
    return &txn;
}

void EditorTraceR31ResetTxn(EditorTraceR31Txn& txn, int seq, const Vector3& mouse,
                            int width, int height, int touchCount) {
    txn.state.store(0, std::memory_order_relaxed);
    txn.startX100.store(static_cast<int>(mouse.x * 100.0f), std::memory_order_relaxed);
    txn.startY100.store(static_cast<int>(mouse.y * 100.0f), std::memory_order_relaxed);
    txn.endX100.store(static_cast<int>(mouse.x * 100.0f), std::memory_order_relaxed);
    txn.endY100.store(static_cast<int>(mouse.y * 100.0f), std::memory_order_relaxed);
    txn.screenW.store(width, std::memory_order_relaxed);
    txn.screenH.store(height, std::memory_order_relaxed);
    txn.touchStart.store(touchCount, std::memory_order_relaxed);
    txn.handleFrames.store(0, std::memory_order_relaxed);
    txn.heldFrames.store(0, std::memory_order_relaxed);
    txn.releaseSeen.store(0, std::memory_order_relaxed);
    txn.postReleaseFrames.store(0, std::memory_order_relaxed);
    txn.maxDx100.store(0, std::memory_order_relaxed);
    txn.maxDy100.store(0, std::memory_order_relaxed);
    txn.objectsCalls.store(0, std::memory_order_relaxed);
    txn.objectsLastCount.store(-3, std::memory_order_relaxed);
    txn.raycastCalls.store(0, std::memory_order_relaxed);
    txn.ray1Count.store(-3, std::memory_order_relaxed);
    txn.ray2Count.store(-3, std::memory_order_relaxed);
    txn.ray1X1000.store(0, std::memory_order_relaxed);
    txn.ray1Y1000.store(0, std::memory_order_relaxed);
    txn.ray2X1000.store(0, std::memory_order_relaxed);
    txn.ray2Y1000.store(0, std::memory_order_relaxed);
    txn.ray1Mask.store(0, std::memory_order_relaxed);
    txn.ray2Mask.store(0, std::memory_order_relaxed);
    txn.smartCalls.store(0, std::memory_order_relaxed);
    txn.smartNonNull.store(0, std::memory_order_relaxed);
    txn.gizmoCalls.store(0, std::memory_order_relaxed);
    txn.gizmoNonNull.store(0, std::memory_order_relaxed);
    txn.selectCalls.store(0, std::memory_order_relaxed);
    txn.selectNonNull.store(0, std::memory_order_relaxed);
    txn.selectCameraJump.store(0, std::memory_order_relaxed);
    txn.dragCameraCalls.store(0, std::memory_order_relaxed);
    txn.dragTilesStartCalls.store(0, std::memory_order_relaxed);
    txn.dragTilesCalls.store(0, std::memory_order_relaxed);
    txn.seq.store(seq, std::memory_order_release);
    txn.state.store(1, std::memory_order_release);
}

void EditorTraceR31BeginTxn(const Vector3& mouse, int width, int height, int touchCount) {
    if (EditorTraceR31Txn* previous = EditorTraceR31ActiveTxn()) {
        previous->state.store(3, std::memory_order_release);
    }
    g_editorTraceR31FinishOnNextHandle = false;
    const int seq = g_editorTraceR31NextSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    const int slot = (seq - 1) % kEditorTraceR31Slots;
    EditorTraceR31ResetTxn(g_editorTraceR31Txns[slot], seq, mouse, width, height, touchCount);
    g_editorTraceR31ActiveSlot = slot;
    g_editorTraceR31ActiveSeq = seq;
}

void EditorTraceR31UpdatePointer(const Vector3& mouse, bool held) {
    EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn();
    if (!txn) return;
    const int x100 = static_cast<int>(mouse.x * 100.0f);
    const int y100 = static_cast<int>(mouse.y * 100.0f);
    txn->endX100.store(x100, std::memory_order_relaxed);
    txn->endY100.store(y100, std::memory_order_relaxed);
    txn->handleFrames.fetch_add(1, std::memory_order_relaxed);
    if (held) txn->heldFrames.fetch_add(1, std::memory_order_relaxed);
    const int dx = x100 - txn->startX100.load(std::memory_order_relaxed);
    const int dy = y100 - txn->startY100.load(std::memory_order_relaxed);
    EditorTraceR31AtomicMax(txn->maxDx100, dx < 0 ? -dx : dx);
    EditorTraceR31AtomicMax(txn->maxDy100, dy < 0 ? -dy : dy);
}

void EditorTraceR31FinishTxn() {
    EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn();
    if (txn) txn->state.store(2, std::memory_order_release);
    g_editorTraceR31ActiveSlot = -1;
    g_editorTraceR31ActiveSeq = 0;
    g_editorTraceR31FinishOnNextHandle = false;
}

void EditorTraceR31RecordObjects(int count) {
    EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn();
    if (!txn) return;
    txn->objectsCalls.fetch_add(1, std::memory_order_relaxed);
    txn->objectsLastCount.store(count, std::memory_order_relaxed);
}

void EditorTraceR31RecordRaycast(int ordinal, int count, const Vector2& origin, int mask) {
    EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn();
    if (!txn) return;
    txn->raycastCalls.fetch_add(1, std::memory_order_relaxed);
    if (ordinal == 1) {
        txn->ray1Count.store(count, std::memory_order_relaxed);
        txn->ray1X1000.store(static_cast<int>(origin.x * 1000.0f), std::memory_order_relaxed);
        txn->ray1Y1000.store(static_cast<int>(origin.y * 1000.0f), std::memory_order_relaxed);
        txn->ray1Mask.store(mask, std::memory_order_relaxed);
    } else if (ordinal == 2) {
        txn->ray2Count.store(count, std::memory_order_relaxed);
        txn->ray2X1000.store(static_cast<int>(origin.x * 1000.0f), std::memory_order_relaxed);
        txn->ray2Y1000.store(static_cast<int>(origin.y * 1000.0f), std::memory_order_relaxed);
        txn->ray2Mask.store(mask, std::memory_order_relaxed);
    }
}

int EditorTraceR31CallProvenMask() {
    int mask = 0;
    for (int i = 0; i < kEditorTraceR31HookCount; ++i) {
        if (g_editorTraceR31CallProven[i].load(std::memory_order_relaxed)) mask |= 1 << i;
    }
    return mask;
}

bool EditorTraceR31BeginCanary(int index, bool* first) {
    if (first) *first = false;
    if (index < 0 || index >= kEditorTraceR31HookCount) return false;
    if (g_editorTraceR31CallProven[index].load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> guard(g_editorTraceR31CallMutex[index]);
    if (g_editorTraceR31CallProven[index].load(std::memory_order_relaxed)) return true;
    if (g_editorTraceR31CallMarkers[index].empty() ||
        !WriteMarker(g_editorTraceR31CallMarkers[index])) {
        g_markerWriteFailures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (first) *first = true;
    return true;
}

void EditorTraceR31EndCanary(int index, bool first) {
    if (!first || index < 0 || index >= kEditorTraceR31HookCount) return;
    ClearMarker(g_editorTraceR31CallMarkers[index]);
    g_editorTraceR31CallProven[index].store(1, std::memory_order_release);
}

void AppendEditorTraceR31(std::ostringstream& out) {
    out << "editorTraceR31LatestSeq=" << g_editorTraceR31NextSeq.load() << '\n';
    for (int i = 0; i < kEditorTraceR31Slots; ++i) {
        const EditorTraceR31Txn& t = g_editorTraceR31Txns[i];
        out << "editorTraceR31Txn" << i << '='
            << "seq:" << t.seq.load() << ",state:" << t.state.load()
            << ",start:" << t.startX100.load() << ':' << t.startY100.load()
            << ",end:" << t.endX100.load() << ':' << t.endY100.load()
            << ",screen:" << t.screenW.load() << 'x' << t.screenH.load()
            << ",touch:" << t.touchStart.load()
            << ",hf:" << t.handleFrames.load() << ",held:" << t.heldFrames.load()
            << ",release:" << t.releaseSeen.load() << ':' << t.postReleaseFrames.load()
            << ",maxd:" << t.maxDx100.load() << ':' << t.maxDy100.load()
            << ",obj:" << t.objectsCalls.load() << ':' << t.objectsLastCount.load()
            << ",ray:" << t.raycastCalls.load() << ':' << t.ray1Count.load()
            << ':' << t.ray2Count.load()
            << ",r1:" << t.ray1X1000.load() << ':' << t.ray1Y1000.load()
            << ':' << t.ray1Mask.load()
            << ",r2:" << t.ray2X1000.load() << ':' << t.ray2Y1000.load()
            << ':' << t.ray2Mask.load()
            << ",smart:" << t.smartCalls.load() << ':' << t.smartNonNull.load()
            << ",gizmo:" << t.gizmoCalls.load() << ':' << t.gizmoNonNull.load()
            << ",select:" << t.selectCalls.load() << ':' << t.selectNonNull.load()
            << ':' << t.selectCameraJump.load()
            << ",drag:" << t.dragCameraCalls.load() << ':'
            << t.dragTilesStartCalls.load() << ':' << t.dragTilesCalls.load()
            << '\n';
    }
}

'''
once(
    "constexpr size_t kTileR30MaxObservedArray = 4096;\n",
    state + "constexpr size_t kTileR30MaxObservedArray = 4096;\n",
)

once(
    "        const int count = ObserveTileR30ArrayCount(result);\n"
    "        g_tileR30ObjectsLastCount.store(count, std::memory_order_relaxed);\n",
    "        const int count = ObserveTileR30ArrayCount(result);\n"
    "        EditorTraceR31RecordObjects(count);\n"
    "        g_tileR30ObjectsLastCount.store(count, std::memory_order_relaxed);\n",
)

once(
    "    const int count = ObserveTileR30ArrayCount(result);\n"
    "    if (count == -1) {\n",
    "    const int count = ObserveTileR30ArrayCount(result);\n"
    "    EditorTraceR31RecordRaycast(ordinal, count, origin, layerMask);\n"
    "    if (count == -1) {\n",
)

hooks = r'''
void HookEditorTraceR31Handle(IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31Handle) return;
    bool first = false;
    if (!EditorTraceR31BeginCanary(0, &first)) {
        g_oldEditorTraceR31Handle(self, methodInfo);
        return;
    }

    const bool down = g_editorTraceR31GetMouseButtonDown.IsValid() &&
            g_editorTraceR31GetMouseButtonDown.Call(0);
    const bool held = g_editorTraceR31GetMouseButton.IsValid() &&
            g_editorTraceR31GetMouseButton.Call(0);
    const bool up = g_editorTraceR31GetMouseButtonUp.IsValid() &&
            g_editorTraceR31GetMouseButtonUp.Call(0);
    const Vector3 mouse = g_editorTraceR31GetMousePosition.Call();
    const int touchCount = g_editorTraceR31GetTouchCount.Call();
    const int width = g_editorTraceR31GetScreenWidth.Call();
    const int height = g_editorTraceR31GetScreenHeight.Call();

    const bool finishReleasedAfterThisHandle =
            g_editorTraceR31FinishOnNextHandle && !down;
    if (down) EditorTraceR31BeginTxn(mouse, width, height, touchCount);
    EditorTraceR31UpdatePointer(mouse, held);
    g_oldEditorTraceR31Handle(self, methodInfo);

    if (up) {
        if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
            txn->releaseSeen.store(1, std::memory_order_relaxed);
        }
        g_editorTraceR31FinishOnNextHandle = true;
    } else if (finishReleasedAfterThisHandle) {
        if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
            txn->postReleaseFrames.fetch_add(1, std::memory_order_relaxed);
        }
        EditorTraceR31FinishTxn();
    }
    EditorTraceR31EndCanary(0, first);
}

void HookEditorTraceR31Select(IL2CPP::Il2CppObject* self, IL2CPP::Il2CppObject* floor,
                              bool cameraJump, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31Select) return;
    bool first = false;
    if (!EditorTraceR31BeginCanary(1, &first)) {
        g_oldEditorTraceR31Select(self, floor, cameraJump, methodInfo);
        return;
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->selectCalls.fetch_add(1, std::memory_order_relaxed);
        if (floor) txn->selectNonNull.fetch_add(1, std::memory_order_relaxed);
        txn->selectCameraJump.store(cameraJump ? 1 : 0, std::memory_order_relaxed);
    }
    g_oldEditorTraceR31Select(self, floor, cameraJump, methodInfo);
    EditorTraceR31EndCanary(1, first);
}

IL2CPP::Il2CppObject* HookEditorTraceR31Smart(
        IL2CPP::Il2CppObject* self, bool allowCycling, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31Smart) return nullptr;
    bool first = false;
    if (!EditorTraceR31BeginCanary(2, &first)) {
        return g_oldEditorTraceR31Smart(self, allowCycling, methodInfo);
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->smartCalls.fetch_add(1, std::memory_order_relaxed);
    }
    IL2CPP::Il2CppObject* result =
            g_oldEditorTraceR31Smart(self, allowCycling, methodInfo);
    if (result) {
        if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
            txn->smartNonNull.fetch_add(1, std::memory_order_relaxed);
        }
    }
    EditorTraceR31EndCanary(2, first);
    return result;
}

IL2CPP::Il2CppObject* HookEditorTraceR31Gizmo(
        IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31Gizmo) return nullptr;
    bool first = false;
    if (!EditorTraceR31BeginCanary(3, &first)) {
        return g_oldEditorTraceR31Gizmo(self, methodInfo);
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->gizmoCalls.fetch_add(1, std::memory_order_relaxed);
    }
    IL2CPP::Il2CppObject* result = g_oldEditorTraceR31Gizmo(self, methodInfo);
    if (result) {
        if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
            txn->gizmoNonNull.fetch_add(1, std::memory_order_relaxed);
        }
    }
    EditorTraceR31EndCanary(3, first);
    return result;
}

void HookEditorTraceR31DragCamera(
        IL2CPP::Il2CppObject* self, Vector3 delta, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31DragCamera) return;
    bool first = false;
    if (!EditorTraceR31BeginCanary(4, &first)) {
        g_oldEditorTraceR31DragCamera(self, delta, methodInfo);
        return;
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->dragCameraCalls.fetch_add(1, std::memory_order_relaxed);
    }
    g_oldEditorTraceR31DragCamera(self, delta, methodInfo);
    EditorTraceR31EndCanary(4, first);
}

void HookEditorTraceR31DragTilesStart(
        IL2CPP::Il2CppObject* self, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31DragTilesStart) return;
    bool first = false;
    if (!EditorTraceR31BeginCanary(5, &first)) {
        g_oldEditorTraceR31DragTilesStart(self, methodInfo);
        return;
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->dragTilesStartCalls.fetch_add(1, std::memory_order_relaxed);
    }
    g_oldEditorTraceR31DragTilesStart(self, methodInfo);
    EditorTraceR31EndCanary(5, first);
}

void HookEditorTraceR31DragTiles(
        IL2CPP::Il2CppObject* self, Vector3 delta, IL2CPP::MethodInfo* methodInfo) {
    if (!g_oldEditorTraceR31DragTiles) return;
    bool first = false;
    if (!EditorTraceR31BeginCanary(6, &first)) {
        g_oldEditorTraceR31DragTiles(self, delta, methodInfo);
        return;
    }
    if (EditorTraceR31Txn* txn = EditorTraceR31ActiveTxn()) {
        txn->dragTilesCalls.fetch_add(1, std::memory_order_relaxed);
    }
    g_oldEditorTraceR31DragTiles(self, delta, methodInfo);
    EditorTraceR31EndCanary(6, first);
}

bool EditorTraceR31VoidInstance0Abi(const MethodBase& method) {
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    return info && info->methodPointer && !method._isStatic &&
            info->parameters_count == 0 && info->return_type &&
            TypeCode(info->return_type) == 1;
}

bool EditorTraceR31Vector3VoidAbi(const MethodBase& method, const Class& vector3Class) {
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    return info && info->methodPointer && !method._isStatic &&
            info->parameters_count == 1 && info->parameters &&
            info->parameters[0] && TypeByRef(info->parameters[0]) == 0 &&
            SameClass(Class(info->parameters[0]), vector3Class) &&
            info->return_type && TypeCode(info->return_type) == 1;
}

bool EditorTraceR31StaticInt0Abi(const MethodBase& method, const Class& intClass) {
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    return info && info->methodPointer && method._isStatic &&
            info->parameters_count == 0 && info->return_type &&
            SameClass(Class(info->return_type), intClass);
}

bool EditorTraceR31StaticMouseButtonAbi(
        const MethodBase& method, const Class& intClass, const Class& boolClass) {
    IL2CPP::MethodInfo* info = method.IsValid() ? method.GetInfo() : nullptr;
    return info && info->methodPointer && method._isStatic &&
            info->parameters_count == 1 && info->parameters &&
            info->parameters[0] && TypeByRef(info->parameters[0]) == 0 &&
            SameClass(Class(info->parameters[0]), intClass) &&
            info->return_type && SameClass(Class(info->return_type), boolClass);
}

bool PrepareEditorTraceR31Fuse() {
    const std::string dir = RuntimeDir();
    if (dir.empty()) {
        g_editorTraceR31RecoveryState.store(10);
        return false;
    }
    g_editorTraceR31InstallMarker = dir + "/editor-r31-trace-install.pending";
    for (int i = 0; i < kEditorTraceR31HookCount; ++i) {
        g_editorTraceR31CallMarkers[i] =
                dir + "/editor-r31-trace-call-" + std::to_string(i) + ".pending";
    }
    g_editorTraceR31MarkerReady.store(1);
    if (MarkerExists(g_editorTraceR31InstallMarker)) {
        g_editorTraceR31RecoveryState.store(1);
        return false;
    }
    for (int i = 0; i < kEditorTraceR31HookCount; ++i) {
        if (MarkerExists(g_editorTraceR31CallMarkers[i])) {
            g_editorTraceR31RecoveryState.store(2 + i);
            return false;
        }
    }
    const std::string probe = dir + "/editor-r31-trace-probe.tmp";
    ClearMarker(probe);
    if (!WriteMarker(probe)) {
        g_editorTraceR31MarkerReady.store(0);
        g_editorTraceR31RecoveryState.store(10);
        return false;
    }
    ClearMarker(probe);
    return true;
}

void MaybeInstallEditorTraceR31() {
    if (!g_bnmLoadedCallback.load(std::memory_order_acquire) ||
        g_editorTraceR31InstallAttempted.load(std::memory_order_acquire) ||
        g_editorTraceR31RecoveryState.load(std::memory_order_relaxed) != 0) return;

    std::lock_guard<std::mutex> guard(g_installMutex);
    if (g_editorTraceR31InstallAttempted.load(std::memory_order_relaxed)) return;

    Class editor("", "scnEditor");
    Class floor("", "scrFloor");
    Class gameObject("UnityEngine", "GameObject");
    Class gizmo("ADOFAI", "TransformGizmo");
    Class input("UnityEngine", "Input");
    Class screen("UnityEngine", "Screen");
    Class boolClass = Defaults::Get<bool>();
    Class intClass = Defaults::Get<int>();
    Class vector3Class = Defaults::Get<Vector3>();
    if (!editor || !floor || !gameObject || !gizmo || !input || !screen ||
        !boolClass || !intClass || !vector3Class) return;

    MethodBase handle = editor.GetMethod("HandleMouseActions", 0);
    MethodBase select = editor.GetMethod("SelectFloor", 2);
    MethodBase smart = editor.GetMethod("SmartObjectSelect", 1);
    MethodBase gizmoAtMouse = editor.GetMethod("GizmoAtMouse", 0);
    MethodBase dragCamera = editor.GetMethod("DragCamera", 1);
    MethodBase dragTilesStart = editor.GetMethod("DragTilesStart", 0);
    MethodBase dragTiles = editor.GetMethod("DragTiles", 1);

    int abiMask = 0;
    const bool handleAbi = EditorTraceR31VoidInstance0Abi(handle);
    if (handleAbi) abiMask |= kEditorTraceR31HandleBit;

    IL2CPP::MethodInfo* selectInfo = select.IsValid() ? select.GetInfo() : nullptr;
    const bool selectParams = selectInfo && selectInfo->parameters_count == 2 &&
            selectInfo->parameters && selectInfo->parameters[0] && selectInfo->parameters[1];
    const bool selectAbi = selectInfo && selectInfo->methodPointer && !select._isStatic &&
            selectParams && TypeByRef(selectInfo->parameters[0]) == 0 &&
            TypeByRef(selectInfo->parameters[1]) == 0 &&
            SameClass(Class(selectInfo->parameters[0]), floor) &&
            SameClass(Class(selectInfo->parameters[1]), boolClass) &&
            selectInfo->return_type && TypeCode(selectInfo->return_type) == 1;
    if (selectAbi) abiMask |= kEditorTraceR31SelectBit;

    IL2CPP::MethodInfo* smartInfo = smart.IsValid() ? smart.GetInfo() : nullptr;
    const bool smartAbi = smartInfo && smartInfo->methodPointer && !smart._isStatic &&
            smartInfo->parameters_count == 1 && smartInfo->parameters &&
            smartInfo->parameters[0] && TypeByRef(smartInfo->parameters[0]) == 0 &&
            SameClass(Class(smartInfo->parameters[0]), boolClass) &&
            smartInfo->return_type && SameClass(Class(smartInfo->return_type), gameObject);
    if (smartAbi) abiMask |= kEditorTraceR31SmartBit;

    IL2CPP::MethodInfo* gizmoInfo = gizmoAtMouse.IsValid() ? gizmoAtMouse.GetInfo() : nullptr;
    const bool gizmoAbi = gizmoInfo && gizmoInfo->methodPointer && !gizmoAtMouse._isStatic &&
            gizmoInfo->parameters_count == 0 && gizmoInfo->return_type &&
            SameClass(Class(gizmoInfo->return_type), gizmo);
    if (gizmoAbi) abiMask |= kEditorTraceR31GizmoBit;

    if (EditorTraceR31Vector3VoidAbi(dragCamera, vector3Class))
        abiMask |= kEditorTraceR31DragCameraBit;
    if (EditorTraceR31VoidInstance0Abi(dragTilesStart))
        abiMask |= kEditorTraceR31DragTilesStartBit;
    if (EditorTraceR31Vector3VoidAbi(dragTiles, vector3Class))
        abiMask |= kEditorTraceR31DragTilesBit;

    MethodBase mousePosition = input.GetMethod("get_mousePosition", 0);
    MethodBase touchCount = input.GetMethod("get_touchCount", 0);
    MethodBase mouseHeld = input.GetMethod("GetMouseButton", 1);
    MethodBase mouseDown = input.GetMethod("GetMouseButtonDown", 1);
    MethodBase mouseUp = input.GetMethod("GetMouseButtonUp", 1);
    MethodBase screenWidth = screen.GetMethod("get_width", 0);
    MethodBase screenHeight = screen.GetMethod("get_height", 0);

    IL2CPP::MethodInfo* mousePosInfo =
            mousePosition.IsValid() ? mousePosition.GetInfo() : nullptr;
    const bool mousePosAbi = mousePosInfo && mousePosInfo->methodPointer &&
            mousePosition._isStatic && mousePosInfo->parameters_count == 0 &&
            mousePosInfo->return_type &&
            SameClass(Class(mousePosInfo->return_type), vector3Class);
    const bool inputAbi = mousePosAbi &&
            EditorTraceR31StaticInt0Abi(touchCount, intClass) &&
            EditorTraceR31StaticInt0Abi(screenWidth, intClass) &&
            EditorTraceR31StaticInt0Abi(screenHeight, intClass) &&
            EditorTraceR31StaticMouseButtonAbi(mouseHeld, intClass, boolClass) &&
            EditorTraceR31StaticMouseButtonAbi(mouseDown, intClass, boolClass) &&
            EditorTraceR31StaticMouseButtonAbi(mouseUp, intClass, boolClass);

    g_editorTraceR31AbiMask.store(abiMask);
    g_editorTraceR31InputAbiGuard.store(inputAbi ? 1 : 0);
    if ((abiMask & kEditorTraceR31HandleBit) == 0 || !inputAbi ||
        !PrepareEditorTraceR31Fuse()) return;

    if (!WriteMarker(g_editorTraceR31InstallMarker)) {
        g_editorTraceR31MarkerReady.store(0);
        g_editorTraceR31RecoveryState.store(10);
        return;
    }
    g_editorTraceR31InstallAttempted.store(1);

    g_editorTraceR31GetMousePosition = mousePosition;
    g_editorTraceR31GetTouchCount = touchCount;
    g_editorTraceR31GetScreenWidth = screenWidth;
    g_editorTraceR31GetScreenHeight = screenHeight;
    g_editorTraceR31GetMouseButton = mouseHeld;
    g_editorTraceR31GetMouseButtonDown = mouseDown;
    g_editorTraceR31GetMouseButtonUp = mouseUp;

    int installedMask = 0;
    BasicHook(handle, HookEditorTraceR31Handle, g_oldEditorTraceR31Handle);
    if (g_oldEditorTraceR31Handle) installedMask |= kEditorTraceR31HandleBit;

    if (abiMask & kEditorTraceR31SelectBit) {
        BasicHook(select, HookEditorTraceR31Select, g_oldEditorTraceR31Select);
        if (g_oldEditorTraceR31Select) installedMask |= kEditorTraceR31SelectBit;
    }
    if (abiMask & kEditorTraceR31SmartBit) {
        BasicHook(smart, HookEditorTraceR31Smart, g_oldEditorTraceR31Smart);
        if (g_oldEditorTraceR31Smart) installedMask |= kEditorTraceR31SmartBit;
    }
    if (abiMask & kEditorTraceR31GizmoBit) {
        BasicHook(gizmoAtMouse, HookEditorTraceR31Gizmo, g_oldEditorTraceR31Gizmo);
        if (g_oldEditorTraceR31Gizmo) installedMask |= kEditorTraceR31GizmoBit;
    }
    if (abiMask & kEditorTraceR31DragCameraBit) {
        BasicHook(dragCamera, HookEditorTraceR31DragCamera, g_oldEditorTraceR31DragCamera);
        if (g_oldEditorTraceR31DragCamera) installedMask |= kEditorTraceR31DragCameraBit;
    }
    if (abiMask & kEditorTraceR31DragTilesStartBit) {
        BasicHook(dragTilesStart, HookEditorTraceR31DragTilesStart,
                  g_oldEditorTraceR31DragTilesStart);
        if (g_oldEditorTraceR31DragTilesStart)
            installedMask |= kEditorTraceR31DragTilesStartBit;
    }
    if (abiMask & kEditorTraceR31DragTilesBit) {
        BasicHook(dragTiles, HookEditorTraceR31DragTiles, g_oldEditorTraceR31DragTiles);
        if (g_oldEditorTraceR31DragTiles) installedMask |= kEditorTraceR31DragTilesBit;
    }

    g_editorTraceR31InstalledMask.store(installedMask);
    if ((installedMask & kEditorTraceR31HandleBit) != 0 && installedMask == abiMask) {
        ClearMarker(g_editorTraceR31InstallMarker);
    } else {
        g_editorTraceR31RecoveryState.store(11);
    }
}

'''
once(
    "void ReconcileInstallState() {\n",
    hooks + "void ReconcileInstallState() {\n",
)

once(
    "    MaybeInstallTileR21();\n"
    "    MaybeInstallSfbExtraHooks();\n",
    "    MaybeInstallTileR21();\n"
    "    MaybeInstallEditorTraceR31();\n"
    "    MaybeInstallSfbExtraHooks();\n",
)

once(
    '        << "tileR30SecondLayerMask=" << g_tileR30SecondLayerMask.load() << \'\\n\'\n',
    '        << "tileR30SecondLayerMask=" << g_tileR30SecondLayerMask.load() << \'\\n\'\n'
    '        << "editorTraceRevision=31" << \'\\n\'\n'
    '        << "editorTraceR31Policy=bounded-read-only-pointer-transaction" << \'\\n\'\n'
    '        << "editorTraceR31Mutation=0" << \'\\n\'\n'
    '        << "editorTraceR31Slots=8" << \'\\n\'\n'
    '        << "editorTraceR31AbiMask=" << g_editorTraceR31AbiMask.load() << \'\\n\'\n'
    '        << "editorTraceR31InputAbiGuard=" << g_editorTraceR31InputAbiGuard.load() << \'\\n\'\n'
    '        << "editorTraceR31InstalledMask=" << g_editorTraceR31InstalledMask.load() << \'\\n\'\n'
    '        << "editorTraceR31InstallAttempted=" << g_editorTraceR31InstallAttempted.load() << \'\\n\'\n'
    '        << "editorTraceR31MarkerReady=" << g_editorTraceR31MarkerReady.load() << \'\\n\'\n'
    '        << "editorTraceR31RecoveryState=" << g_editorTraceR31RecoveryState.load() << \'\\n\'\n'
    '        << "editorTraceR31CallProvenMask=" << EditorTraceR31CallProvenMask() << \'\\n\'\n'
)

once(
    '    std::ostringstream out;\n'
    '    out << base\n'
    '        << "sfbOpenFiltersCanaryCalls=" << g_sfbCalls.load() << \'\\n\'\n',
    '    std::ostringstream out;\n'
    '    out << base;\n'
    '    AppendEditorTraceR31(out);\n'
    '    out << "sfbOpenFiltersCanaryCalls=" << g_sfbCalls.load() << \'\\n\'\n',
)

for marker in (
    "editorTraceRevision=31",
    "editorTraceR31Policy=bounded-read-only-pointer-transaction",
    "editorTraceR31Mutation=0",
    "editorTraceR31Slots=8",
    "editorTraceR31AbiMask=",
    "editorTraceR31InstalledMask=",
    "editorTraceR31Txn",
    "MaybeInstallEditorTraceR31();",
    'GetMethod("HandleMouseActions", 0)',
    'GetMethod("SelectFloor", 2)',
    'GetMethod("SmartObjectSelect", 1)',
    'GetMethod("GizmoAtMouse", 0)',
    'GetMethod("DragCamera", 1)',
    'GetMethod("DragTilesStart", 0)',
    'GetMethod("DragTiles", 1)',
    "editor-r31-trace-install.pending",
    "editor-r31-trace-call-",
    "EditorTraceR31RecordObjects(count);",
    "EditorTraceR31RecordRaycast(ordinal, count, origin, layerMask);",
):
    if marker not in s:
        raise SystemExit(f"r31 marker missing: {marker}")

for forbidden in (
    "origin.x =",
    "origin.y =",
    "SelectFloor(self",
):
    # Restrict mutation checks to R31 source additions; older source may contain unrelated text.
    if forbidden in hooks:
        raise SystemExit(f"r31 forbidden mutation-like marker: {forbidden}")

if s.count("BasicHook(") != 25:
    raise SystemExit(f"r31 expected 25 compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
