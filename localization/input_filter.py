"""Preserve non-overlay touch input; does not modify any game/DLC entitlement."""
from pathlib import Path
import sys

p=Path(sys.argv[1])/'loader/src/input.cpp'
s=p.read_text(encoding='utf-8')
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)
replace('static int32_t g_btn_finger = -1;', 'static std::atomic<int32_t> g_btn_finger{-1};')
replace('// ---------------------------------------------------------------- hooks\n', '''// Unity Touch is a blittable value type. Preserve complete original Touch
// payloads and translate the game's filtered indices back to Unity's indices.
static int consumed_index(int count, int consumed) {
    if (consumed < 0 || !orig_gettouch) return -1;
    for (int i = 0; i < count; ++i) {
        alignas(8) uint8_t buf[TOUCH_SIZE + 8]{};
        call_get_touch_orig((void*)orig_gettouch, buf, i);
        int32_t id;
        memcpy(&id, buf, sizeof(id));
        if (id == consumed) return i;
    }
    return -1;
}
static void canceled_touch(void* retbuf) {
    memset(retbuf, 0, TOUCH_SIZE);
    const int32_t phase = 4;
    memcpy((uint8_t*)retbuf + TOUCH_PHASE, &phase, sizeof(phase));
}
// ---------------------------------------------------------------- hooks
''')
replace('if (!g_btn_moved) g_button_tapped.store(true);','if (!g_btn_moved) { if (t.phase == 3) g_button_tapped.store(true); }')
replace('''            bool present = false;
            for (int i = 0; i < n; i++)
                if (g_touches[i].finger == g_consumed_finger)
                    present = true;
            if (!present) g_consumed_finger = -1;
            else return 0;''','''            if (consumed_index(real, g_consumed_finger) < 0)
                g_consumed_finger = -1;
            else return real - 1;''')
replace('''    call_get_touch_orig((void*)orig_gettouch, retbuf, index);
    bool ui_open = g_ui_open.load();
    bool consumed = false;
    {
        std::lock_guard<std::mutex> lk(g_consumed_mutex);
        consumed = g_consumed_finger >= 0;
    }
    if (ui_open || consumed) {
        memset(retbuf, 0, TOUCH_SIZE);
        *(int32_t*)((uint8_t*)retbuf + TOUCH_PHASE) = 4; // Canceled
    }''','''    if (g_ui_open.load() || !orig_touchcount || !orig_gettouch) {
        canceled_touch(retbuf);
        return;
    }
    const int count = orig_touchcount();
    int consumed;
    {
        std::lock_guard<std::mutex> lk(g_consumed_mutex);
        consumed = g_consumed_finger;
    }
    const int removed = consumed_index(count, consumed);
    const int visible = count - (removed >= 0 ? 1 : 0);
    if (index < 0 || index >= visible) {
        canceled_touch(retbuf);
        return;
    }
    const int raw_index = index + (removed >= 0 && index >= removed ? 1 : 0);
    call_get_touch_orig((void*)orig_gettouch, retbuf, raw_index);''')
replace('''    if (g_ui_open.load() && r) {
        // il2cpp array: max_length at +0x18 (il2cpp_array_size_t) -> present an empty array
        *(uint64_t*)((uint8_t*)r + 0x18) = 0;
    }''','''    if (r) {
        // Unity's Input.touches returns a newly allocated value-type array.
        // Match GetTouch/TouchCount by removing ONLY the overlay's finger.
        auto* bytes = (uint8_t*)r;
        uint64_t count = 0;
        memcpy(&count, bytes + 0x18, sizeof(count));
        if (g_ui_open.load()) {
            const uint64_t zero = 0;
            memcpy(bytes + 0x18, &zero, sizeof(zero));
        } else if (count <= 1024) {
            int consumed;
            {
                std::lock_guard<std::mutex> lk(g_consumed_mutex);
                consumed = g_consumed_finger;
            }
            if (consumed >= 0) {
                uint8_t* values = bytes + 0x20;
                for (uint64_t i = 0; i < count; ++i) {
                    int32_t id;
                    memcpy(&id, values + i * TOUCH_SIZE, sizeof(id));
                    if (id != consumed) continue;
                    memmove(values + i * TOUCH_SIZE, values + (i + 1) * TOUCH_SIZE,
                            (count - i - 1) * TOUCH_SIZE);
                    --count;
                    memcpy(bytes + 0x18, &count, sizeof(count));
                    break;
                }
            }
        }
    }''')
p.write_text(s,encoding='utf-8')
print('Custom map multi-touch filter applied')
