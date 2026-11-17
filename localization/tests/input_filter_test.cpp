// Host simulation of the actual patched native input.cpp.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef _MSC_VER
#define __attribute__(x)
#endif
#define INPUT_TEST
#include "input.cpp"
uint64_t g_il2cpp_base = 0;
void* hook_install(void*, void*) { return nullptr; }
extern "C" void hk_GetTouch_asm() {}
extern "C" void log_line(int, const char*, ...) {}
bool game_ready(){return true;}
bool game_init(){return true;}
bool game_load_level(const char*){return true;}
void game_apply_queued_options(){}
bool game_pause_for_overlay(){return true;}
void game_resume_overlay_pause(){}
static std::vector<TouchEvent> fake;
static int raw_count(){return (int)fake.size();}
static void raw_touch(int32_t i, void* b){
    assert(i>=0 && i<(int)fake.size());
    auto t=fake[i];
    memset(b,0,TOUCH_SIZE);
    memcpy(b,&t.finger,4);
    memcpy((uint8_t*)b+4,&t.x,4);
    memcpy((uint8_t*)b+8,&t.y,4);
    memcpy((uint8_t*)b+TOUCH_PHASE,&t.phase,4);
}
extern "C" void call_get_touch_orig(void* fn,void* b,int i){
   ((void(*)(int32_t,void*))fn)(i,b);
}
alignas(8) static uint8_t arr[32+68*128];
static void* raw_touches(){
    memset(arr,0,sizeof(arr));
    uint64_t n=fake.size();
    memcpy(arr+24,&n,8);
    for(int i=0;i<raw_count();i++)raw_touch(i,arr+32+68*i);
    return arr;
}
static int finger(const void* b){int32_t n;memcpy(&n,b,4);return n;}
static int phase(const void* b){int32_t n;memcpy(&n,(const uint8_t*)b+36,4);return n;}
static void check(std::initializer_list<int> expected){
    assert(hk_get_touchCount()==(int)expected.size());
    int i=0;uint8_t data[68];
    for(int id:expected){hk_GetTouch_c(i++,data);assert(finger(data)==id);}
    hk_GetTouch_c(i,data);assert(phase(data)==4);
    auto* a=(uint8_t*)hk_get_touches();
    uint64_t n;memcpy(&n,a+24,8);
    assert(n==expected.size());
    i=0;for(int id:expected){assert(finger(a+32+68*i++)==id);}
}
int main(){
    input_test_set_orig((void*)raw_count,(void*)raw_touch,nullptr,nullptr,nullptr,nullptr,(void*)raw_touches);
    input_set_button_rect(10,10,40,40,300,200); // Unity y=160..190
    fake={{200,100,2,10},{20,170,0,11},{250,100,0,12}};
    check({10,12}); // only overlay touch removed; hold and other tap survive
    fake={{20,170,1,11},{200,100,2,10},{250,100,2,12}};
    check({10,12}); // reordered raw indices
    fake={{200,100,2,10},{20,170,3,11},{250,100,3,12}};
    check({10,12});assert(input_button_tapped());
    fake={{200,100,2,10}};check({10});
    fake={{20,170,0,14},{200,100,2,10}};check({10});
    fake={{20,170,4,14},{200,100,2,10}};check({10});assert(!input_button_tapped());
    fake={{200,100,2,10}};check({10});
    input_set_ui_open(true);check({});
    input_set_ui_open(false);check({10});
    fake.clear();check({});
    puts("PASS: touch count/index/array agree; hold, reorder, release, cancel, overlay states");
}
