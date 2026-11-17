from pathlib import Path
import sys
root=Path(sys.argv[1])/'loader/src'
def edit(file,old,new):
 p=root/file
 s=p.read_text(encoding='utf-8')
 assert s.count(old)==1,(file,old,s.count(old))
 p.write_text(s.replace(old,new),encoding='utf-8')
edit('game.hpp','#include <stdbool.h>','#include <stdbool.h>\n#include <string>\n\n// Read-only runtime file diagnostics; does not alter DLC entitlement.\nvoid game_queue_dlc_check();\nstd::string game_dlc_check_report();')
edit('game.cpp','#include <time.h>','#include <time.h>\n#include <atomic>\n#include <mutex>\n#include <string>\n#include <sys/stat.h>\n#include <stdio.h>')
edit('game.cpp','void game_apply_queued_options() {',r'''// Execute managed path getters only in the existing Unity main-thread queue.
static std::atomic<bool> dlc_check_pending{false};
static std::mutex dlc_report_lock;
static std::string dlc_report;
void game_queue_dlc_check() {
    {std::lock_guard<std::mutex> guard(dlc_report_lock); dlc_report="Checking...";}
    dlc_check_pending.store(true);
}
std::string game_dlc_check_report() {
    std::lock_guard<std::mutex> guard(dlc_report_lock);
    return dlc_report;
}
static std::string dlc_path(const char* getter) {
    if (!cls_gcs || !il2cpp.runtime_invoke) return {};
    // GCNS is a different class than GCS.
    auto cls=il2cpp_find_class("Assembly-CSharp","","GCNS");
    if(!cls)return {};
    auto method=il2cpp.class_get_method_from_name(cls,getter,0);
    if(!method)return {};
    Il2CppException* exception=nullptr;
    auto result=il2cpp.runtime_invoke(method,nullptr,nullptr,&exception);
    if(exception||!result)return {};
    auto str=reinterpret_cast<Il2CppString*>(result);
    if(str->length<=0||str->length>8192)return {};
    // Game data paths are normally ASCII; encode non-ASCII safely as UTF-8.
    std::string out;
    for(int i=0;i<str->length;i++){
        uint32_t cp=str->chars[i];
        if(cp>=0xD800&&cp<=0xDBFF&&i+1<str->length){
            uint32_t lo=str->chars[i+1];
            if(lo>=0xDC00&&lo<=0xDFFF){cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00);i++;}
        }
        if(cp==0||(cp>=0xD800&&cp<=0xDFFF))return {};
        if(cp<0x80)out+=(char)cp;
        else if(cp<0x800){out+=(char)(0xC0|(cp>>6));out+=(char)(0x80|(cp&63));}
        else if(cp<0x10000){out+=(char)(0xE0|(cp>>12));out+=(char)(0x80|((cp>>6)&63));out+=(char)(0x80|(cp&63));}
        else {out+=(char)(0xF0|(cp>>18));out+=(char)(0x80|((cp>>12)&63));out+=(char)(0x80|((cp>>6)&63));out+=(char)(0x80|(cp&63));}
    }
    if(out.compare(0,8,"file:///")==0)out.erase(0,7);
    if(out.empty()||out[0]!='/')return {};
    return out;
}
static std::string dlc_file_status(const std::string& path) {
    if(path.empty())return "PATH UNAVAILABLE";
    struct stat st{};
    if(stat(path.c_str(),&st)!=0||!S_ISREG(st.st_mode))return "NOT FOUND";
    FILE* f=fopen(path.c_str(),"rb");
    if(!f)return "UNREADABLE";
    char magic[8]{};size_t n=fread(magic,1,sizeof(magic),f);fclose(f);
    if(n<8||memcmp(magic,"UnityFS",7)!=0||magic[7]!=0)return "INVALID HEADER";
    return "UnityFS present ("+std::to_string((long long)st.st_size)+" bytes)";
}
static void dlc_check_main_thread() {
    if(!dlc_check_pending.exchange(false))return;
    std::string report;
    if(!game_init())report="Game runtime not ready";
    else {
       auto base=dlc_path("get_BundlesLoadPath");
       auto assets=dlc_path("get_neoCosmosBundleAssetsPath");
       auto scenes=dlc_path("get_neoCosmosBundleScenesPath");
       if(!base.empty()&&base.back()!='/')base+='/';
       if(assets.empty()&&!base.empty())assets=base+"neocosmos_assets_all.bundle";
       if(scenes.empty()&&!base.empty())scenes=base+"neocosmos_scenes_all.bundle";
       report="Assets: "+dlc_file_status(assets)+"\nScenes: "+dlc_file_status(scenes);
       if(!base.empty())report+="\nDirectory: "+base;
       report+="\nFile presence only; purchase and runtime compatibility not verified.";
    }
    {std::lock_guard<std::mutex> guard(dlc_report_lock);dlc_report=report;}
    LOGI("DLC check: %s",report.c_str());
}
void game_apply_queued_options() {
    dlc_check_main_thread();''')
edit('overlay.cpp','static void draw_game_page() {',r'''static void draw_game_page() {
    ImGui::TextUnformatted("Neo Cosmos");
    const char* check=g_lang==LANG_KO?"DLC 파일 확인":g_lang==LANG_ZH?"检查 DLC 文件":"Check DLC files";
    if (ImGui::Button(check,ImVec2(-1,0))) game_queue_dlc_check();
    std::string report=game_dlc_check_report();
    if (!report.empty()) ImGui::TextWrapped("%s",report.c_str());
    ImGui::TextWrapped("%s",g_lang==LANG_KO?"파일 존재 여부만 확인합니다. 구매 인증을 변경하지 않습니다.":g_lang==LANG_ZH?"仅检查文件，不修改购买验证。":"Read-only file check; purchase verification unchanged.");
    ImGui::Separator();
    ImGui::Spacing();''')
print('DLC diagnostic source checks passed')
