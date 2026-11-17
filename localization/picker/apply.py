from pathlib import Path
import sys
root=Path(sys.argv[1])
base=Path(__file__).parent
src=root/'loader/src'
def edit(file,old,new):
 p=src/file
 s=p.read_text(encoding='utf-8')
 assert s.count(old)==1,(file,old,s.count(old))
 p.write_text(s.replace(old,new),encoding='utf-8')
(src/'picker.hpp').write_bytes((base/'picker.hpp').read_bytes())
edit('main.cpp','#include <jni.h>','#include <jni.h>\n#include "picker.hpp"')
edit('main.cpp','(void)vm; (void)reserved;','(void)reserved; JNIEnv* env=nullptr; if(vm->GetEnv((void**)&env,JNI_VERSION_1_6)==JNI_OK) picker::init(vm,env);')
edit('overlay.cpp','#include "zipimport.hpp"','#include "zipimport.hpp"\n#include "picker.hpp"')
edit('overlay.cpp','static void draw_file_browser() {','''static bool g_picker_pending=false;
static picker::State g_picker_state;
static double g_picker_poll_at=0;
static const char* picker_text(const char* ko,const char* zh,const char* en) {
 return g_lang==LANG_KO?ko:g_lang==LANG_ZH?zh:en;
}
static void poll_picker() {
 if(!g_picker_pending)return;
 const double now=ImGui::GetTime();
 if(now<g_picker_poll_at)return;
 g_picker_poll_at=now+0.2;
 g_picker_state=picker::snapshot();
 if(g_picker_state.code==3) {
    g_picker_pending=false;
    if(!g_picker_state.path.empty())start_zip_import(g_picker_state.path);
 } else if(g_picker_state.code==4) {
    g_picker_pending=false;
    set_status(2,"","");snprintf(g_status,sizeof(g_status),"%s",g_picker_state.message.empty()?picker_text("파일 선택기 오류","文件选择器错误","File picker error"):g_picker_state.message.c_str());
 } else if(g_picker_state.code==5) {
    g_picker_pending=false;
    set_status(1,"","");
 }
}
static void draw_file_browser() {
    if(g_picker_pending) {
      if(g_picker_state.code==1)ImGui::TextWrapped("%s",picker_text("시스템 파일 선택기에서 ZIP을 선택하세요.","请选择 ZIP 文件","Select a ZIP file in the system picker."));
      else {
       ImGui::TextWrapped("%s",picker_text("ZIP 파일 복사 중","正在复制 ZIP","Copying ZIP file"));
       char info[100];snprintf(info,sizeof(info),"%.1f MB / %.1f MB",g_picker_state.done/1048576.0,g_picker_state.total/1048576.0);
       if(g_picker_state.total>0)ImGui::ProgressBar((float)((double)g_picker_state.done/g_picker_state.total),ImVec2(-1,0),info);
       else {snprintf(info,sizeof(info),"%.1f MB",g_picker_state.done/1048576.0);ImGui::TextUnformatted(info);}
      }
    } else if(zip_import_state()!=ZIPIMP_WORKING) {
      if(ImGui::Button(picker_text("내 파일에서 ZIP 열기","从系统文件选择 ZIP","Open ZIP from Android files"),ImVec2(-1,0))){
       if(picker::open()){g_picker_pending=true;g_picker_poll_at=0;g_picker_state.code=1;}
       else {set_status(2,"","");snprintf(g_status,sizeof(g_status),"%s",picker_text("파일 선택기를 열지 못했습니다.","无法打开文件选择器","Could not open the file picker"));}
      }
      ImGui::TextWrapped("%s",picker_text("다운로드 / 내 파일 / 클라우드에 있는 ZIP을 선택하세요.","选择下载文件夹或云端的 ZIP","Choose a ZIP from Downloads, Files or cloud storage."));
    }
''')
edit('overlay.cpp','    draw_overlay((float)fb_w, (float)fb_h);','    poll_picker();\n    draw_overlay((float)fb_w, (float)fb_h);')
