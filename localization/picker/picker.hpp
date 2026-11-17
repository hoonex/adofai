#pragma once
#include <jni.h>
#include <string>
#include <cstdlib>
#include "util.hpp"
namespace picker {
inline JavaVM* vm=nullptr;
inline jclass cls=nullptr;
inline jmethodID open_id=nullptr, snapshot_id=nullptr;
struct State {int code=0; std::string path,message; long long done=0,total=0;};
inline void init(JavaVM* v,JNIEnv* e){
 vm=v; jclass local=e->FindClass("dev/hoonex/adofai/bridge/ZipPicker");
 if(!local||e->ExceptionCheck()){e->ExceptionClear();LOGE("ZIP picker bridge missing");return;}
 cls=(jclass)e->NewGlobalRef(local);e->DeleteLocalRef(local);
 open_id=e->GetStaticMethodID(cls,"open","()Z");
 snapshot_id=e->GetStaticMethodID(cls,"snapshot","()[Ljava/lang/String;");
 if(e->ExceptionCheck()){e->ExceptionClear();open_id=nullptr;snapshot_id=nullptr;}
}
struct Env {
 JNIEnv* e=nullptr;bool attached=false;
 Env(){if(!vm)return;int r=vm->GetEnv((void**)&e,JNI_VERSION_1_6);if(r==JNI_EDETACHED){if(vm->AttachCurrentThread(&e,nullptr)==JNI_OK)attached=true;else e=nullptr;}else if(r!=JNI_OK)e=nullptr;}
 ~Env(){if(attached)vm->DetachCurrentThread();}
};
inline std::string str(JNIEnv* e,jobject o){
 if(!o)return ""; auto s=(jstring)o;const char* p=e->GetStringUTFChars(s,nullptr);std::string result=p?p:"";if(p)e->ReleaseStringUTFChars(s,p);e->DeleteLocalRef(o);return result;
}
inline bool open(){
 Env a;if(!a.e||!cls||!open_id)return false;auto r=a.e->CallStaticBooleanMethod(cls,open_id);if(a.e->ExceptionCheck()){a.e->ExceptionClear();return false;}return r==JNI_TRUE;
}
inline State snapshot(){
 State r;Env a;if(!a.e||!cls||!snapshot_id){r.code=4;return r;}auto e=a.e;
 auto arr=(jobjectArray)e->CallStaticObjectMethod(cls,snapshot_id);
 if(e->ExceptionCheck()){e->ExceptionClear();r.code=4;return r;}
 if(!arr||e->GetArrayLength(arr)!=5){if(arr)e->DeleteLocalRef(arr);r.code=4;return r;}
 r.code=atoi(str(e,e->GetObjectArrayElement(arr,0)).c_str());
 r.path=str(e,e->GetObjectArrayElement(arr,1));r.message=str(e,e->GetObjectArrayElement(arr,2));
 r.done=atoll(str(e,e->GetObjectArrayElement(arr,3)).c_str());r.total=atoll(str(e,e->GetObjectArrayElement(arr,4)).c_str());e->DeleteLocalRef(arr);return r;
}
}
