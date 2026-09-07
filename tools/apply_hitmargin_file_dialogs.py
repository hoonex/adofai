#!/usr/bin/env python3
"""Complete the HitMargin mobile file-dialog bridge on the pinned upstream source.

The upstream Java facade already implements open/save/folder operations, while the
native hook only wires OpenFilePanel. This transform verifies the original
FilePicker.cpp Git blob before changing it, then connects all three synchronous SFB
dialogs and makes cancellation return the shapes expected by the C# API.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
FILE = "app/src/main/jni/FilePicker.cpp"
EXPECTED_BLOB = "38925e7e3428aaf0fef4a525377f057d38afb912"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one replacement anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def verify_identity(source: Path) -> None:
    head = git(source, "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"upstream HEAD mismatch: expected {EXPECTED_HEAD}, got {head}")
    blob = git(source, "hash-object", FILE)
    if blob != EXPECTED_BLOB:
        raise SystemExit(f"upstream blob mismatch for {FILE}: expected {EXPECTED_BLOB}, got {blob}")


def apply(source: Path) -> None:
    verify_identity(source)
    path = source / FILE

    replace_once(
        path,
        '''static jclass  g_selectorClass = nullptr;
static jmethodID g_method_selectFile = nullptr;
static jfieldID  g_field_isDone = nullptr;
static jmethodID g_method_getFilePath = nullptr;

static std::mutex g_pickerMutex;
static std::condition_variable g_pickerCV;
static bool g_pickerResultReady = false;
static std::string g_pickerSelectedPath;
''',
        '''static jclass  g_selectorClass = nullptr;
static jmethodID g_method_selectFile = nullptr;
static jmethodID g_method_saveAs = nullptr;
static jmethodID g_method_selectFolder = nullptr;
static jfieldID  g_field_isDone = nullptr;
static jmethodID g_method_getFilePath = nullptr;
static jmethodID g_method_getFolderPath = nullptr;

enum class PickerMode {
    OpenFile,
    SaveFile,
    OpenFolder,
};

static std::mutex g_pickerCallMutex;
static std::mutex g_pickerMutex;
static std::condition_variable g_pickerCV;
static bool g_pickerResultReady = false;
static std::string g_pickerSelectedPath;

static void FinishPickerResult(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(g_pickerMutex);
        g_pickerSelectedPath = path;
        g_pickerResultReady = true;
    }
    g_pickerCV.notify_one();
}
''',
        "FilePicker bridge declarations",
    )

    replace_once(
        path,
        '''// 轮询等待 Selector 结果（在独立线程运行，避免阻塞 UI 线程）
static void WaitForJavaPickerResult() {
    JNIEnv* env = nullptr;
    bool attached = false;
    jint res = g_vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (res == JNI_EDETACHED) {
        g_vm->AttachCurrentThread(&env, nullptr);
        attached = true;
    } else if (res != JNI_OK || !env) {
        return;
    }

    {
        jstring typeStr = env->NewStringUTF("adofai,zip");
        env->CallStaticVoidMethod(g_selectorClass, g_method_selectFile, typeStr);
        env->DeleteLocalRef(typeStr);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }

    constexpr int kMaxTries = 2000;
    int tries = 0;
    bool done = false;
    while (!done && tries < kMaxTries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        tries++;
        if (g_selectorClass && g_field_isDone) {
            jboolean isDone = env->GetStaticBooleanField(g_selectorClass, g_field_isDone);
            if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
            done = (isDone == JNI_TRUE);
        } else {
            break;
        }
    }

    if (done && g_method_getFilePath) {
        jstring jpath = (jstring)env->CallStaticObjectMethod(g_selectorClass, g_method_getFilePath);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        } else if (jpath) {
            const char* cpath = env->GetStringUTFChars(jpath, nullptr);
            if (cpath) {
                {
                    std::lock_guard<std::mutex> lock(g_pickerMutex);
                    g_pickerSelectedPath = cpath;
                    g_pickerResultReady = true;
                }
                g_pickerCV.notify_one();
                env->ReleaseStringUTFChars(jpath, cpath);
            }
            env->DeleteLocalRef(jpath);
        }
    } else if (!done) {
        {
            std::lock_guard<std::mutex> lock(g_pickerMutex);
            g_pickerSelectedPath.clear();
            g_pickerResultReady = true;
        }
        g_pickerCV.notify_one();
    }

    if (attached) g_vm->DetachCurrentThread();
}
''',
        '''// 轮询等待 Selector 结果（在独立线程运行，避免阻塞 Unity 调用线程）
static void WaitForJavaPickerResult(PickerMode mode, std::string argument) {
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!g_vm) {
        FinishPickerResult("");
        return;
    }

    jint res = g_vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (res == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
            FinishPickerResult("");
            return;
        }
        attached = true;
    } else if (res != JNI_OK || !env) {
        FinishPickerResult("");
        return;
    }

    if (mode == PickerMode::OpenFile) {
        jstring typeStr = env->NewStringUTF(argument.c_str());
        env->CallStaticVoidMethod(g_selectorClass, g_method_selectFile, typeStr);
        env->DeleteLocalRef(typeStr);
    } else if (mode == PickerMode::SaveFile) {
        jstring nameStr = env->NewStringUTF(argument.c_str());
        env->CallStaticVoidMethod(g_selectorClass, g_method_saveAs, nameStr);
        env->DeleteLocalRef(nameStr);
    } else {
        env->CallStaticVoidMethod(g_selectorClass, g_method_selectFolder);
    }

    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        LOGE("FileSelector invocation failed");
        FinishPickerResult("");
        if (attached) g_vm->DetachCurrentThread();
        return;
    }

    constexpr int kMaxTries = 2000;
    int tries = 0;
    bool done = false;
    while (!done && tries < kMaxTries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        tries++;
        if (g_selectorClass && g_field_isDone) {
            jboolean isDone = env->GetStaticBooleanField(g_selectorClass, g_field_isDone);
            if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
            done = (isDone == JNI_TRUE);
        } else {
            break;
        }
    }

    std::string selectedPath;
    jmethodID getter = mode == PickerMode::OpenFolder ? g_method_getFolderPath : g_method_getFilePath;
    if (done && getter) {
        jstring jpath = (jstring)env->CallStaticObjectMethod(g_selectorClass, getter);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        } else if (jpath) {
            const char* cpath = env->GetStringUTFChars(jpath, nullptr);
            if (cpath) {
                selectedPath = cpath;
                env->ReleaseStringUTFChars(jpath, cpath);
            }
            env->DeleteLocalRef(jpath);
        }
    }

    FinishPickerResult(selectedPath);
    if (!done) LOGW("FileSelector timed out or failed before returning a result");
    if (attached) g_vm->DetachCurrentThread();
}
''',
        "FilePicker worker",
    )

    replace_once(
        path,
        '''// ---- OpenFilePanel 的替换实现 ----
Array<String*>* Hooked_OpenFilePanel(String* title, String* directory, String* extension, bool multiselect) {
    if (!g_selectorClass || !g_method_selectFile) {
        LOGE("FileSelector bridge not initialized");
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_pickerMutex);
        g_pickerResultReady = false;
        g_pickerSelectedPath.clear();
    }

    std::thread waiter(WaitForJavaPickerResult);
    waiter.detach();

    std::unique_lock<std::mutex> lock(g_pickerMutex);
    g_pickerCV.wait(lock, []{ return g_pickerResultReady; });
    std::string filePath = g_pickerSelectedPath;
    lock.unlock();

    if (filePath.empty()) {
        LOGW("File picker cancelled or no file selected");
        return nullptr;
    }
    LOGD("Selected level path: %s", filePath.c_str());

    auto array = g_stringClass.NewArray<String*>(1);
    if (!array) {
        LOGE("Failed to create string array");
        return nullptr;
    }
    array->m_Items[0] = CreateMonoString(filePath);
    LOGD("Returning file path to Unity: %s", array->m_Items[0]->str().c_str());
    return array;
}
''',
        '''static std::string RunJavaPicker(PickerMode mode, const std::string& argument) {
    std::lock_guard<std::mutex> callLock(g_pickerCallMutex);
    {
        std::lock_guard<std::mutex> lock(g_pickerMutex);
        g_pickerResultReady = false;
        g_pickerSelectedPath.clear();
    }

    std::thread waiter(WaitForJavaPickerResult, mode, argument);
    waiter.detach();

    std::unique_lock<std::mutex> lock(g_pickerMutex);
    g_pickerCV.wait(lock, []{ return g_pickerResultReady; });
    return g_pickerSelectedPath;
}

static Array<String*>* PathToStringArray(const std::string& path) {
    auto array = g_stringClass.NewArray<String*>(path.empty() ? 0 : 1);
    if (!array) {
        LOGE("Failed to create string array");
        return nullptr;
    }
    if (!path.empty()) array->m_Items[0] = CreateMonoString(path);
    return array;
}

// ---- OpenFilePanel 的替换实现 ----
Array<String*>* Hooked_OpenFilePanel(String* title, String* directory, String* extension, bool multiselect) {
    if (!g_selectorClass || !g_method_selectFile) {
        LOGE("FileSelector open bridge not initialized");
        return PathToStringArray("");
    }

    std::string filter = extension && !extension->str().empty() ? extension->str() : "adofai,zip";
    std::string filePath = RunJavaPicker(PickerMode::OpenFile, filter);
    if (filePath.empty()) LOGW("File picker cancelled or no file selected");
    else LOGD("Selected file path: %s", filePath.c_str());
    return PathToStringArray(filePath);
}

String* Hooked_SaveFilePanel(String* title, String* directory, String* defaultName, String* extension) {
    if (!g_selectorClass || !g_method_saveAs) {
        LOGE("FileSelector save bridge not initialized");
        return CreateMonoString("");
    }

    std::string name = defaultName ? defaultName->str() : "level";
    std::string ext = extension ? extension->str() : "";
    while (!ext.empty() && ext.front() == '.') ext.erase(ext.begin());
    if (!ext.empty()) {
        std::string suffix = "." + ext;
        if (name.size() < suffix.size() || name.substr(name.size() - suffix.size()) != suffix) {
            name += suffix;
        }
    }

    std::string filePath = RunJavaPicker(PickerMode::SaveFile, name);
    if (filePath.empty()) LOGW("Save picker cancelled");
    else LOGD("Selected save path: %s", filePath.c_str());
    return CreateMonoString(filePath);
}

Array<String*>* Hooked_OpenFolderPanel(String* title, String* directory, bool multiselect) {
    if (!g_selectorClass || !g_method_selectFolder || !g_method_getFolderPath) {
        LOGE("FileSelector folder bridge not initialized");
        return PathToStringArray("");
    }

    std::string folderPath = RunJavaPicker(PickerMode::OpenFolder, "");
    if (folderPath.empty()) LOGW("Folder picker cancelled");
    else LOGD("Selected folder path: %s", folderPath.c_str());
    return PathToStringArray(folderPath);
}
''',
        "FilePicker synchronous SFB hooks",
    )

    replace_once(
        path,
        '''    g_method_selectFile     = env->GetStaticMethodID(c, "selectFile", "(Ljava/lang/String;)V");
    g_field_isDone          = env->GetStaticFieldID(c, "isDone", "Z");
    g_method_getFilePath    = env->GetStaticMethodID(c, "getFilePath", "()Ljava/lang/String;");
    if (!g_method_selectFile || !g_field_isDone || !g_method_getFilePath) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(c);
        LOGE("Failed to get FileSelector method/field IDs");
        return;
    }
''',
        '''    g_method_selectFile     = env->GetStaticMethodID(c, "selectFile", "(Ljava/lang/String;)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_method_saveAs         = env->GetStaticMethodID(c, "saveAs", "(Ljava/lang/String;)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_method_selectFolder   = env->GetStaticMethodID(c, "selectFolder", "()V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_field_isDone          = env->GetStaticFieldID(c, "isDone", "Z");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_method_getFilePath    = env->GetStaticMethodID(c, "getFilePath", "()Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_method_getFolderPath  = env->GetStaticMethodID(c, "getFolderPath", "()Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (!g_method_selectFile || !g_field_isDone || !g_method_getFilePath) {
        env->DeleteLocalRef(c);
        LOGE("Failed to get required FileSelector open method/field IDs");
        return;
    }
    if (!g_method_saveAs) LOGW("FileSelector.saveAs unavailable; save dialog hook will stay disabled");
    if (!g_method_selectFolder || !g_method_getFolderPath)
        LOGW("FileSelector folder methods unavailable; folder dialog hook will stay disabled");
''',
        "FilePicker JNI method cache",
    )

    replace_once(
        path,
        '''void InstallFilePickerHook() {
    auto browserClass = Class("SFB", "StandaloneFileBrowser");
    auto openFilePanel = browserClass.GetMethod("OpenFilePanel", {"title","directory","extension","multiselect"});
    if (!openFilePanel.IsValid()) {
        LOGE("Failed to find StandaloneFileBrowser.OpenFilePanel");
        return;
    }
    BasicHook(openFilePanel, Hooked_OpenFilePanel, (void*)nullptr);
    LOGD("OpenFilePanel hook installed");
}
''',
        '''void InstallFilePickerHook() {
    auto browserClass = Class("SFB", "StandaloneFileBrowser");

    auto openFilePanel = browserClass.GetMethod("OpenFilePanel", {"title","directory","extension","multiselect"});
    if (openFilePanel.IsValid()) {
        BasicHook(openFilePanel, Hooked_OpenFilePanel, (void*)nullptr);
        LOGD("OpenFilePanel hook installed");
    } else {
        LOGE("Failed to find StandaloneFileBrowser.OpenFilePanel");
    }

    auto saveFilePanel = browserClass.GetMethod("SaveFilePanel", {"title","directory","defaultName","extension"});
    if (saveFilePanel.IsValid() && g_method_saveAs) {
        BasicHook(saveFilePanel, Hooked_SaveFilePanel, (void*)nullptr);
        LOGD("SaveFilePanel hook installed");
    } else {
        LOGW("StandaloneFileBrowser.SaveFilePanel hook unavailable");
    }

    auto openFolderPanel = browserClass.GetMethod("OpenFolderPanel", {"title","directory","multiselect"});
    if (openFolderPanel.IsValid() && g_method_selectFolder && g_method_getFolderPath) {
        BasicHook(openFolderPanel, Hooked_OpenFolderPanel, (void*)nullptr);
        LOGD("OpenFolderPanel hook installed");
    } else {
        LOGW("StandaloneFileBrowser.OpenFolderPanel hook unavailable");
    }
}
''',
        "FilePicker SFB installation",
    )

    print(f"Applied file-dialog transform to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
