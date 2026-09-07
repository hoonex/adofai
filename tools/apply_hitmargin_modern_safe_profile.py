#!/usr/bin/env python3
"""Keep ADOFAI 3.3+ away from legacy hook ABIs that changed after 2.10.

The pinned HitMargin source targets 2.8.3-2.10.1. ADOFAI 3.3.1 still exposes many
of the same IL2CPP method names, but several signatures/classes changed (for
example scnGame.Play, scrPlanet.GetMultipressPenalty and LevelEventInfo accessors).
Name-only lookup is therefore not enough to make those hooks safe.

This transform runs after the identity-checked legacy transforms. When the current
runtime structurally exposes UnityFileDialog.FileBrowser.PickFile (the file-dialog
backend used by ADOFAI 3.3.1), start() installs only the Android file-picker bridge
and returns before any legacy gameplay/editor hook is installed. Older supported
runtimes keep the existing behavior unchanged.

This is deliberately conservative: it trades optional legacy mod features for a
bounded crash surface while the new mobile editor shell is built on the current
runtime.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
FILE = "app/src/main/jni/Main.cpp"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one replacement anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def verify_post_transforms(source: Path) -> None:
    head = git(source, "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"upstream HEAD mismatch: expected {EXPECTED_HEAD}, got {head}")

    text = (source / FILE).read_text(encoding="utf-8")
    required = (
        "enableEditorDesktopMode",
        "InitJavaFilePicker(env);",
        "InstallFilePickerHook();",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise SystemExit(
            "modern safe profile must run after the existing editor/file transforms; "
            f"missing markers: {missing}"
        )


def apply(source: Path) -> None:
    verify_post_transforms(source)
    path = source / FILE

    replace_once(
        path,
        '''JavaVM* g_vm = nullptr;

// ============ start() 函数 ============
''',
        '''JavaVM* g_vm = nullptr;

static bool IsModernUnityFileDialogRuntime() {
    auto pickFile = Class("UnityFileDialog", "FileBrowser").GetMethod(
        "PickFile", {"directory", "filterName", "filterExtensions", "title"});
    return pickFile.IsValid();
}

static void InstallAndroidFilePickerBridge() {
    JNIEnv* env = nullptr;
    bool attached = false;
    if (!g_vm) {
        LOGE("Android file picker bridge unavailable: JavaVM is null");
        return;
    }

    jint result = g_vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (result == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
            LOGE("Android file picker bridge could not attach current thread");
            return;
        }
        attached = true;
    } else if (result != JNI_OK || !env) {
        LOGE("Android file picker bridge could not obtain JNIEnv");
        return;
    }

    InitJavaFilePicker(env);
    InstallFilePickerHook();
    if (attached) g_vm->DetachCurrentThread();
}

// ============ start() 函数 ============
''',
        "Main.cpp modern runtime helpers",
    )

    replace_once(
        path,
        '''    // 初始化所有缓存
    InitModCache();

    // ---- 安装 Hook（根据配置条件安装） ----
''',
        '''    // 初始化所有缓存
    InitModCache();

    // ADOFAI 3.3.1 uses UnityFileDialog and retained many legacy method names while
    // changing their ABI. Do not install the 2.8-2.10 gameplay/editor hooks merely
    // because those names still exist; several would corrupt registers/return state.
    if (IsModernUnityFileDialogRuntime()) {
        LOGD("ADOFAI modern runtime profile detected (UnityFileDialog)");
        LOGD("Legacy 2.x gameplay/editor hooks intentionally disabled for ABI safety");
        InstallAndroidFilePickerBridge();
        LOGD("Modern runtime platform bridge installed");
        return;
    }

    // ---- 安装 Hook（根据配置条件安装） ----
''',
        "Main.cpp modern runtime early profile",
    )

    legacy_picker_block = '''    // 文件选择器
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        g_vm->AttachCurrentThread(&env, nullptr);
        attached = true;
    }
    if (env) {
        InitJavaFilePicker(env);
        InstallFilePickerHook();
    }
    if (attached) g_vm->DetachCurrentThread();

    LOGD("All hooks installed.");
'''
    replace_once(
        path,
        legacy_picker_block,
        '''    // 文件选择器
    InstallAndroidFilePickerBridge();

    LOGD("All hooks installed.");
''',
        "Main.cpp shared picker bridge",
    )

    print(f"Applied modern safe runtime profile to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
