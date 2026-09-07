#!/usr/bin/env python3
"""Wire the repository-owned Android editor shell into the pinned hook build.

This transform intentionally runs after all HitMargin compatibility transforms. It
copies only our native bridge into the prepared source tree, adds the source to the
NDK module, teaches the custom DEX loader to launch MobileEditorShell, and enables
preview only inside the ABI-safe modern runtime profile.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args,], text=True).strip()


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

    main = (source / "app/src/main/jni/Main.cpp").read_text(encoding="utf-8")
    picker = (source / "app/src/main/jni/FilePicker.cpp").read_text(encoding="utf-8")
    required_main = (
        "IsModernUnityFileDialogRuntime",
        "Legacy 2.x gameplay/editor hooks intentionally disabled",
        "InstallAndroidFilePickerBridge();",
    )
    required_picker = (
        "UnityFileDialog.FileBrowser.PickFile hook installed",
        "g_ourClassLoader",
        "LoadClass(JNIEnv* env",
    )
    missing = [marker for marker in required_main if marker not in main]
    missing += [marker for marker in required_picker if marker not in picker]
    if missing:
        raise SystemExit(f"mobile editor shell transform prerequisites missing: {missing}")


def apply(source: Path, repo_root: Path) -> None:
    verify_post_transforms(source)
    jni = source / "app/src/main/jni"

    for name in ("MobileEditorBridge.cpp", "MobileEditorBridge.h"):
        src = repo_root / "native" / name
        if not src.is_file():
            raise SystemExit(f"repository native source missing: {src}")
        shutil.copy2(src, jni / name)

    android_mk = jni / "Android.mk"
    replace_once(
        android_mk,
        "    FilePicker.cpp \\\n    Hooks.cpp\n",
        "    FilePicker.cpp \\\n    MobileEditorBridge.cpp \\\n    Hooks.cpp\n",
        "Android.mk mobile editor bridge source",
    )

    picker_h = jni / "FilePicker.h"
    replace_once(
        picker_h,
        "// 安装文件选择器 hook\nvoid InstallFilePickerHook();\n",
        "// 启动 repository-owned Android mobile editor shell.\nvoid InitJavaMobileEditorShell(JNIEnv* env);\n\n// 安装文件选择器 hook\nvoid InstallFilePickerHook();\n",
        "FilePicker.h editor shell declaration",
    )

    picker_cpp = jni / "FilePicker.cpp"
    replace_once(
        picker_cpp,
        "void InstallFilePickerHook() {\n",
        '''void InitJavaMobileEditorShell(JNIEnv* env) {
    if (!env) return;
    jclass editorClass = LoadClass(env, "com/unity3d/player/MobileEditorShell");
    if (!editorClass) {
        LOGE("MobileEditorShell class not found in injected DEX");
        return;
    }

    jmethodID install = env->GetStaticMethodID(editorClass, "installLauncher", "()V");
    if (!install) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        LOGE("MobileEditorShell.installLauncher missing");
        return;
    }

    env->CallStaticVoidMethod(editorClass, install);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        LOGE("MobileEditorShell.installLauncher threw");
        return;
    }
    LOGD("MobileEditorShell launcher installed through injected DEX loader");
}

void InstallFilePickerHook() {
''',
        "FilePicker.cpp editor shell loader",
    )

    main = jni / "Main.cpp"
    replace_once(
        main,
        '#include "FilePicker.h"\n#include "Hooks.h"\n',
        '#include "FilePicker.h"\n#include "MobileEditorBridge.h"\n#include "Hooks.h"\n',
        "Main.cpp mobile editor bridge include",
    )
    replace_once(
        main,
        "static void InstallAndroidFilePickerBridge() {\n",
        "static void InstallAndroidFilePickerBridge(bool installEditorShell = false) {\n",
        "Main.cpp picker helper argument",
    )
    replace_once(
        main,
        "    InitJavaFilePicker(env);\n    InstallFilePickerHook();\n",
        "    InitJavaFilePicker(env);\n    if (installEditorShell) InitJavaMobileEditorShell(env);\n    InstallFilePickerHook();\n",
        "Main.cpp editor shell initialization",
    )
    replace_once(
        main,
        '''        InstallAndroidFilePickerBridge();
        LOGD("Modern runtime platform bridge installed");
        return;
''',
        '''        InstallAndroidFilePickerBridge(true);
        InstallMobileEditorPreviewBridge();
        LOGD("Modern runtime platform + mobile editor shell installed");
        return;
''',
        "Main.cpp modern editor shell profile",
    )

    print(f"Applied Android mobile editor shell to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    apply(args.source.resolve(), args.repo_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
