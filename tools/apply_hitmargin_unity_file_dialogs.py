#!/usr/bin/env python3
"""Add the ADOFAI 3.3 UnityFileDialog backend to the pinned mobile picker bridge.

ADOFAI 3.3.1 no longer contains SFB.StandaloneFileBrowser. Its current editor uses
UnityFileDialog.FileBrowser instead. This transform runs after
apply_hitmargin_file_dialogs.py, keeps the legacy SFB hooks for older supported
builds, and adds exact PickFile/PickFiles/SaveFile hooks for UnityFileDialog.

The transform intentionally fails closed if the expected post-transform anchors are
missing. Runtime signature safety is still bounded by the exact 3.3.1 metadata
evidence and must be proven on-device after packaging.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
FILE = "app/src/main/jni/FilePicker.cpp"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one replacement anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def verify_post_dialog_transform(source: Path) -> None:
    head = git(source, "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"upstream HEAD mismatch: expected {EXPECTED_HEAD}, got {head}")

    text = (source / FILE).read_text(encoding="utf-8")
    required = (
        "enum class PickerMode",
        "static std::string RunJavaPicker",
        "static Array<String*>* PathToStringArray",
        'Class("SFB", "StandaloneFileBrowser")',
        "Hooked_SaveFilePanel",
        "Hooked_OpenFolderPanel",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise SystemExit(
            "UnityFileDialog transform must run after apply_hitmargin_file_dialogs.py; "
            f"missing post-transform markers: {missing}"
        )


def apply(source: Path) -> None:
    verify_post_dialog_transform(source)
    path = source / FILE

    legacy_install = '''void InstallFilePickerHook() {
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
'''

    dual_backend = '''static std::string UnityFileDialogExtensions(Array<String*>* extensions) {
    if (!extensions || extensions->capacity == 0) return "*";

    std::string result;
    for (int i = 0; i < extensions->capacity; i++) {
        auto item = extensions->m_Items[i];
        if (!item) continue;

        std::string ext = item->str();
        while (!ext.empty() && (ext.front() == ' ' || ext.front() == '\\t')) ext.erase(ext.begin());
        while (!ext.empty() && (ext.back() == ' ' || ext.back() == '\\t')) ext.pop_back();
        if (ext == "*" || ext == "*.*") return "*";
        if (ext.rfind("*.", 0) == 0) ext.erase(0, 2);
        else if (!ext.empty() && ext.front() == '.') ext.erase(ext.begin());
        if (ext.empty()) continue;

        if (!result.empty()) result += ",";
        result += ext;
    }
    return result.empty() ? "*" : result;
}

// ADOFAI 3.3.1 ships UnityFileDialog.dll instead of SFB.StandaloneFileBrowser.
// Signatures are matched to UnityFileDialog.FileBrowser in the 3.3.1 metadata:
//   string PickFile(string, string, string[], string)
//   string[] PickFiles(string, string, string[], string)
//   string SaveFile(string, string, string, string[], string)
String* Hooked_UnityFileDialog_PickFile(
        String* directory,
        String* filterName,
        Array<String*>* filterExtensions,
        String* title) {
    if (!g_selectorClass || !g_method_selectFile) {
        LOGE("FileSelector open bridge not initialized for UnityFileDialog.PickFile");
        return CreateMonoString("");
    }

    std::string filter = UnityFileDialogExtensions(filterExtensions);
    std::string filePath = RunJavaPicker(PickerMode::OpenFile, filter);
    if (filePath.empty()) LOGW("UnityFileDialog.PickFile cancelled or returned no file");
    else LOGD("UnityFileDialog.PickFile selected: %s", filePath.c_str());
    return CreateMonoString(filePath);
}

Array<String*>* Hooked_UnityFileDialog_PickFiles(
        String* directory,
        String* filterName,
        Array<String*>* filterExtensions,
        String* title) {
    if (!g_selectorClass || !g_method_selectFile) {
        LOGE("FileSelector open bridge not initialized for UnityFileDialog.PickFiles");
        return PathToStringArray("");
    }

    // The Android bridge currently exposes one selected path. Returning a one-item
    // array preserves the managed API shape without pretending multi-select exists.
    std::string filter = UnityFileDialogExtensions(filterExtensions);
    std::string filePath = RunJavaPicker(PickerMode::OpenFile, filter);
    if (filePath.empty()) LOGW("UnityFileDialog.PickFiles cancelled or returned no file");
    else LOGD("UnityFileDialog.PickFiles selected one file: %s", filePath.c_str());
    return PathToStringArray(filePath);
}

String* Hooked_UnityFileDialog_SaveFile(
        String* directory,
        String* filename,
        String* filterName,
        Array<String*>* filterExtensions,
        String* title) {
    if (!g_selectorClass || !g_method_saveAs) {
        LOGE("FileSelector save bridge not initialized for UnityFileDialog.SaveFile");
        return CreateMonoString("");
    }

    std::string name = filename ? filename->str() : "level";
    std::string filter = UnityFileDialogExtensions(filterExtensions);
    if (name.find('.') == std::string::npos && filter != "*" && filter.find(',') == std::string::npos) {
        name += "." + filter;
    }

    std::string filePath = RunJavaPicker(PickerMode::SaveFile, name);
    if (filePath.empty()) LOGW("UnityFileDialog.SaveFile cancelled");
    else LOGD("UnityFileDialog.SaveFile selected: %s", filePath.c_str());
    return CreateMonoString(filePath);
}

static bool InstallUnityFileDialogHooks() {
    auto fileBrowserClass = Class("UnityFileDialog", "FileBrowser");

    auto pickFile = fileBrowserClass.GetMethod(
        "PickFile", {"directory", "filterName", "filterExtensions", "title"});
    auto pickFiles = fileBrowserClass.GetMethod(
        "PickFiles", {"directory", "filterName", "filterExtensions", "title"});
    auto saveFile = fileBrowserClass.GetMethod(
        "SaveFile", {"directory", "filename", "filterName", "filterExtensions", "title"});

    bool installed = false;
    if (pickFile.IsValid()) {
        BasicHook(pickFile, Hooked_UnityFileDialog_PickFile, (void*)nullptr);
        LOGD("UnityFileDialog.FileBrowser.PickFile hook installed");
        installed = true;
    }
    if (pickFiles.IsValid()) {
        BasicHook(pickFiles, Hooked_UnityFileDialog_PickFiles, (void*)nullptr);
        LOGD("UnityFileDialog.FileBrowser.PickFiles hook installed");
        installed = true;
    }
    if (saveFile.IsValid() && g_method_saveAs) {
        BasicHook(saveFile, Hooked_UnityFileDialog_SaveFile, (void*)nullptr);
        LOGD("UnityFileDialog.FileBrowser.SaveFile hook installed");
        installed = true;
    }

    if (!installed) LOGW("UnityFileDialog.FileBrowser backend unavailable");
    return installed;
}

static bool InstallLegacySfbHooks() {
    auto browserClass = Class("SFB", "StandaloneFileBrowser");
    bool installed = false;

    auto openFilePanel = browserClass.GetMethod("OpenFilePanel", {"title","directory","extension","multiselect"});
    if (openFilePanel.IsValid()) {
        BasicHook(openFilePanel, Hooked_OpenFilePanel, (void*)nullptr);
        LOGD("Legacy SFB OpenFilePanel hook installed");
        installed = true;
    }

    auto saveFilePanel = browserClass.GetMethod("SaveFilePanel", {"title","directory","defaultName","extension"});
    if (saveFilePanel.IsValid() && g_method_saveAs) {
        BasicHook(saveFilePanel, Hooked_SaveFilePanel, (void*)nullptr);
        LOGD("Legacy SFB SaveFilePanel hook installed");
        installed = true;
    }

    auto openFolderPanel = browserClass.GetMethod("OpenFolderPanel", {"title","directory","multiselect"});
    if (openFolderPanel.IsValid() && g_method_selectFolder && g_method_getFolderPath) {
        BasicHook(openFolderPanel, Hooked_OpenFolderPanel, (void*)nullptr);
        LOGD("Legacy SFB OpenFolderPanel hook installed");
        installed = true;
    }

    if (!installed) LOGD("Legacy SFB backend not present; this is expected on ADOFAI 3.3+");
    return installed;
}

void InstallFilePickerHook() {
    bool unityFileDialog = InstallUnityFileDialogHooks();
    bool legacySfb = InstallLegacySfbHooks();
    if (!unityFileDialog && !legacySfb) {
        LOGE("No supported managed file-dialog backend was found");
    }
}
'''

    replace_once(
        path,
        legacy_install,
        dual_backend,
        "FilePicker legacy/UnityFileDialog backend installation",
    )

    print(f"Applied UnityFileDialog backend transform to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
