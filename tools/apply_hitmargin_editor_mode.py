#!/usr/bin/env python3
"""Apply the mobile editor-mode fixes to the pinned HitMargin source exactly.

Unlike a hand-written unified diff, this transform verifies the upstream commit and
Git blob identities first, then performs unique semantic replacements. If upstream
moves, the transform fails closed instead of partially applying to the wrong game
hook source.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
EXPECTED_BLOBS = {
    "ADOFAI-Mod-Info.json.example": "cbf673dfb5a891b3af0c666ba720bb11a0b8e2c4",
    "app/src/main/jni/Config.h": "45546af8cf6768cf94337cb416929d8838b8cfe3",
    "app/src/main/jni/Config.cpp": "a26b5d7badcf9693a5a5ebba56981fef4848aeb3",
    "app/src/main/jni/Hooks.cpp": "bbfb752ecdeaed8c4136f4610f3394ee8fc53809",
    "app/src/main/jni/Main.cpp": "eb0a4e215a1021fd27a06c44544882fd70527d02",
}


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_identity(source: Path) -> None:
    actual_head = git(source, "rev-parse", "HEAD")
    if actual_head != EXPECTED_HEAD:
        raise SystemExit(f"upstream HEAD mismatch: expected {EXPECTED_HEAD}, got {actual_head}")

    for relative, expected_blob in EXPECTED_BLOBS.items():
        actual_blob = git(source, "hash-object", relative)
        if actual_blob != expected_blob:
            raise SystemExit(
                f"upstream blob mismatch for {relative}: expected {expected_blob}, got {actual_blob}"
            )


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one replacement anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def apply(source: Path) -> None:
    verify_identity(source)

    replace_once(
        source / "ADOFAI-Mod-Info.json.example",
        '  "enableCustomUIHitTest": true,           // 使用自定义 UI 点击检测逻辑（IsScreenPointInsideUIElements 使用 EventSystem Raycast）\n',
        '  "enableCustomUIHitTest": true,           // 使用自定义 UI 点击检测逻辑（IsScreenPointInsideUIElements 使用 EventSystem Raycast）\n'
        '  "enableEditorDesktopMode": true,         // 编辑器场景使用桌面编辑器布局/输入分支\n',
        "config example editor mode",
    )

    replace_once(
        source / "app/src/main/jni/Config.h",
        "    bool enableCustomUIHitTest = true;\n",
        "    bool enableCustomUIHitTest = true;\n"
        "    bool enableEditorDesktopMode = true;\n",
        "Config.h editor mode",
    )

    replace_once(
        source / "app/src/main/jni/Config.cpp",
        '            else SET_BOOL("enableCustomUIHitTest", enableCustomUIHitTest);\n',
        '            else SET_BOOL("enableCustomUIHitTest", enableCustomUIHitTest);\n'
        '            else SET_BOOL("enableEditorDesktopMode", enableEditorDesktopMode);\n',
        "Config.cpp parse editor mode",
    )

    replace_once(
        source / "app/src/main/jni/Config.cpp",
        '            outFile << "  \\"enableCustomUIHitTest\\": true,\\n";\n',
        '            outFile << "  \\"enableCustomUIHitTest\\": true,\\n";\n'
        '            outFile << "  \\"enableEditorDesktopMode\\": true,\\n";\n',
        "Config.cpp default editor mode",
    )

    old_mobile = '''bool (*old_isMobile)() = nullptr;
bool IsMobile() {
    auto scene = g_get_sceneName.Call();
    if (scene && (scene->str() == "scnTaroMenu0" || scene->str() == "scnTaroMenu1" ||
                  scene->str() == "scnTaroMenu2" || scene->str() == "scnTaroMenu3")) {
        return false;
    }
    return old_isMobile();
}
'''
    new_mobile = '''static bool NeedsDesktopEditorMode(const std::string& sceneName) {
    if (sceneName == "scnEditor" || sceneName.rfind("scnEditor", 0) == 0) {
        return true;
    }
    return sceneName == "scnTaroMenu0" || sceneName == "scnTaroMenu1" ||
           sceneName == "scnTaroMenu2" || sceneName == "scnTaroMenu3";
}

bool (*old_isMobile)() = nullptr;
bool IsMobile() {
    auto scene = g_get_sceneName.Call();
    if (scene && NeedsDesktopEditorMode(scene->str())) {
        return false;
    }
    return old_isMobile ? old_isMobile() : true;
}
'''
    replace_once(
        source / "app/src/main/jni/Hooks.cpp",
        old_mobile,
        new_mobile,
        "Hooks.cpp mobile/editor mode",
    )

    old_install = '''    if (g_modConfig.enableLoadLevel) {
        auto isEditor = Class("","ADOBase").GetMethod("get_isUnityEditor");
'''
    new_install = '''    if (g_modConfig.enableEditorDesktopMode) {
        auto mobileMode = Class("", "ADOBase").GetMethod("get_isMobile");
        if (mobileMode.IsValid()) {
            BasicHook(mobileMode, IsMobile, old_isMobile);
            LOGD("Hook: enableEditorDesktopMode enabled");
        } else {
            LOGE("Hook: enableEditorDesktopMode unavailable (ADOBase.get_isMobile not found)");
        }
    }

    if (g_modConfig.enableLoadLevel) {
        auto isEditor = Class("","ADOBase").GetMethod("get_isUnityEditor");
'''
    replace_once(
        source / "app/src/main/jni/Main.cpp",
        old_install,
        new_install,
        "Main.cpp install editor mode hook",
    )

    print(f"Applied editor-mode transform to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
