#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    required = (
        "ADOFAI Companion Editor",
        "shareCurrent",
        "confirmSaveAndShare",
        "requestClose",
        "Unexpected trailing content after JSON value",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"custom-game prerequisites missing: {missing}")

    text = replace_once(
        text,
        'TextView title = text("ADOFAI Companion Editor", 20, Color.WHITE);',
        'TextView title = text("ADOFAI Custom Editor", 20, Color.WHITE);',
        "custom product title",
    )
    text = replace_once(
        text,
        '''        actions.addView(makeAction("ADOFAI / 공유", new View.OnClickListener() {
            @Override public void onClick(View v) { shareCurrent(); }
        }));
''',
        '''        actions.addView(makeAction("Play", new View.OnClickListener() {
            @Override public void onClick(View v) { playCurrent(); }
        }));
''',
        "custom Play action",
    )

    marker = "    private static void confirmSaveAndShare() {\n"
    play = '''    private static void playCurrent() {
        if (document == null) {
            setStatus("먼저 맵을 열거나 새로 만드세요", true);
            return;
        }
        if (currentPath == null) {
            setStatus("플레이하기 전에 Save As로 맵을 저장하세요", true);
            return;
        }
        if (dirty && !saveCurrent(false)) return;
        if (!CustomPlayerBridge.open(currentPath)) {
            setStatus("Custom Player를 열 수 없습니다", true);
            return;
        }
        setStatus("Custom Player 실행", false);
        if (dialog != null) dialog.dismiss();
    }

'''
    text = replace_once(text, marker, play + marker, "custom player method")
    return text


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.java> <output.java>", file=sys.stderr)
        return 2
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        rendered = transform(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        print(f"custom-game transform failed: {error}", file=sys.stderr)
        return 3
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared ADOFAI Custom editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
