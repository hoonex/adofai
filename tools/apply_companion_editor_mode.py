#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    required = (
        "requestClose",
        "confirmOpenPath",
        "pickerInFlight",
        "Unexpected trailing content after JSON value",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"companion prerequisites missing: {missing}")

    text = replace_exact(
        text,
        'TextView title = text("ADOFAI Mobile Editor", 20, Color.WHITE);',
        'TextView title = text("ADOFAI Companion Editor", 20, Color.WHITE);',
        "product title",
    )

    text = replace_exact(
        text,
        '''        actions.addView(makeAction("Open", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpen(); }
        }));
''',
        '''        actions.addView(makeAction("New", new View.OnClickListener() {
            @Override public void onClick(View v) { newChart(); }
        }));
        actions.addView(makeAction("Open", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpen(); }
        }));
        actions.addView(makeAction("ZIP URL", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpenUrl(); }
        }));
''',
        "New and ZIP URL action insertion",
    )

    text = replace_exact(
        text,
        '''        actions.addView(makeAction("Preview", new View.OnClickListener() {
            @Override public void onClick(View v) { previewCurrent(); }
        }));
''',
        '''        actions.addView(makeAction("공식 ADOFAI", new View.OnClickListener() {
            @Override public void onClick(View v) { shareCurrent(); }
        }));
''',
        "official handoff action replacement",
    )

    text = replace_exact(
        text,
        '''    private static Activity getCurrentActivity() {
''',
        '''    public static void openStandalone() {
        final Activity resolved = getCurrentActivity();
        if (resolved == null) {
            Log.e(TAG, "Cannot open companion editor: no foreground Activity");
            return;
        }
        activity = resolved;
        resolved.runOnUiThread(new Runnable() {
            @Override public void run() {
                openEditor(resolved);
            }
        });
    }

    public static void openStandalonePath(final String path) {
        final Activity resolved = getCurrentActivity();
        if (resolved == null) {
            Log.e(TAG, "Cannot open companion chart: no foreground Activity");
            return;
        }
        activity = resolved;
        resolved.runOnUiThread(new Runnable() {
            @Override public void run() {
                openEditor(resolved);
                if (path != null && path.length() > 0) confirmOpenPath(path);
            }
        });
    }

    public static void showOfficialHandoffDiagnostic(final String message) {
        if (message == null || message.length() == 0) return;
        final Activity resolved = getCurrentActivity();
        if (resolved == null || resolved.isFinishing()) return;
        resolved.runOnUiThread(new Runnable() {
            @Override public void run() {
                if (statusView != null) setStatus(message, false);
                Toast.makeText(resolved, message, Toast.LENGTH_LONG).show();
            }
        });
    }

    private static Activity getCurrentActivity() {
''',
        "standalone entry points and handoff diagnostic surface",
    )

    text = replace_exact(
        text,
        '''            Os.rename(temp.getAbsolutePath(), target.getAbsolutePath());
            currentPath = target.getAbsolutePath();
''',
        '''            Os.rename(temp.getAbsolutePath(), target.getAbsolutePath());
            if (!FileSelector.syncSavedPath(target.getAbsolutePath())) {
                throw new IllegalStateException("Could not synchronize saved chart/bundle to the selected Android document");
            }
            currentPath = target.getAbsolutePath();
''',
        "SAF save synchronization",
    )

    text = replace_exact(
        text,
        '''        String value = currentPath == null ? "No chart open" : currentPath;
''',
        '''        String value = currentPath == null ? "No chart open" : FileSelector.displayNameForPath(currentPath);
''',
        "document display name",
    )

    text = replace_exact(
        text,
        '''    private static void beginOpen() {
''',
        '''    private static void beginOpenUrl() {
        setStatus("ZIP URL 입력…", false);
        FileSelector.selectUrlBundle();
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("ZIP URL 열기 취소 또는 실패", false);
                    return;
                }
                loadPath(path);
            }
        });
    }

    private static void beginOpen() {
''',
        "ZIP URL open flow",
    )

    text = replace_exact(
        text,
        '''    private static void markDirty(String message) {
''',
        '''    private static void newChart() {
        if (dirty) {
            setStatus("현재 수정사항을 저장하거나 버린 뒤 새 맵을 만드세요", true);
            return;
        }
        try {
            JSONObject fresh = new JSONObject();
            fresh.put("pathData", "R");
            JSONObject settings = new JSONObject();
            settings.put("version", 15);
            settings.put("bpm", 100);
            settings.put("song", "Untitled");
            fresh.put("settings", settings);
            fresh.put("actions", new JSONArray());
            fresh.put("decorations", new JSONArray());
            document = fresh;
            currentPath = null;
            dirty = true;
            updatePath();
            showTab(0);
            setStatus("새 맵을 만들었습니다. Save As로 .adofai 파일을 저장하세요", false);
        } catch (JSONException error) {
            reportError("New chart failed", error);
        }
    }

    private static void confirmSaveAndShare() {
        final Activity owner = activity;
        if (owner == null || owner.isFinishing()) {
            setStatus("Unsaved changes: cannot confirm official handoff without a foreground Activity", true);
            return;
        }
        new android.app.AlertDialog.Builder(owner)
                .setTitle("저장 후 공식 ADOFAI 열기")
                .setMessage("현재 수정사항과 ZIP bundle을 먼저 저장한 뒤 설치된 공식 Play판 ADOFAI 3.3.1에 전달합니다.")
                .setPositiveButton("저장 후 계속", new android.content.DialogInterface.OnClickListener() {
                    @Override public void onClick(android.content.DialogInterface ignored, int which) {
                        if (saveCurrent(false)) shareCurrent();
                    }
                })
                .setNegativeButton("계속 편집", null)
                .show();
    }

    private static void shareCurrent() {
        if (document == null) {
            setStatus("먼저 맵 또는 ZIP bundle을 여세요", true);
            return;
        }
        if (currentPath == null) {
            setStatus("공식 ADOFAI로 넘기기 전에 Save As로 파일을 저장하세요", true);
            return;
        }
        if (dirty) {
            confirmSaveAndShare();
            return;
        }
        boolean opened = OfficialGameBridge.open(currentPath);
        setStatus(OfficialGameBridge.getLastStatus(), !opened);
    }

    private static void markDirty(String message) {
''',
        "companion actions",
    )

    return text


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.java> <output.java>", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    try:
        rendered = transform(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        print(f"companion transform failed: {error}", file=sys.stderr)
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared ADOFAI companion editor shell: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
