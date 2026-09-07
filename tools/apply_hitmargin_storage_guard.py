#!/usr/bin/env python3
"""Harden the pinned mobile file selector for modern Android storage/runtime behavior.

The original bridge can leave isDone=false indefinitely when no Activity can be
resolved or storage permission is missing. This transform fails closed, prefers
UnityPlayer.currentActivity over hidden ActivityThread internals, and guides
Android 11+ users to the app-specific all-files-access settings required by the
upstream raw-path file browser.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_HEAD = "74bcc7a0d8c8be1267504e21e28a35e199b5d4eb"
FILE = "app/src/main/java/com/unity3d/player/FileSelector.java"
EXPECTED_BLOB = "6484281a04bea2f9f3104c3441cb3fca6022dea0"


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
        '''import android.app.Activity;
import android.util.Log;

import java.lang.reflect.Field;
''',
        '''import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;

import java.lang.reflect.Field;
''',
        "FileSelector imports",
    )

    replace_once(
        path,
        '''        final Activity activity = getCurrentActivity();
        if (activity != null) {
            activity.runOnUiThread(new Runnable() {
                @Override public void run() {
                    new CustomFileChooser(activity, true, false, name, finalExt).show();
                }
            });
        }
''',
        '''        final Activity activity = getCurrentActivity();
        if (activity == null) {
            Log.e("Unity", "FileSelector.saveAs: no foreground Activity");
            setPath(null);
            return;
        }
        if (!ensureStorageAccess(activity)) return;
        activity.runOnUiThread(new Runnable() {
            @Override public void run() {
                new CustomFileChooser(activity, true, false, name, finalExt).show();
            }
        });
''',
        "saveAs completion guard",
    )

    replace_once(
        path,
        '''        final Activity activity = getCurrentActivity();
        if (activity != null) {
            activity.runOnUiThread(new Runnable() {
                @Override public void run() {
                    new CustomFileChooser(activity, false, false, null, finalExt).show();
                }
            });
        }
''',
        '''        final Activity activity = getCurrentActivity();
        if (activity == null) {
            Log.e("Unity", "FileSelector.selectFile: no foreground Activity");
            setPath(null);
            return;
        }
        if (!ensureStorageAccess(activity)) return;
        activity.runOnUiThread(new Runnable() {
            @Override public void run() {
                new CustomFileChooser(activity, false, false, null, finalExt).show();
            }
        });
''',
        "selectFile completion guard",
    )

    replace_once(
        path,
        '''        final Activity activity = getCurrentActivity();
        if (activity != null) {
            activity.runOnUiThread(new Runnable() {
                @Override public void run() {
                    new CustomFileChooser(activity, false, true, null, "*").show();
                }
            });
        }
''',
        '''        final Activity activity = getCurrentActivity();
        if (activity == null) {
            Log.e("Unity", "FileSelector.selectFolder: no foreground Activity");
            setPath(null);
            return;
        }
        if (!ensureStorageAccess(activity)) return;
        activity.runOnUiThread(new Runnable() {
            @Override public void run() {
                new CustomFileChooser(activity, false, true, null, "*").show();
            }
        });
''',
        "selectFolder completion guard",
    )

    old_activity = '''    private static Activity getCurrentActivity() {
        if (context == null) {
            try {
                Class activityThreadClass = Class.forName("android.app.ActivityThread");
                Object activityThread = activityThreadClass.getMethod("currentActivityThread").invoke(null);
                Field activitiesField = activityThreadClass.getDeclaredField("mActivities");
                activitiesField.setAccessible(true);
                Map activities = (Map) activitiesField.get(activityThread);
                for (Object activityRecord : activities.values()) {
                    Class activityRecordClass = activityRecord.getClass();
                    Field pausedField = activityRecordClass.getDeclaredField("paused");
                    pausedField.setAccessible(true);
                    if (!pausedField.getBoolean(activityRecord)) {
                        Field activityField = activityRecordClass.getDeclaredField("activity");
                        activityField.setAccessible(true);
                        Activity activity = (Activity) activityField.get(activityRecord);
                        context = activity;
                        return activity;
                    }
                }
            } catch (ClassNotFoundException e) {
                Log.e("Unity", "Could not get current activity", e);
            } catch (NoSuchFieldException e) {
                Log.e("Unity", "Could not get current activity", e);
            } catch (IllegalAccessException e) {
                Log.e("Unity", "Could not get current activity", e);
            } catch (NoSuchMethodException e) {
                Log.e("Unity", "Could not get current activity", e);
            } catch (InvocationTargetException e) {
                Log.e("Unity", "Could not get current activity", e);
            }
        }
        return context;
    }
'''
    new_activity = '''    private static Activity getCurrentActivity() {
        if (context != null && !context.isFinishing()) return context;

        // Prefer Unity's public currentActivity field. This avoids relying on hidden
        // ActivityThread internals on newer Android releases.
        try {
            Class unityPlayerClass = Class.forName("com.unity3d.player.UnityPlayer");
            Field currentActivityField = unityPlayerClass.getField("currentActivity");
            Activity activity = (Activity) currentActivityField.get(null);
            if (activity != null) {
                context = activity;
                return activity;
            }
        } catch (Throwable e) {
            Log.w("Unity", "UnityPlayer.currentActivity unavailable; using fallback", e);
        }

        try {
            Class activityThreadClass = Class.forName("android.app.ActivityThread");
            Object activityThread = activityThreadClass.getMethod("currentActivityThread").invoke(null);
            Field activitiesField = activityThreadClass.getDeclaredField("mActivities");
            activitiesField.setAccessible(true);
            Map activities = (Map) activitiesField.get(activityThread);
            for (Object activityRecord : activities.values()) {
                Class activityRecordClass = activityRecord.getClass();
                Field pausedField = activityRecordClass.getDeclaredField("paused");
                pausedField.setAccessible(true);
                if (!pausedField.getBoolean(activityRecord)) {
                    Field activityField = activityRecordClass.getDeclaredField("activity");
                    activityField.setAccessible(true);
                    Activity activity = (Activity) activityField.get(activityRecord);
                    context = activity;
                    return activity;
                }
            }
        } catch (Throwable e) {
            Log.e("Unity", "Could not get current activity", e);
        }
        return null;
    }

    private static boolean ensureStorageAccess(final Activity activity) {
        if (Build.VERSION.SDK_INT >= 30) {
            if (Environment.isExternalStorageManager()) return true;

            // The upstream chooser returns raw filesystem paths so sibling audio,
            // images, videos and other level assets continue to resolve normally.
            // That requires all-files access on Android 11+.
            activity.runOnUiThread(new Runnable() {
                @Override public void run() {
                    new AlertDialog.Builder(activity)
                        .setTitle("File access required")
                        .setMessage("Allow file access for the ADOFAI editor, then open or save the level again.")
                        .setPositiveButton("Open settings", new DialogInterface.OnClickListener() {
                            @Override public void onClick(DialogInterface dialog, int which) {
                                try {
                                    Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                                    intent.setData(Uri.parse("package:" + activity.getPackageName()));
                                    activity.startActivity(intent);
                                } catch (Throwable appSettingsError) {
                                    try {
                                        activity.startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
                                    } catch (Throwable ignored) {
                                        Log.e("Unity", "Unable to open all-files access settings", appSettingsError);
                                    }
                                }
                            }
                        })
                        .setNegativeButton("Cancel", null)
                        .show();
                }
            });
            setPath(null);
            return false;
        }

        if (Build.VERSION.SDK_INT >= 23 &&
                activity.checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
            activity.runOnUiThread(new Runnable() {
                @Override public void run() {
                    activity.requestPermissions(
                        new String[] { Manifest.permission.READ_EXTERNAL_STORAGE }, 9301);
                }
            });
            setPath(null);
            return false;
        }
        return true;
    }
'''
    replace_once(path, old_activity, new_activity, "Activity/storage guard")

    print(f"Applied Android storage/runtime guard to pinned upstream {EXPECTED_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    apply(args.source.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
