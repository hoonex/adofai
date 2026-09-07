from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import patch_unity_activity


class UnityActivityPatchTests(unittest.TestCase):
    def test_injects_zero_register_secondary_dex_bootstrap(self):
        source = '''.class public Lcom/unity3d/player/UnityPlayerActivity;
.super Landroid/app/Activity;

.method protected onCreate(Landroid/os/Bundle;)V
    .locals 0
    .param p1, "savedInstanceState"    # Landroid/os/Bundle;

    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V
    return-void
.end method
'''
        patched, changed = patch_unity_activity.patch_smali_text(source)
        self.assertTrue(changed)
        self.assertIn('.locals 0', patched)
        self.assertIn(
            'invoke-static {}, Lcom/unity3d/player/MobileEditorBootstrap;->init()V',
            patched,
        )
        self.assertNotIn('const-string v0, "October"', patched)

        patched_again, changed_again = patch_unity_activity.patch_smali_text(patched)
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_registers_form_without_scratch_local_is_now_safe(self):
        source = '''.class public Lcom/unity3d/player/UnityPlayerActivity;
.super Landroid/app/Activity;

.method public onCreate(Landroid/os/Bundle;)V
    .registers 2
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V
    return-void
.end method
'''
        patched, changed = patch_unity_activity.patch_smali_text(source)
        self.assertTrue(changed)
        self.assertIn('.registers 2', patched)
        self.assertIn(
            'invoke-static {}, Lcom/unity3d/player/MobileEditorBootstrap;->init()V',
            patched,
        )

    def test_existing_direct_october_bootstrap_is_preserved(self):
        source = '''.class public Lcom/unity3d/player/UnityPlayerActivity;
.super Landroid/app/Activity;

.method public onCreate(Landroid/os/Bundle;)V
    .locals 1
    const-string v0, "October"
    invoke-static {v0}, Ljava/lang/System;->loadLibrary(Ljava/lang/String;)V
    return-void
.end method
'''
        patched, changed = patch_unity_activity.patch_smali_text(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)

    def test_missing_register_directive_fails_closed(self):
        source = '''.class public Lcom/unity3d/player/UnityPlayerActivity;
.super Landroid/app/Activity;

.method public onCreate(Landroid/os/Bundle;)V
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V
    return-void
.end method
'''
        with self.assertRaisesRegex(ValueError, "neither .locals nor .registers"):
            patch_unity_activity.patch_smali_text(source)

    def test_missing_oncreate_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_unity_activity.patch_smali_text(
                '.class public Lcom/unity3d/player/UnityPlayerActivity;\n'
                '.super Landroid/app/Activity;\n'
            )

    def test_canonical_payload_contains_bootstrap_class(self):
        build = (ROOT / "scripts" / "build-payload.sh").read_text(encoding="utf-8")
        bootstrap = (
            ROOT
            / "android"
            / "mobile-editor-shell"
            / "src"
            / "com"
            / "unity3d"
            / "player"
            / "MobileEditorBootstrap.java"
        ).read_text(encoding="utf-8")
        self.assertIn('EDITOR_BOOTSTRAP="${ROOT}/android/mobile-editor-shell/src/com/unity3d/player/MobileEditorBootstrap.java"', build)
        self.assertIn('"${EDITOR_BOOTSTRAP}" \\', build)
        self.assertIn('System.loadLibrary("October")', bootstrap)
        self.assertIn('public static synchronized void init()', bootstrap)


if __name__ == "__main__":
    unittest.main()
