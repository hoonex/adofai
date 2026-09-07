from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import patch_unity_activity


class UnityActivityPatchTests(unittest.TestCase):
    def test_injects_load_library_and_bumps_locals(self):
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
        self.assertIn('.locals 1', patched)
        self.assertIn('const-string v0, "October"', patched)
        self.assertIn('System;->loadLibrary(Ljava/lang/String;)V', patched)

        patched_again, changed_again = patch_unity_activity.patch_smali_text(patched)
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_registers_form_reserves_v0(self):
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
        self.assertIn('.registers 3', patched)
        self.assertIn('const-string v0, "October"', patched)

    def test_missing_oncreate_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_unity_activity.patch_smali_text(
                '.class public Lcom/unity3d/player/UnityPlayerActivity;\n'
                '.super Landroid/app/Activity;\n'
            )


if __name__ == "__main__":
    unittest.main()
