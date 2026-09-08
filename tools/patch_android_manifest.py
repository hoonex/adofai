#!/usr/bin/env python3
"""Patch an apktool-decoded AndroidManifest.xml for the mobile editor payload.

Only storage capabilities needed by the raw-path editor file browser are added.
The operation is idempotent and preserves existing application attributes unless
one of the required attributes is absent.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ANDROID_NS = "http://schemas.android.com/apk/res/android"
A = "{%s}" % ANDROID_NS
ET.register_namespace("android", ANDROID_NS)

REQUIRED_PERMISSIONS = (
    ("android.permission.READ_EXTERNAL_STORAGE", {}),
    ("android.permission.WRITE_EXTERNAL_STORAGE", {A + "maxSdkVersion": "28"}),
    ("android.permission.MANAGE_EXTERNAL_STORAGE", {}),
)


def patch_manifest(path: Path) -> bool:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "manifest":
        raise ValueError("AndroidManifest.xml root is not <manifest>")

    changed = False
    existing = {
        node.get(A + "name"): node
        for node in root.findall("uses-permission")
        if node.get(A + "name")
    }

    app = root.find("application")
    if app is None:
        raise ValueError("AndroidManifest.xml does not contain <application>")

    app_index = list(root).index(app)
    for permission_name, attrs in REQUIRED_PERMISSIONS:
        node = existing.get(permission_name)
        if node is None:
            node = ET.Element("uses-permission")
            node.set(A + "name", permission_name)
            for key, value in attrs.items():
                node.set(key, value)
            root.insert(app_index, node)
            app_index += 1
            existing[permission_name] = node
            changed = True
        else:
            for key, value in attrs.items():
                if node.get(key) != value:
                    node.set(key, value)
                    changed = True

    if app.get(A + "requestLegacyExternalStorage") != "true":
        app.set(A + "requestLegacyExternalStorage", "true")
        changed = True

    if changed:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch decoded AndroidManifest.xml for ADOFAI mobile editor storage access")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    changed = patch_manifest(args.manifest)
    print("manifest patched" if changed else "manifest already satisfies requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
