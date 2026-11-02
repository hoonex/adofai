"""Read-only browser port baseline for the PC scnEditor serialized scene.

The proprietary Unity scene and DLL are fetched on demand through byte ranges
from the user's existing Drive ZIP, verified against pinned SHA-256 hashes, and
never bundled into, exposed by, or committed with this web project.

This endpoint reports structural scene metadata, not the executable editor.
"""
from http.server import BaseHTTPRequestHandler
from functools import lru_cache
from urllib.request import Request, urlopen
import hashlib
import json
import struct
import zlib

SOURCE = (
    "https://drive.usercontent.google.com/download"
    "?id=1-JQP1fFwr_cXsl2rInd2HVnBcr0tTFi-&export=download&confirm=t"
)
ENTRIES = {
    "level5": {
        "offset": 9009230, "compressed": 83622, "size": 602692,
        "sha256": "45cffa740bd852b8ad9821eba5984f8c93b9016123a7ca1b0c645e36ef8d4800",
    },
    "Assembly-CSharp.dll": {
        "offset": 12372723, "compressed": 1061355, "size": 2785280,
        "sha256": "0d1524dabf0f8c67aa58ba1992d6ed858522b9094490ebbb00e5e8aeff7f40cd",
    },
}
OWNER_TYPES = ("scnEditor", "FloorMesh", "scrFloor")


def _range(start, end):
    req = Request(
        SOURCE,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "ADOFAI-PC-Editor-Scene-Inspector/1.0",
        },
    )
    with urlopen(req, timeout=25) as response:
        if response.status != 206:
            raise ValueError(f"Expected HTTP 206, received {response.status}")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError(f"Byte range truncated: {len(data)}")
    return data


@lru_cache(maxsize=2)
def original_entry(name):
    data = ENTRIES[name]
    offset = data["offset"]
    header = _range(offset, offset + 79)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Original PC ZIP local header mismatch")
    if struct.unpack_from("<H", header, 8)[0] != 8:
        raise ValueError("Unexpected ZIP entry compression method")
    file_name_len, extra_len = struct.unpack_from("<HH", header, 26)
    first = offset + 30 + file_name_len + extra_len
    packed = _range(first, first + data["compressed"] - 1)
    unpacked = zlib.decompress(packed, -15)
    if len(unpacked) != data["size"]:
        raise ValueError("PC source file length mismatch")
    if hashlib.sha256(unpacked).hexdigest() != data["sha256"]:
        raise ValueError("PC source SHA256 mismatch")
    return unpacked


@lru_cache(maxsize=1)
def scene_layout():
    import UnityPy

    environment = UnityPy.load(original_entry("level5"))
    names = {}
    transforms = {}
    for item in environment.objects:
        if item.type.name == "GameObject":
            obj = item.read()
            names[item.path_id] = {
                "name": obj.m_Name,
                "active": bool(obj.m_IsActive),
                "layer": int(obj.m_Layer),
            }
        elif item.type.name in ("RectTransform", "Transform"):
            obj = item.read()
            go = getattr(obj, "m_GameObject", None)
            if go is None or not go.path_id:
                continue
            t = {
                "transformId": item.path_id,
                "parentTransformId": obj.m_Father.path_id if obj.m_Father else 0,
                "type": item.type.name,
            }
            if item.type.name == "RectTransform":
                for key, field in (
                    ("position", "m_AnchoredPosition"),
                    ("size", "m_SizeDelta"),
                    ("anchorMin", "m_AnchorMin"),
                    ("anchorMax", "m_AnchorMax"),
                    ("pivot", "m_Pivot"),
                ):
                    value = getattr(obj, field)
                    t[key] = [round(value.x, 4), round(value.y, 4)]
            transforms[go.path_id] = t

    transform_to_go = {
        info["transformId"]: go_id for go_id, info in transforms.items()
    }
    children = {}
    for go_id, t in transforms.items():
        parent = transform_to_go.get(t["parentTransformId"])
        children.setdefault(parent, []).append(go_id)

    roots = sorted(children.get(None, []))
    result = []
    def walk(go_id, depth, parent_id, path):
        obj = names.get(go_id)
        if obj is None:
            return
        current = (path + "/" if path else "") + obj["name"]
        result.append({
            "id": go_id,
            "parentId": parent_id,
            "depth": depth,
            "path": current,
            **obj,
            **transforms.get(go_id, {}),
        })
        for child in children.get(go_id, []):
            if depth < 30:
                walk(child, depth + 1, go_id, current)
    for root in roots:
        walk(root, 0, None, "")

    return {
        "source": "PC Unity 6000.3.10f1 — level5 / scnEditor",
        "sceneSha256": ENTRIES["level5"]["sha256"],
        "serializedObjects": len(environment.objects),
        "gameObjects": len(names),
        "monoBehaviours": sum(
            o.type.name == "MonoBehaviour" for o in environment.objects
        ),
        "nodes": result,
        "limitations": (
            "Serialized hierarchy and rects only. "
            "Windows Unity editor scripts and game rendering are not running in browser."
        ),
    }


@lru_cache(maxsize=1)
def method_index():
    import dnfile
    import tempfile
    from pathlib import Path

    # dnfile's PE loader takes an ordinary path. File remains temporary
    # and is not included in web deployment artifacts.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "Assembly-CSharp.dll"
        path.write_bytes(original_entry("Assembly-CSharp.dll"))
        pe = dnfile.dnPE(str(path))
        index = {}
        for row in pe.net.mdtables.TypeDef.rows:
            name = str(row.TypeName)
            if name in OWNER_TYPES:
                index[name] = [str(method.row.Name) for method in row.MethodList]
    return {
        "assemblySha256": ENTRIES["Assembly-CSharp.dll"]["sha256"],
        "classMethods": index,
        "limitations": "Names only: C# / Unity method bodies are not executed by this site.",
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from urllib.parse import urlsplit, parse_qs
            query = parse_qs(urlsplit(self.path).query)
            if query.get("view", [""])[0] == "methods":
                data = method_index()
            else:
                data = scene_layout()
            body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "private, max-age=300")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as err:
            body = json.dumps({
                "error": "Original PC source could not be parsed",
                "detail": str(err),
            }).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
