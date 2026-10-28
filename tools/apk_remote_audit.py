#!/usr/bin/env python3
"""Read *only* APK container metadata by bounded HTTP Range requests.

No code from the APK is executed. No APK contents are uploaded or published.
The remote download must support honest HTTP 206 Content-Range replies.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import struct
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

MAX_TRANSFER = 16 * 1024 * 1024
TAIL_BYTES = 131072
MAX_MANIFEST = 4 * 1024 * 1024
EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


class AuditError(Exception):
    pass


class RangeFile:
    def __init__(self, url: str):
        self.url = url
        self.bytes_read = 0
        self.requests = 0
        self.total_size = None

    def _request(self, header: str):
        req = urllib.request.Request(
            self.url,
            headers={
                "Range": header,
                "User-Agent": "ADOFAI-static-zip-audit/1.0",
                "Accept-Encoding": "identity",
            },
        )
        try:
            response = urllib.request.urlopen(req, timeout=35)
        except urllib.error.HTTPError as exc:
            raise AuditError(f"range request HTTP {exc.code}") from exc
        with response:
            if response.status != 206:
                raise AuditError(
                    f"server ignored Range request (HTTP {response.status}); "
                    "refusing full APK download"
                )
            range_header = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", range_header)
            if not match:
                raise AuditError("missing/invalid Content-Range")
            start, end, total = map(int, match.groups())
            if end < start or end - start + 1 > MAX_TRANSFER:
                raise AuditError("response would exceed per-request size limit")
            if self.total_size is not None and self.total_size != total:
                raise AuditError("total size changed during audit")
            self.total_size = total
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise AuditError("incomplete or oversized range response")
        self.bytes_read += len(data)
        self.requests += 1
        return start, end, total, data

    def suffix(self, length: int):
        start, end, total, data = self._request(f"bytes=-{length}")
        if end + 1 != total:
            raise AuditError("suffix range didn't end at file EOF")
        return start, total, data

    def slice(self, offset: int, length: int) -> bytes:
        if length < 0 or length > MAX_TRANSFER or offset < 0:
            raise AuditError("invalid range length/offset")
        if length == 0:
            return b""
        start, end, total, data = self._request(f"bytes={offset}-{offset + length - 1}")
        if start != offset or end != offset + length - 1:
            raise AuditError("server supplied different bytes than requested")
        return data


class LocalFile:
    def __init__(self, path: Path):
        self.path = path
        self.total_size = path.stat().st_size
        self.bytes_read = 0
        self.requests = 0

    def suffix(self, length: int):
        offset = max(0, self.total_size - length)
        return offset, self.total_size, self.slice(offset, self.total_size - offset)

    def slice(self, offset: int, length: int):
        if length > MAX_TRANSFER or offset < 0:
            raise AuditError("invalid local read")
        with self.path.open("rb") as f:
            f.seek(offset)
            data = f.read(length)
        if len(data) != length:
            raise AuditError("unexpected EOF")
        self.bytes_read += length
        self.requests += 1
        return data


def inspect_zip(source):
    tail_start, total, tail = source.suffix(TAIL_BYTES)
    ix = tail.rfind(EOCD)
    if ix < 0 or ix + 22 > len(tail):
        raise AuditError("ZIP end-of-central-directory record not found")
    (
        disk, cd_disk, disk_entries, entries, cd_size,
        cd_offset, comment_length
    ) = struct.unpack_from("<HHHHIIH", tail, ix + 4)
    if disk or cd_disk or disk_entries != entries:
        raise AuditError("multi-disk ZIP not supported")
    if entries == 0xffff or cd_size == 0xffffffff or cd_offset == 0xffffffff:
        raise AuditError("ZIP64 central-directory metadata requires another parser")
    if ix + 22 + comment_length != len(tail):
        raise AuditError("ZIP footer/comment length mismatch")
    if cd_size > MAX_TRANSFER or cd_offset + cd_size > total:
        raise AuditError("central directory out of range or too large")
    directory = source.slice(cd_offset, cd_size)
    files = []
    pos = 0
    for _ in range(entries):
        if directory[pos:pos + 4] != CENTRAL or pos + 46 > len(directory):
            raise AuditError("invalid central-directory signature")
        method = struct.unpack_from("<H", directory, pos + 10)[0]
        compressed = struct.unpack_from("<I", directory, pos + 20)[0]
        original = struct.unpack_from("<I", directory, pos + 24)[0]
        n, extra, comment = struct.unpack_from("<HHH", directory, pos + 28)
        offset = struct.unpack_from("<I", directory, pos + 42)[0]
        flags = struct.unpack_from("<H", directory, pos + 8)[0]
        end = pos + 46 + n + extra + comment
        if end > len(directory):
            raise AuditError("truncated central-directory entry")
        raw_name = directory[pos + 46:pos + 46 + n]
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437", "replace")
        if compressed == 0xffffffff or original == 0xffffffff or offset == 0xffffffff:
            raise AuditError("ZIP64 file entry requires another parser")
        files.append({
            "name": name, "size": original, "compressed": compressed,
            "method": method, "offset": offset,
        })
        pos = end
    if pos != len(directory):
        raise AuditError("central-directory byte count mismatch")
    return total, files, cd_offset


def read_entry(source, entry):
    if entry["compressed"] > MAX_MANIFEST or entry["size"] > MAX_MANIFEST:
        raise AuditError("requested entry is larger than manifest cap")
    header = source.slice(entry["offset"], 30)
    if header[:4] != LOCAL:
        raise AuditError("invalid local ZIP header")
    n, e = struct.unpack_from("<HH", header, 26)
    start = entry["offset"] + 30 + n + e
    packed = source.slice(start, entry["compressed"])
    if entry["method"] == 0:
        payload = packed
    elif entry["method"] == 8:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(packed, MAX_MANIFEST + 1)
        if len(payload) > MAX_MANIFEST:
            raise AuditError("decompressed manifest exceeded cap")
        payload += decoder.flush(max(1, MAX_MANIFEST + 1 - len(payload)))
        if not decoder.eof:
            raise AuditError("incomplete deflate stream")
    else:
        raise AuditError("unsupported manifest compression method")
    if len(payload) != entry["size"] or len(payload) > MAX_MANIFEST:
        raise AuditError("entry uncompressed size mismatch")
    return payload


def _utf8len(b: bytes, off: int):
    x = b[off]
    if x & 0x80:
        return ((x & 0x7f) << 8) | b[off + 1], off + 2
    return x, off + 1


def _utf16len(b: bytes, off: int):
    x = struct.unpack_from("<H", b, off)[0]
    if x & 0x8000:
        return ((x & 0x7fff) << 16) | struct.unpack_from("<H", b, off + 2)[0], off + 4
    return x, off + 2


def manifest_info(blob: bytes):
    """Bounded Android binary XML attribute read; not a signature verifier."""
    if blob.startswith(b"<?xml") or blob.startswith(b"<manifest"):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(blob)
        android = "{http://schemas.android.com/apk/res/android}"
        return {
            "package": root.attrib.get("package"),
            "versionCode": root.attrib.get(android + "versionCode"),
            "versionName": root.attrib.get(android + "versionName"),
        }
    if len(blob) < 8 or struct.unpack_from("<H", blob)[0] != 0x0003:
        raise AuditError("unrecognized AndroidManifest.xml encoding")
    strings = []
    cursor = 8
    while cursor + 8 <= len(blob):
        typ, header, size = struct.unpack_from("<HHI", blob, cursor)
        if size < header or cursor + size > len(blob):
            raise AuditError("invalid Android binary XML chunk")
        if typ == 0x0001:  # RES_STRING_POOL_TYPE
            count, flags, data_start = struct.unpack_from("<III", blob, cursor + 8)[0], \
                struct.unpack_from("<I", blob, cursor + 16)[0], \
                struct.unpack_from("<I", blob, cursor + 20)[0]
            if count > 200000:
                raise AuditError("string pool has too many entries")
            for i in range(count):
                relative = struct.unpack_from("<I", blob, cursor + header + 4 * i)[0]
                at = cursor + data_start + relative
                if flags & 0x100:
                    _, at = _utf8len(blob, at)
                    n, at = _utf8len(blob, at)
                    value = blob[at:at + n].decode("utf-8", "replace")
                else:
                    n, at = _utf16len(blob, at)
                    value = blob[at:at + n * 2].decode("utf-16-le", "replace")
                strings.append(value)
        if typ == 0x0102 and strings:
            name_id = struct.unpack_from("<I", blob, cursor + 20)[0]
            if name_id >= len(strings) or strings[name_id] != "manifest":
                cursor += size
                continue
            attr_start, attr_size, count = struct.unpack_from("<HHH", blob, cursor + 24)
            if count > 5000 or attr_size < 20:
                raise AuditError("invalid manifest attribute count")
            props = {}
            for i in range(count):
                at = cursor + 16 + attr_start + i * attr_size
                if at + 20 > cursor + size:
                    raise AuditError("attribute exceeds binary XML chunk")
                name_idx, raw_idx = struct.unpack_from("<II", blob, at + 4)
                value_type, value_data = struct.unpack_from("<BI", blob, at + 15)
                if name_idx >= len(strings):
                    continue
                attr_name = strings[name_idx]
                if raw_idx < len(strings):
                    value = strings[raw_idx]
                elif value_type == 0x03 and value_data < len(strings):
                    value = strings[value_data]
                else:
                    value = value_data
                if attr_name in ("package", "versionCode", "versionName"):
                    props[attr_name] = value
            return props
        cursor += size
    raise AuditError("manifest tag not found in binary XML")


def audit(source):
    total, files, directory_offset = inspect_zip(source)
    selected = [
        e for e in files if
        e["name"] == "AndroidManifest.xml" or
        e["name"].startswith(("lib/", "META-INF/", "assets/bin/Data/", "assets/aa/")) or
        e["name"].startswith("classes") and e["name"].endswith(".dex") or
        e["name"].startswith(("res/", "resources.arsc"))
    ]
    manifest = next((e for e in files if e["name"] == "AndroidManifest.xml"), None)
    props = None
    error = None
    if manifest is not None:
        try:
            props = manifest_info(read_entry(source, manifest))
        except (AuditError, ValueError, IndexError, struct.error) as exc:
            error = f"{type(exc).__name__}: {exc}"
    names = {e["name"] for e in files}
    return {
        "apk_size": total,
        "file_count": len(files),
        "manifest": props,
        "manifest_error": error,
        "has_arm64_il2cpp": "lib/arm64-v8a/libil2cpp.so" in names,
        "has_unity_runtime": any("libunity.so" in n for n in names),
        "has_managed_metadata": any("global-metadata.dat" in n for n in names),
        "has_unity_assets": any(n.startswith("assets/bin/Data/") for n in names),
        "has_signature_meta_inf": any(n.upper().startswith("META-INF/") and n.upper().endswith((".RSA", ".DSA", ".EC")) for n in names),
        "warning": "Static ZIP/manifest inspection, NOT cryptographic signature or malware verification.",
        "selected_entries": [
            {k: e[k] for k in ("name", "size", "compressed", "method")}
            for e in selected[:250]
        ],
        "truncated_selected_entries": len(selected) > 250,
        "bytes_read": source.bytes_read,
        "range_requests": source.requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="local APK file")
    group.add_argument("--drive-id", help="Google Drive file ID (never printed)")
    group.add_argument("--url", help="HTTP URL supporting Range requests")
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    args = parser.parse_args()
    if args.file:
        src = LocalFile(args.file)
    else:
        url = args.url or (
            "https://drive.usercontent.google.com/download"
            f"?id={args.drive_id}&export=download&confirm=t"
        )
        src = RangeFile(url)
    try:
        result = audit(src)
        result["status"] = "ok"
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    except (AuditError, OSError, urllib.error.URLError) as exc:
        raise SystemExit(f"APK static audit unavailable: {exc}")


if __name__ == "__main__":
    main()
