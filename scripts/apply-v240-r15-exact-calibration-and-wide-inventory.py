#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply-v240-r15-exact-calibration-and-wide-inventory.py <V240CacheLoader.cpp>")

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")


def replace_exact(old: str, new: str, expected: int = 1) -> None:
    global s
    count = s.count(old)
    if count != expected:
        raise SystemExit(f"expected {expected} anchors, found {count}: {old[:180]!r}")
    s = s.replace(old, new)


# r15 consumes real-device r14 metadata instead of guessing names:
# - Persistence.GetInputOffset() -> float is static and pointer-backed.
# - Persistence.SetInputOffset(float) -> void is static and pointer-backed.
# - inputOffsetNotSet remains the game's own float sentinel.
# Re-enable only that exact self-fused calibration path. Tile-hit mutation remains disabled;
# widen the read-only scnEditor inventory so the actual v2.4 floor-selection helper is exposed.

replace_exact(
    'persistence.GetMethod("get_inputOffset", 0)',
    'persistence.GetMethod("GetInputOffset", 0)',
    expected=2,
)
replace_exact(
    'persistence.GetMethod("set_inputOffset", 1)',
    'persistence.GetMethod("SetInputOffset", 1)',
    expected=2,
)

replace_exact(
    'constexpr size_t kMetadataMaxMethodsPerClass = 12;',
    'constexpr size_t kMetadataMaxMethodsPerClass = 96;',
)
replace_exact(
    'constexpr size_t kMetadataMaxFieldsPerClass = 12;',
    'constexpr size_t kMetadataMaxFieldsPerClass = 64;',
)
replace_exact(
    'metadataInventoryRevision=1\\n',
    'metadataInventoryRevision=2\\n',
)
replace_exact(
    'metadataInventoryPolicy=bounded-read-only-no-invoke\\n',
    'metadataInventoryPolicy=wide-read-only-exact-name-followup\\n',
)

replace_exact(
    '        << "editorHitExecution=disabled-r14-metadata-inventory" << \'\\n\'\n',
    '        << "editorHitExecution=disabled-r15-wide-metadata-inventory" << \'\\n\'\n',
)
replace_exact(
    '        << "calibrationPolicy=neutralize-only-exact-game-sentinel-bnm-const-v2" << \'\\n\'\n',
    '        << "calibrationPolicy=exact-Persistence-GetInputOffset-SetInputOffset-sentinel-v3" << \'\\n\'\n',
)
replace_exact(
    '        << "calibrationExecution=disabled-r14-metadata-inventory" << \'\\n\'\n',
    '        << "calibrationExecution=enabled-r15-exact-persistence-self-fused" << \'\\n\'\n'
    '        << "calibrationResolvedGetterName=GetInputOffset" << \'\\n\'\n'
    '        << "calibrationResolvedSetterName=SetInputOffset" << \'\\n\'\n'
    '        << "calibrationExactMetadataProven=1" << \'\\n\'\n',
)

replace_exact(
    '    CollectV240MetadataInventory();\n',
    '    CollectV240MetadataInventory();\n'
    '    MaybeNeutralizeUnsetCalibration();\n',
)

for marker in (
    'metadataInventoryRevision=2',
    'metadataInventoryPolicy=wide-read-only-exact-name-followup',
    'constexpr size_t kMetadataMaxMethodsPerClass = 96;',
    'constexpr size_t kMetadataMaxFieldsPerClass = 64;',
    'persistence.GetMethod("GetInputOffset", 0)',
    'persistence.GetMethod("SetInputOffset", 1)',
    'calibrationExecution=enabled-r15-exact-persistence-self-fused',
    'calibrationResolvedGetterName=GetInputOffset',
    'calibrationResolvedSetterName=SetInputOffset',
    'calibrationExactMetadataProven=1',
    'editorHitExecution=disabled-r15-wide-metadata-inventory',
    '    MaybeNeutralizeUnsetCalibration();',
):
    if marker not in s:
        raise SystemExit(f"r15 marker missing after transform: {marker}")

if 'persistence.GetMethod("get_inputOffset", 0)' in s:
    raise SystemExit("legacy guessed getter survived r15")
if 'persistence.GetMethod("set_inputOffset", 1)' in s:
    raise SystemExit("legacy guessed setter survived r15")
if s.count('BasicHook(') != 5:
    raise SystemExit(f"r15 must not add hooks; expected five compiled hook sites, got {s.count('BasicHook(')}")

path.write_text(s, encoding="utf-8")
r16 = Path(__file__).with_name("apply-v240-r16-objects-at-mouse-probe.py")
if not r16.is_file():
    raise SystemExit(f"missing r16 overlay: {r16}")
__import__("subprocess").run([sys.executable, str(r16), str(path)], check=True)
