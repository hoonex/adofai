# Web ADOFAI Native Studio v2 — import & editing UX

This is the native-studio **UI / file-management source**, not a copy of Web ADOFAI's proprietary game code.

## What changed

- Local ZIP (`.zip`) importer with CRC checks, zip-bomb thresholds (250MB archive/expanded data, 100MB single assets, 2,500 entries), filename validation, and backup/non-backup chart selection.
- Native `.adofai` and whole-folder imports (webkitdirectory on supported browsers).
- Audio auto-detection from `settings.songFilename`, audio preview, images, asset list, add/remove/choose resources, background-image selection.
- Better mobile drawer with expanded controls, quick song/BPM/offset/author settings, 9 angle presets, event presets, raw JSON editing, undo/redo and save.
- Editor save: `.adofai` or reconstituted `.zip`; preserves unrelated chart attributes and all unmodified audio/backup/media bytes. Automatic sync is debounced to avoid reloading the entire WebGPU scene on every keypress.
- Connected external chart viewer via **documented** `web-adofai.chart` v1 `chart.load` postMessage; the upstream rendered chart is distinct from playable in-game simulation.
- The original engine's local `adofai-web-editor-level-packages` IndexedDB data format is supported **only with explicit `?native=1` and an authorized same-origin engine service**.

## Validated

Tested the user-provided `Extra Stage（part2 - Doduy）.zip` in isolated Chromium:

- 500 tiles, 1,522 actions, 43 decorations, 3 user files
- Archive recognized and main chart (not backup) chosen
- Audio preview attached to original OGG
- BPM, offset, angle, tile insertion, events and undo/redo work
- Re-export ZIP includes unchanged 3,078,653-byte audio and unchanged 308,551-byte backup chart
- Mobile test: 390×844 viewport with collapsible editing drawer, no page JS errors

## Actual playable native engine

When deployed as an independent static site, `studio-v2` uses **the upstream Web ADOFAI site's external iframe** for searching/playing their existing levels and their chart-only renderer for local chart editing. **It cannot silently register a locally edited ZIP into the upstream site's origin-isolated cache or start playable custom levels by postMessage.**

When explicitly served from an **authorized same-origin Web ADOFAI runtime**, adding `?native=1` enables local package-cache registration, the real `/levels/cache/:id` game page and native Play button. This integration was tested in an isolated same-origin proxy only. The actual gameplay judgment remains unverified on the test GPU; an isolated test with no WebGPU interruption would be needed before claiming full parity.

No third-party engine JavaScript is included in this repository. Upstream software is hosted by its own authors and may be subject to their terms.

## Files

- `index.html` — responsive studio shell / panes / resource UI
- `studio.js` — chart parsing, editing commands, schema validation, package-cache bridge, outside chart embed
- `studio-extras.js` — complete browser-side ZIP, folder, media and UX management
- `jszip.min.js` — third-party JSZip v3.10.1, MIT license

All file imports are local in browser memory and are not sent to our app server. **External chart rendering sends chart JSON to the upstream iframe** through postMessage; the upstream website's separate privacy policy applies.
