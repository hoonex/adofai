# Web ADOFAI Live Engine Integration (2026-11-03)

This project integrates, but does not copy, the third-party **Web ADOFAI** site from https://web-adofai.impl1113.dev.

- **Search & live play**: displays the actual vendor's `/levels` via a cross-origin iframe when its embedding policy permits. "Open original" is provided for frame-blocking browsers.
- **Editor + actual Web ADOFAI rendering**: displays `/embed/chart?parentOrigin=<origin>` and uses the publicly documented **`web-adofai.chart`, version 1** `postMessage` protocol to transmit chart text and receive floor-selection events. The public contract appears in [KGH1113/TUFReplay](https://github.com/KGH1113/TUFReplay/blob/main/web/src/shared/clients/chart-embed-client.ts).
- **Local editable charts**: load `.adofai` or ZIP; edit angleData tile directions and list of events, apply settings, add/remove tiles, undo/redo and export; ZIP export preserves other entries including original music and backup.
- The imported files, including music and images, remain in the browser. No local song or archive data is uploaded to the site backend.
- Exact site source files and game engine code are **not** bundled, nor reverse-engineered server private APIs or copyrighted game assets redistributed.

**Limitations:** There is currently no publicly documented command for starting *custom local level gameplay* in the `/embed/chart` protocol. Its `chart.load` command is for visualization; **do not label it playable**. The official `/levels` route handles real online level search and play separately. The upstream iframe may be blocked depending on third-party Content-Security-Policy and frame restrictions; test on the final deployed domain rather than assume success.

**Source evidence:** `KGH1113/TUFReplay/web/src/hooks/activity/use-embedded-chart.ts` generates `parentOrigin` and loads an external `/embed/chart`. `web/src/shared/clients/chart-embed-client.ts` defines `chart.ready`, `chart.load`, `chart.loaded`, `chart.error`, `chart.floor-selected`, `chart.focus` and related events.

## Verification

Tested against user's supplied `Extra Stage（part2 - Doduy）.zip`: 499 angleData entries = 500 tiles, 1,522 actions, 43 decorations; edited a tile angle, inserted a tile, undo and redo, exported ZIP and re-opened it to confirm exact original 3,078,653-byte `audio.ogg` retained. JavaScript page errors: none. This test does **not** verify that upstream remote iframe accepts every origin or that the custom level can be rhythm-played.
