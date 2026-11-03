# Web ADOFAI Unified Studio v3 · single-workspace editor

The user requested one continuous screen: **original Web ADOFAI renderer on the right, editing + ZIP/audio/image controls on the left, and original online level search/play in a floating panel**, not two separate websites or a mode-switch tab bar.

## Implemented

- A permanent single editor workspace with tool sidebar and map iframe, no separate "search page" or "editing page".
- Search/play button opens the original `https://web-adofai.impl1113.dev/levels` within the same stage in a resizable/expandable overlay. Closing the overlay returns to the edited chart without throwing away the document.
- Opening `.zip`, `.adofai`, folder or creating a chart activates the documented original `/embed/chart` renderer **in the same stage**.
- Chart edits go to the original renderer using its public `web-adofai.chart` v1 `chart.load` message.
- Selecting a tile sends `chart.focus` with a `floorIndex` to the original chart renderer.
- Top bar gives immediate ZIP open/export/search; editing panel includes resource manager, events/settings, angle presets, undo and redo.
- Small viewport gets an independently collapsible tool drawer so the map stays visible.
- Preserves original ZIP music, backup chart, assets and non-selected files on export.
- In an **authorized same-origin** native game setup (`?native=1`) edits are committed to the engine's package storage without restarting the game *on every edit*, and the game page reloads when requested for play.

## Proven tests

The user's `Extra Stage（part2 - Doduy）.zip` loaded on desktop and emulated phone (390×840): 500 tiles / 1522 events / 43 decorations, audio blob playing and two chart candidates present. Changed tile 2 from 150° to 175°, added tile, undid, then exported ZIP and verified all original paths, unchanged OGG and unchanged backup; no page-level JavaScript exceptions. Desktop and phone had no horizontal overflow.

## Actual engine boundary

The upstream engine's source is not bundled or redistributed. The public remote `/embed/chart` accepts live chart content and selection/focus; **it cannot start actual custom gameplay**. The public inline search panel opens the real original website, where published levels are playable. Loading a ZIP and pressing the "custom play" button in a public build displays that limitation. Direct real custom gameplay needs an approved same-origin game runtime; GPU / timing validation is not complete. Do not represent the embed renderer as a fully functional original game.

Original source provenance is documented in [WEB_ADOFAI_ACTUAL_ENGINE_SOURCE_INVESTIGATION.md](../../../docs/WEB_ADOFAI_ACTUAL_ENGINE_SOURCE_INVESTIGATION.md). JSZip is MIT licensed. Other web game code remains hosted on its original author's site.
