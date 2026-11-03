# Native Web ADOFAI Studio — original engine compatibility experiment

**Status (2026-11-03):** Native game engine integration **verified inside an isolated, same-origin development test**, **not deployed as a public mirror**. The original Web ADOFAI code is owned by its creators, and their permission to republish/proxy the entire application has not been confirmed.

## What was built

- `studio.html` — responsive single-page toolbar, native level-search/game frame, tile and event editing controls.
- `studio.js` — loads an `.adofai`/ZIP, preserves sibling audio/backup/decorations, edits tile directions, adds/removes tiles, edits events/settings, maintains undo/redo, exports ZIP.
- Registers chart and audio **locally** using the original game's IndexedDB cache format, which was established by inspecting its deployed `app-router.component-BF3hr6k2.js` bundle and by importing a test folder through the official `Local Levels` UI.
- Navigates the real game UI to `/levels/cache/local%3A...` on the **same origin**, displaying the original game, its native settings, and its Play button. No homegrown SVG playfield or homemade scoring engine is used.

### Original level cache format verified

- DB: `adofai-web-editor-level-packages`, version `1`.
- stores: `packages` (keyPath `packageId`) and `fileBlobs` (key/value).
- records contain `packageId`, `source:'local'`, `displayName`, `targetLevelPath`, `levelBaseDir`, `files`, `fileCount`, `totalBytes`, `createdAt`, `updatedAt`.
- each `files` item has `relativePath`, `name`, `type`, `size`, `sourcePath`, `storage:'indexeddb'`, `blobKey`. The key indexes the actual `Blob` in `fileBlobs`.
- other storage `opfs` can be used by the official site's own folder importer. This compatibility layer deliberately uses the documented-by-inspection IndexedDB fallback.

## Verified test evidence

In an isolated cloud browser (Chromium with experimental SwiftShader WebGPU), an HTTP same-origin proxy **only for development tests** fetched the original site's publicly served app modules on demand. It was not made public.

1. The original search UI displayed **8,428 live search results** in the proxied origin.
2. Uploaded a synthetic `TestLevel.zip` with `TestLevel/level.adofai` and `TestLevel/song.wav`. The editor created a local record and opened the original `/levels/cache/:id` game route.
3. The actual game's settings button reported `TestLevel opened from cache with song.wav`, and native **Play** and timeline buttons were present.
4. Editing tile 2 from **90° → 175°** updated the active chart and the native game's cache. The iframe reloaded that same game route to read updated content.
5. **ZIP export preserved the original 320,044-byte WAV** and wrote the changed `.adofai` with angle 175°.
6. The original Play button was programmatically clicked. Its UI entered `Preparing`, but browser WebGPU reported `GPUCanvasContext: context is not configured` and the renderer disconnected. Therefore **successful rhythm input and complete playback were not verified** in this cloud environment.

## What is not yet delivered

- No authorization from the third-party operator to mirror or publicly redistribute their proprietary deploy bundle.
- The public deployment service declined the proposed full-site proxy. This request was not circumvented.
- Since `web-adofai.impl1113.dev` and any separate domain have different origins, this editor cannot insert records into the official site's IndexedDB from another origin. Thus deploying just `studio.html` without same-origin integration will **not** provide the advertised one-click playable custom levels.
- No independent fork/rebuild of the original game engine from source, as the original `.ts/.tsx` project and license remain unavailable.

## Ownership and privacy

The source here is the independent browser editing and IndexedDB compatibility layer, **not the Web ADOFAI source**. The real site is [Web ADOFAI](https://web-adofai.impl1113.dev/levels). No copyrighted game engine binaries or the user's music archive are included in these files. ZIP chart/music assets are stored in the browser's own origin storage during the test workflow. If future hosting is authorized, the original site and its original creators should be credited and the app's user data handling disclosed.

## Source investigation

See [actual engine investigation](../../docs/WEB_ADOFAI_ACTUAL_ENGINE_SOURCE_INVESTIGATION.md).
