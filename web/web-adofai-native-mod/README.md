# Web ADOFAI Native Custom Studio

**Actual site:** https://webadofai-native-studio.agfvrd.workers.dev/levels

**Previous URL:** https://adofai-pc-editor-port.vercel.app/ redirects to the same native app.

## Architecture

This is **one actual Web ADOFAI application, not an embedded iframe or a second visual approximation**.

- A Cloudflare Worker serves the **original Web ADOFAI site's real HTML, JS, CSS, game scenes, assets and routes** from `https://web-adofai.impl1113.dev`, under one browser origin.
- The same-origin Worker also forwards the original site's public TUF API reads under `/__tuf_api`, rewriting the known hard-coded API URL in the original browser JS to that same-origin path. This addresses the browser CORS limitation that occurs when running the app at a separate origin.
- The worker adds **two scripts** to the original application's HTML: the MIT-licensed JSZip and `native-addon.js`. No iframe is created.
- Source builds use the owner's original runtime (as publicly deployed), which the user stated they were authorized to use; this repo stores **our addon code**, not a fork of the original proprietary game sources.

## End-user workflow

1. Open the site. The original TUF level search, original renderer and original native game play UI are present.
2. Press `📂 ZIP 열기` and choose a ZIP with an `.adofai` chart and music, or select a standalone `.adofai` alongside its audio.
3. The addon parses and validates the archive locally and registers **every included file** in the **same IndexedDB cache the original Web ADOFAI game itself uses**: database `adofai-web-editor-level-packages`, stores `packages` and `fileBlobs`.
4. The browser moves directly to the native original game's `/levels/cache/:id` route; original tile renderer, game state and built-in Play button are used.
5. Use `✎ 편집` to change floor directions, add/remove floors, undo/redo, edit events, BPM, offset and full JSON.
6. `⟳ 원본 게임에 적용` updates the original game cache and reloads that **same original engine**. `▶ 플레이` triggers the native game Play control when available.
7. `💾 ZIP 저장` exports the edited chart with the exact original audio/image/backup file contents preserved.

## Test evidence

Synthetic `Extra Stage`-shape 500-floor ZIP with 1,522 events, 43 decorations, 3,078,653 bytes of music and a backup `.adofai`, tested in real desktop and mobile Chromium:

- no iframe elements;
- 500 floors / 1,522 events / 43 decorations / 3 files registered;
- floor 2 changed to 175°, saved, navigated through the native original route, and persisted;
- BPM changed to 197, floor added, undo restored 500 floors;
- re-export ZIP contained the same music Blob and byte-for-byte identical backup chart;
- zero JavaScript page exceptions.

Native level search also loaded actual current TUF results after same-origin API forwarding. The original Play button appeared in the native custom game route at least in the mobile test browser. **Cloud software GPU occasionally reported a lost WebGPU device**, so timing judgement and continuous audio gameplay were not validated end-to-end. In those cases, the addon reports the underlying original engine's result rather than pretending it completed actual play. Android device GPU support varies.

## Notes

Imported user ZIPs are parsed locally in the browser and placed into browser IndexedDB; our addon does not intentionally upload them to its backend. The original game may make its own external requests under its own policies. Media file names are validated to prevent directory traversal. An imported ZIP's main chart and all its other files remain distinct in the storage and in exports.

The Worker proxy and addon use the user's stated authorization for the original engine. Public TUF API and engine availability are external dependencies and may change. Source assets remain attributed to their original producers.
