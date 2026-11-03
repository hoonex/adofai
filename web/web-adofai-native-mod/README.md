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

## 2026-11-03: Manual play + renderer compatibility diagnosis

The earlier Play claim was too strong. Tested the actual upstream Web ADOFAI game and this same-origin integration with desktop Chromium software WebGPU in parallel:

- Both engines displayed `WebGPU 장치 연결이 끊겨 렌더러를 복구하고 있습니다`.
- Pressing the upstream `Play` button changed it to `Preparing`, but renderer errors caused a return to `Play`. Sending Space and touch during this state did **not** establish successful timing judgement.
- A WebGPU `GPUDevice.lost` event returned `reason=unknown` and `message=A valid external Instance reference no longer exists.` on the tested software GPU.
- Source in the actual production `button.component-BR8Evgkc.js` explicitly rejects a non-WebGPU backend with `WebGL fallback is disabled.` There is **no supported WebGL fallback** currently in that engine.
- This test environment does not prove the same failure cause on every Android device; the browser's GPU adapter, device and driver differ.

### Behavior changed

- The button is now `▶ 직접 플레이`, not "autoplay": no simulated beats or automatic hits. It requests native game mode and leaves timing input to human touch/keyboard.
- If the original game reports a WebGPU failure, the addon exposes that blocker visibly rather than pretending it can play. Pressing Play in that state does not discard pending chart edits.
- If an edited map is saved before entering game mode on a compatible renderer, the request to enter manual mode survives the reload without injecting hit inputs.
- Imported ZIPs open with the editor panel visible and PC-like `FILE / EDIT / LEVEL / EVENTS / DECORATIONS / SETTINGS` navigation. This **is not the original Unity PC scnEditor UI**. The true PC editor resides in the separately extracted Unity scene `level5`.
- Decoration JSON can now be selected, edited, added or deleted with undo and ZIP export.
- Real-browser desktop and mobile smoke tests: 500-tile ZIP imported, four files kept, six editor menu items available, tile and decoration mutations visible, renderer failure warning correctly shown, zero JS page errors.

**Unsolved:** native manual-gameplay correctness and full PC Unity editor UI remain unverified/unported. A genuine renderer compatibility implementation (or a working supported WebGPU adapter) is necessary, not merely another button.
