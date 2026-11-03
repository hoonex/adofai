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

## 2026-11-03: v6 UI and manual-input follow-up

### Exact tested improvement

- Screenshot-informed PC-style **Song Settings** pane on the *left* side of the original Web ADOFAI game. Fields include Filename, BPM, Volume, Offset, Pitch, Hitsound, Sound volume, Countdown ticks, Song title and Artist; matching settings fields are written to actual `.adofai` JSON.
- An audio preview player uses the existing `songFilename` and corresponding ZIP audio Blob. The original audio bytes are not altered by settings edits.
- Additional UI includes `FILE / EDIT / LEVEL / EVENTS / DECORATIONS / SETTINGS` and native file/floor/event/decoration data editing.
- The addon **no longer invokes the original Play button via JavaScript `.click()`**. Synthetic clicks are unreliable for browser audio user-activation and are not a substitute for a real human input gesture. The helper now highlights the game's **actual native Play button** and asks the human to tap/click it; keyboard Space or touch then belongs to the original engine.
- Dirty chart saves persist first and on reload present the native Play button for a **separate direct human gesture**. The addon does not synthesize rhythm hits.
- WebGPU failure banner has an **experimental low-power adapter retry**, which sets `adofai_gpu=low` browser cookie. The same-origin Worker then changes the actual renderer module's two `powerPreference:"high-performance"` options to `powerPreference:"low-power"`. The original renderer and gameplay engine remain otherwise intact.

### Verification

Real desktop (1250×850) and emulated mobile (390×840) Chromium: same-origin original engine remains iframe-free; a 500-tile ZIP imported; settings (pitch 100→115%, hitsoundVolume 70→64%) edited; four original settings including BPM and Song filename read correctly; ZIP exported with changes; music and backup chart preserved byte-for-byte; zero JavaScript exceptions in editor test.

The Worker correctly returns the upstream renderer with two high-performance preferences by default and two low-power preferences with the compatibility cookie, without changing the rest of the bundle. Browser-level low-power test passes route initialization, but **the cloud software GPU still loses its WebGPU device** and native gameplay returns to Editor/Play state. This is *not a verified fix for actual gameplay on user Android hardware*.

The upstream original game's renderer is a custom WebGPU-specific renderer, not a conventional Three.js WebGL renderer with an easy backend fallback. Its code explicitly states `WebGL fallback is disabled.` A usable WebGL renderer would require a significant new compatibility backend and cannot be achieved by renaming one option. No unsupported claim of a working fallback is made.

**Current status: editor ZIP/asset/settings round-trip and real native game screen wiring work. Verified complete manual rhythm-judgement gameplay is still pending.** The full original Unity `scnEditor` scene from the PC game has also not been rebuilt for the web; the new Song Settings panel is a UI adaptation, not the original Unity UI.

### Original engine GPU profile matrix (cloud, 2026-11-03)

Compared the unmodified original `https://web-adofai.impl1113.dev/levels/17647` in real Chromium under five GPU startup profiles: SwiftShader WebGPU only, ANGLE SwiftShader, ANGLE SwiftShader+Vulkan, ANGLE Vulkan, and ANGLE GL. **All five produced the same original WebGPU device-loss warning and prevented a verified complete play session.** This rules out a simple browser GPU flag as a verified fix in our cloud software renderer. The device-loss issue exists on upstream itself in the tested environment, not exclusively our one-origin wrapper.

The deployed compatibility button changes the original renderer's adapter preference, but cloud testing with low-power preference also failed. On a different real GPU it may change adapter selection, but there is currently **no evidence it repairs the user's specific Android device**. Native Play correctness must be validated on hardware with a stable WebGPU adapter and tested rhythm inputs. The game's WebGPU-only custom renderer has no implemented WebGL backend.

Most importantly, the addon does **not** synthesize `button.click()`, `KeyboardEvent`, or hit input. It highlights the native Play control so **a real human gesture** triggers gameplay activation and browser audio policy. No autoplay is provided.
