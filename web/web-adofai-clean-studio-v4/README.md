# Web ADOFAI · Clean Studio v4

A simplified **single-frame** UI over the existing Web ADOFAI site, with an optional editor drawer.

- Default: the actual upstream `https://web-adofai.impl1113.dev/levels` fills the viewport; no second nested search panel.
- `📂 열기`: local ZIP and `.adofai` import. Existing sound, images, backup charts and other files are kept in the browser archive.
- `✎ 편집`: collapsible drawer for tile angles, BPM, events, files and ZIP export.
- Editing uses the upstream `/embed/chart` visualization API with `web-adofai.chart` v1. The same viewport is reused; no duplicate iframe.
- `🎮 레벨`: return to the original online level search and real published-level playback.
- Returning to `✎ 편집` restores the currently opened chart without re-import.
- Desktop and mobile validated on the deployed URL with a 500-tile ZIP test; JavaScript exceptions: 0.

**Important:** Original Web ADOFAI engine code is not bundled/rehosted here; the iframe loads from the original site. Live chart visualization is **not** native custom-level gameplay. The `/embed/chart` protocol does not expose a custom play command, and the original game's local IndexedDB is cross-origin. Therefore real immediate gameplay of edited custom maps is not claimed or supported in the public build.

The project contains no copied original proprietary game assets or code. JSZip 3.10.1 is MIT licensed.
