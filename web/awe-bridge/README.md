# ADOFAI: existing editor handoff

The browser ZIP bridge at `index.html` helps people use the existing [AWE web editor](https://awe.haeengin.com/) with a local `.zip` containing a chart and audio.

This is **not** a copy, fork or port of HaeengIn's or khm041213's code. Both repositories have no explicit GitHub license in their repository metadata as inspected 2026-11-03. Their maintained hosted applications are linked directly, with attribution.

- [AWE](https://awe.haeengin.com/) — externally hosted functional web editor by Ssimille and contributors, using Three.js.
- [Origin project](https://khm041213.github.io/Adofai-Web-Editor/) — earlier mobile web editor.
- [Web ADOFAI levels](https://web-adofai.impl1113.dev/levels) — level catalog/search, not a level editor.

**ZIP handling:** Runs solely in the browser using JSZip, an MIT-licensed dependency. Loads the user's ZIP into browser memory, selects its non-backup `.adofai` plus the `settings.songFilename` audio, and exposes download links for both files. ZIP data is **not uploaded** to a server. The user selects the two local files together in AWE using its Load Level control. Cross-origin browser security prevents this helper from silently inserting files into AWE's file picker.

The former homebrew editor remains under `web/original-editor-port` in this Git branch for historical reference. It is not the recommended product. This page does not contain original proprietary Unity game assets or claim Unity engine parity.
