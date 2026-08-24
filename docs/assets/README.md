# Eclipse — visual system

Seven hand-authored SVGs. No build step, no external font, no remote reference — each
file renders identically from disk, from GitHub, and from any static host.

## Files

| File | Size | Used by |
|---|---|---|
| `01_eclipse_hero.svg` | 1200 × 320 | `README.md` hero |
| `02_system_architecture.svg` | 1200 × 676 | `README.md`, `docs/public/ARCHITECTURE_OVERVIEW.md` |
| `03_research_lifecycle.svg` | 1200 × 768 | `README.md` |
| `04_safety_stack.svg` | 1200 × 576 | `README.md`, `docs/public/ARCHITECTURE_OVERVIEW.md` |
| `05_microstructure_concept.svg` | 1200 × 580 | `README.md` (inside a `<details>`) |
| `06_governance.svg` | 1200 × 540 | `docs/public/RESEARCH_METHOD.md` |
| `07_social_preview.svg` | 1280 × 640 | social preview source — see below |

## Tokens

Identical to `web/assets/css/eclipse.css`, so the repository and the site read as one
surface.

```
ground     #07090D   bg          #0C0F14  raised      #11151C  panel
hairline   rgba(255,255,255,.09)          stronger    rgba(255,255,255,.16)
ink        #E8ECF2   dim         #98A2B3  faint       #5C6675
blue       #4D7CFF   system      cyan     #22D3EE     data / research
violet     #A78BFA   agents / intelligence
radius     2px
display    Bahnschrift → DIN Alternate → Arial Narrow → system-ui
mono       ui-monospace → Cascadia Mono → Consolas → SF Mono → Menlo
```

## Colour rules these files obey

The site's content policy reserves **green, amber and red** for health — healthy,
warning, blocked — and forbids using any of them to label a component, because nothing in
Eclipse is running for a public reader. These assets keep that rule:

- No labelled node in any diagram carries a health colour.
- Blue, cyan and violet appear only as **category accents** — a plane, a side of the
  book, a chain — never as a state.
- `05_microstructure_concept.svg` states in its own footer that colour marks a *side of
  the book, not a state*, because a bid/ask diagram is exactly where a reader would
  otherwise read green and red into it.
- The one warm tone in the palette (`#E9A23B`) appears **only** inside the eclipse
  corona gradient in `01` and `07`. It labels nothing, carries no text, and touches no
  component. It is the sun behind the disc.

## Rules for adding or editing an asset

1. **Presentation attributes, not `<style>` blocks.** GitHub serves these through an
   image proxy; inline attributes are the reliable path.
2. **No `--` inside an XML comment.** It is not valid XML, and the file will fail to
   parse. Three of these files hit that on the first pass.
3. **Nothing outside the viewBox.** Text overflow is the failure that renders fine
   locally and clips on GitHub.
4. **No numbers.** No bps, no rate, no count, no threshold, no horizon. A figure that
   needs a number to make its point belongs in a private document.
5. **`role="img"` plus `<title>` and `<desc>`** on every file. These are read by screen
   readers and by anyone whose images fail to load.
6. **Re-run the checker.** It parses every file, resolves every gradient and marker
   reference, checks geometry against the viewBox, estimates text width per font face,
   and refuses any external reference:

```bash
python docs/maintenance/tools/check_public_docs.py
```

## Regenerating the social preview

GitHub's social preview accepts PNG, JPG or GIF — **not SVG** — at a recommended
1280 × 640, uploaded manually under *Settings → General → Social preview*.

`07_social_preview.svg` is the authored source at exactly that size. To rasterise it,
in order of least effort:

1. **Browser** — open the SVG, set the window to 1280 × 640, screenshot. Fonts render
   exactly as a viewer would see them.
2. **`make_social_preview.py`** — draws the card directly with Pillow, mirroring
   `web/tools/make_og.py`:

   ```bash
   pip install Pillow
   python docs/assets/make_social_preview.py
   ```

   Neither Pillow nor cairosvg is installed in this repository's interpreter, which is
   why this is a manual step rather than part of a build.
3. **Any SVG converter** — `rsvg-convert`, Inkscape, `cairosvg` — at 1280 × 640.

**Interim option needing no work:** `web/assets/og-eclipse.png` already exists at
1200 × 630, is Eclipse-branded, and carries no numbers under the same policy. GitHub
accepts it; it is 80 px narrower than the recommendation and letterboxes slightly.

## On branding

The public identity is **SΞNSE / ECLIPSE**, drawn as an occulted disc with a thin corona
ring. It is original work: no third-party character, logo or copyrighted artwork appears
in any of these files, and none should be added. A public repository's branding is one of
the few things on it that is trivially checkable for infringement.
