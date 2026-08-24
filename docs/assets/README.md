# Eclipse — visual system

Seven hand-authored SVGs, plus one rasterised card. No build step, no external font, no remote reference — each
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
| `07_social_preview.svg` | 1280 × 640 | social preview source |
| `07_social_preview.png` | 1568 × 784 | the rasterised card GitHub is given |

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

## The social preview

GitHub's social preview accepts PNG, JPG or GIF — **not SVG** — at a recommended
1280 × 640. Both files are here:

| File | Role |
|---|---|
| `07_social_preview.svg` | the authored source, 1280 × 640 |
| `07_social_preview.png` | the rasterised card, 1568 × 784, exactly 2:1 — **this is the file to upload** |

**Uploading it is a manual step, and it has to be.** Settings → General → Social preview
→ Edit → *Upload an image…*, then pick `docs/assets/07_social_preview.png`.

That is not a preference. GitHub's uploader runs in the browser: the form records the
image's metadata immediately, but the binary itself is sent separately by JavaScript that
only runs on a genuine, user-initiated file selection. Setting the file input any other
way produces a repository whose `og:image` points at an asset that was never stored —
every social card renders broken, which is worse than the default GitHub generates for
you. If you try to automate it, check `og:image` afterwards and confirm the URL actually
returns the image.

**Re-rasterising after an edit to the SVG.** Open the SVG in a browser at a 2:1 viewport
with no margin and screenshot it — that renders the real file, gradients and all. Pillow
and cairosvg are not installed in this repository's interpreter, so
`make_social_preview.py` (which redraws the card rather than rasterising the SVG) is a
fallback, not the reference. Whatever you use, keep the aspect at exactly 2:1 and stay at
or above 1280 × 640.

## On branding

The public identity is **SΞNSE / ECLIPSE**, drawn as an occulted disc with a thin corona
ring. It is original work: no third-party character, logo or copyrighted artwork appears
in any of these files, and none should be added. A public repository's branding is one of
the few things on it that is trivially checkable for infringement.
