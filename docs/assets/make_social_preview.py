"""Render the Eclipse GitHub social preview card.

Writes ``docs/assets/07_social_preview.png`` at 1280x640 — GitHub's recommended
size for *Settings -> General -> Social preview*, which accepts PNG/JPG/GIF but
not SVG.

    pip install Pillow
    python docs/assets/make_social_preview.py

The authored source is ``07_social_preview.svg``; this script reproduces it with
Pillow so the export does not depend on an SVG rasteriser. Keep the two in step
by eye — the SVG is the reference.

Mirrors ``web/tools/make_og.py``, and is subject to the same publication policy:
the card deliberately carries no number of any kind. See ``web/README.md``.

Pillow is not a dependency of this repository. If it is missing, use the browser
route documented in ``docs/assets/README.md`` instead.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - documented manual step
    sys.exit(
        "Pillow is not installed.\n"
        "  pip install Pillow\n"
        "or use the browser route in docs/assets/README.md."
    )

W, H = 1280, 640

BG = (7, 9, 13)
INK = (232, 236, 242)
DIM = (152, 162, 179)
FAINT = (92, 102, 117)
SEP = (46, 53, 66)
BLUE = (77, 124, 255)
CYAN = (34, 211, 238)
VIOLET = (167, 139, 250)
GOLD = (233, 162, 59)
CREAM = (242, 228, 200)

OUT = Path(__file__).resolve().parent / "07_social_preview.png"
FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
]


def font(names: list[str], size: int):
    """First installed face wins; fall back to Pillow's bitmap font."""
    for name in names:
        for directory in FONT_DIRS:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def corona_colour(t: float) -> tuple:
    """The four-stop gradient used by the SVG: blue -> violet -> gold -> cream."""
    if t < 0.42:
        return lerp(BLUE, VIOLET, t / 0.42)
    if t < 0.78:
        return lerp(VIOLET, GOLD, (t - 0.42) / 0.36)
    return lerp(GOLD, CREAM, (t - 0.78) / 0.22)


def main() -> None:
    # supersample so the thin ring and hairlines stay clean
    s = 2
    img = Image.new("RGB", (W * s, H * s), BG)
    d = ImageDraw.Draw(img)

    def line(x1, y1, x2, y2, fill, width=1):
        d.line((x1 * s, y1 * s, x2 * s, y2 * s), fill=fill, width=max(1, round(width * s)))

    # ---- rail -------------------------------------------------------------
    line(64, 0, 64, H, (30, 33, 39), 1)
    d.ellipse(
        ((64 - 3.4) * s, (320 - 3.4) * s, (64 + 3.4) * s, (320 + 3.4) * s), fill=BLUE
    )
    d.ellipse(
        ((64 - 9) * s, (320 - 9) * s, (64 + 9) * s, (320 + 9) * s),
        outline=(40, 55, 96),
        width=max(1, s),
    )

    # ---- eclipse ----------------------------------------------------------
    cx, cy, r = 1010, 320, 166
    d.ellipse(
        ((cx - 196) * s, (cy - 196) * s, (cx + 196) * s, (cy + 196) * s),
        outline=(24, 27, 33),
        width=max(1, s),
    )
    # corona ring, drawn arc-by-arc so the gradient reads
    for i in range(720):
        a0 = i * 0.5
        col = corona_colour(((a0 + 135) % 360) / 360.0)
        d.arc(
            ((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
            start=a0,
            end=a0 + 0.9,
            fill=col,
            width=max(2, round(3 * s)),
        )
    # occulting disc
    d.ellipse(
        ((cx - 149) * s, (cy - 149) * s, (cx + 149) * s, (cy + 149) * s), fill=(10, 13, 18)
    )
    # the bright limb, and the diamond
    d.arc(
        ((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
        start=250,
        end=305,
        fill=(248, 250, 253),
        width=max(2, round(4.4 * s)),
    )
    dx = cx + r * math.cos(math.radians(-55))
    dy = cy + r * math.sin(math.radians(-55))
    d.ellipse(((dx - 5.6) * s, (dy - 5.6) * s, (dx + 5.6) * s, (dy + 5.6) * s), fill=INK)

    # ---- ladder: abstract, carries no value -------------------------------
    rungs = [
        (812, 212, (24, 27, 33)), (798, 238, (31, 35, 42)),
        (784, 264, (40, 45, 54)), (770, 290, (52, 58, 69)),
    ]
    for x1, y, col in rungs:
        line(x1, y, 852, y, col, 2.4)
        line(x1, 632 - y + 26, 852, 632 - y + 26, col, 2.4)
    line(756, 316, 852, 316, (24, 122, 138), 2.4)
    line(756, 342, 852, 342, (52, 76, 145), 2.4)

    # ---- wordmark ---------------------------------------------------------
    mono_small = font(["consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf"], 15)
    mono_body = font(["consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf"], 13)
    body = font(["segoeui.ttf", "Helvetica.ttc", "DejaVuSans.ttf"], 24)
    display = font(["bahnschrift.ttf", "arialnb.ttf", "DejaVuSans-Bold.ttf"], 148)
    try:  # Bahnschrift is variable — pick the bold condensed instance the site uses
        display.set_variation_by_axes([700, 82])
    except Exception:
        pass

    def text(x, y, s_, f, fill, tracking=0.0):
        if not tracking:
            d.text((x * s, y * s), s_, font=f, fill=fill)
            return
        cur = x
        for ch in s_:
            d.text((cur * s, y * s), ch, font=f, fill=fill)
            cur += d.textlength(ch, font=f) / s + tracking

    text(104, 206, "SΞNSE", mono_small, FAINT, tracking=9.5)
    text(98, 232, "ECLIPSE", display, INK, tracking=7)
    line(102, 396, 700, 396, (30, 33, 39), 1)
    text(104, 414, "Mechanism-first market microstructure", body, DIM)
    text(104, 446, "research and execution.", body, DIM)

    verbs = ["OBSERVE", "MEASURE", "FALSIFY", "REPLICATE", "EXECUTE"]
    x = 104
    for i, v in enumerate(verbs):
        text(x, 510, v, mono_body, (110, 120, 137), tracking=5.2)
        x += (d.textlength(v, font=mono_body) / s) + 5.2 * len(v) + 6
        if i < len(verbs) - 1:
            text(x, 510, "·", mono_body, SEP)
            x += 26

    img.resize((W, H), Image.LANCZOS).save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT}  {W}x{H}  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
