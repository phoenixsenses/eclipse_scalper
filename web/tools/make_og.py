"""Render the Eclipse social preview card.

Writes web/assets/og-eclipse.png (1200x630). Re-run after a wording change:

    python web/tools/make_og.py

Deliberately carries no numbers — the card is subject to the same publication
policy as the site (see web/README.md).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 630
BG = (7, 9, 13)
INK = (232, 236, 242)
DIM = (152, 162, 179)
FAINT = (92, 102, 117)
BLUE = (77, 124, 255)
CYAN = (34, 211, 238)

FONT_DIR = Path("C:/Windows/Fonts")
OUT = Path(__file__).resolve().parents[1] / "assets" / "og-eclipse.png"


def font(names, size):
    """First installed face wins; fall back to PIL's bitmap font."""
    for n in names:
        p = FONT_DIR / n
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


display = font(["bahnschrift.ttf", "arialnb.ttf", "ariblk.ttf"], 146)
try:
    # Bahnschrift is variable; pick the same bold-condensed instance the site uses
    display.set_variation_by_axes([700, 82])
except Exception:
    pass
body = font(["segoeui.ttf", "arial.ttf"], 34)
mono = font(["consola.ttf", "cour.ttf"], 21)

img = Image.new("RGB", (W, H), BG)

# --- corona: a wide blue ring, blurred, composited under the crisp disc ------
cx, cy, r = 928, 315, 172
img = img.convert("RGBA")
for width, blur, colour, alpha in (
    (30, 34, BLUE, 255),
    (16, 16, BLUE, 255),
    (46, 46, CYAN, 120),
):
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        [cx - r, cy - r, cx + r, cy + r], outline=colour + (alpha,), width=width
    )
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(blur)))
img = img.convert("RGB")

d = ImageDraw.Draw(img)

# occulting disc, thin rim, and the bright lower-left limb
d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(4, 5, 8), outline=BLUE, width=2)
d.arc([cx - r, cy - r, cx + r, cy + r], start=118, end=192, fill=INK, width=3)

# --- the rail, as on the site ------------------------------------------------
d.line([(74, 0), (74, H)], fill=(255, 255, 255), width=1)
for y, col in ((196, BLUE), (300, CYAN), (404, BLUE)):
    d.ellipse([69, y - 5, 79, y + 5], fill=col)
    d.line([(80, y), (104, y)], fill=(60, 66, 78), width=1)

# --- type --------------------------------------------------------------------
x = 118
d.text((x, 140), "ECLIPSE", font=display, fill=INK)

d.text((x, 306), "An agent-driven quantitative trading", font=body, fill=DIM)
d.text((x, 350), "intelligence system.", font=body, fill=DIM)

d.line([(x, 452), (700, 452)], fill=(38, 43, 52), width=1)  # stops short of the disc
d.text(
    (x, 476),
    "AGENTS PROPOSE   ·   RESEARCH PROVES   ·   RISK AUTHORIZES",
    font=mono,
    fill=FAINT,
)
d.text((x, 506), "EXECUTION ACTS   ·   SECURITY CONTROLS", font=mono, fill=FAINT)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
