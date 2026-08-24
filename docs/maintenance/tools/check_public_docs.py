"""Static checker for Eclipse's public repository surface.

The website has had a machine-checked content policy since it was built
(``web/tools/check_policy.py``). The README and ``docs/`` never did — and the
result was a front page publishing a threshold rule, four feature formulas, a
fill model as source, and fourteen performance figures, under a project whose own
policy forbids every one of them. Counting the places by hand did not work for
the site and it will not work here.

    python docs/maintenance/tools/check_public_docs.py              # from the repo root
    python docs/maintenance/tools/check_public_docs.py --self-test  # run the mutants
    python docs/maintenance/tools/check_public_docs.py --list       # show the file set

Exit code 0 = clean, 1 = at least one violation. No dependencies beyond the
standard library.

What it covers
  README.md · CONTRIBUTING.md · SECURITY.md · docs/public/**.md · docs/assets/**

What it enforces
  1. No performance figure — a number adjacent to bps, a win/hit rate, a profit
     factor, a drawdown, a Sharpe, or a p-value. The bare *word* is allowed:
     these documents have to be able to say which figures are banned.
  2. No threshold or comparison rule stated with a number.
  3. No horizon — a bare horizon suffix, an ``h=`` binding, or a horizon suffix
     attached to a name.
  4. No formula — an assignment or ratio expression in prose or a fenced block.
  5. No health claim — Active / Healthy / Running as a label, and no green,
     amber or red asserting the state of a component.
  6. No ranking between arms — a ranking word in the same sentence as an arm,
     route, lane or strategy.
  7. No network detail — a non-loopback IPv4, a real host, or a bare port that
     is not on a loopback bind line.
  8. Links resolve: every relative link and image target exists, and (in a git
     checkout) would exist on the remote — tracked, or untracked but not
     ignored. A link into an ignored path never resolves on GitHub.
  9. Every anchor resolves to a real heading in the target document.
 10. SVGs are well-formed XML, every ``url(#id)`` resolves, nothing sits outside
     the viewBox, no text overflows, and no external reference is made.
 11. Only an allowlisted external host appears in a markdown URL, and each entry
     on that allowlist carries its reason.

Adding a genuine new exception means adding a category here **with its reason**,
which is the point: the justification lives next to the check.

It is deliberately mutation-tested. ``--self-test`` injects 29 violations into a
scratch copy — one at a time — and every one must be caught. Each is compared
against that copy's own baseline, so a mutant cannot pass on noise it did not
cause; the scratch copy holds only the public surface, so its links out to
``docs/INVARIANTS.md`` and friends are legitimately dead there. The one rule that
cannot be mutated in the copy is ``ignored-link``, because the copy is not a git
checkout — it is exercised directly instead, so it is not left as the rule that
never fires.

**Extend this checker and you must add mutants for the new rule.** When the
site's checker was widened, one rule was silently broken and reported a clean
site for a while — a rule that never fires reads exactly like a rule that passes.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------- declared allowances

# Each entry is (name, why this is not the thing the rule bans, pattern).
# A line matching any of these is exempt from the *content* rules below.

ALLOWANCES = [
    (
        "policy-quotation",
        "these documents must be able to name the figures they forbid",
        re.compile(
            r"(never publish|may not (be )?publish|forbid|banned|prohibit|reserv\w*"
            r"|no performance figure|deliberately carries no|carries no number"
            r"|refuses any|it enforces|what it enforces|rejected|out of scope|no bps"
            r"|without a number|is not mutation-covered)",
            re.I,
        ),
    ),
    (
        "checker-source",
        "the checker's own patterns are not a publication",
        re.compile(r"^\s*(#|\"\"\"|'''|re\.compile|\(\s*$)"),
    ),
    (
        "loopback",
        "a loopback bind is an operator's own machine, not network layout",
        re.compile(r"127\.0\.0\.1|localhost"),
    ),
    (
        "asset-geometry",
        "an SVG canvas size is a drawing dimension, not a market quantity",
        re.compile(r"\d+\s*(×|x)\s*\d+\s*(px)?"),
    ),
]

# --------------------------------------------------------------- content rules

PERF_FIGURE = re.compile(
    r"(?:[-+]?\d+(?:\.\d+)?\s*bps\b)"
    r"|(?:\bbps\b\s*[:=]\s*[-+]?\d)"
    r"|(?:\b(?:win|hit|fill|touch)\s*rate\b[^.\n]{0,24}?[-+]?\d+(?:\.\d+)?\s*%?)"
    r"|(?:[-+]?\d+(?:\.\d+)?\s*%[^.\n]{0,24}?\b(?:win|hit|fill|touch)\s*rate\b)"
    r"|(?:\bprofit\s*factor\b[^.\n]{0,20}?\d)"
    r"|(?:\bdrawdown\b[^.\n]{0,20}?[-+]?\d+(?:\.\d+)?\s*%)"
    r"|(?:\bsharpe\b[^.\n]{0,20}?\d)"
    r"|(?:\bp\s*[=<>]\s*0?\.\d+)"
    r"|(?:\bmc_?p\s*[=:]\s*\d)",
    re.I,
)

THRESHOLD_RULE = re.compile(
    r"(?:[<>]=?\s*[-+]?\d+(?:\.\d+)?)"
    r"|(?:\babs\s*\([^)]*\)\s*[<>]=?)"
    r"|(?:\b(?:threshold|cutoff|offset)\b[^.\n]{0,20}?[=:]\s*[-+]?\d)",
)

HORIZON = re.compile(
    r"(?:\bh\s*=\s*\d+)"
    r"|(?:\bhorizon\b\s*[=:]\s*\d)"
    r"|(?:\+\d+\s*m\b)"
    r"|(?:\b[A-Z][A-Za-z_]{2,}\s+\d{1,3}\s*[HhMm]\b)",
)

FORMULA = re.compile(
    # an assignment whose right-hand side does arithmetic twice over.
    # Operators must be space-delimited and the segment quote-free, so a file
    # path in an attribute (src="docs/assets/x.svg") is not a formula.
    r"(?:\b[a-z_]\w*\s*=\s*[^\"'\n|]*\s[-+*/]\s[^\"'\n|]*\s[-+*/]\s)"
    r"|(?:\b\w+\s*=\s*\([^)]*[-+*/][^)]*\)\s*[-+*/])"
    r"|(?:\b\w+_bps\s*=)"
    r"|(?:\*\s*10000\b)"
    r"|(?:/\s*\(\s*\w+\s*[-+]\s*\w+\s*\))",
)

HEALTH_LABEL = re.compile(
    r"(?:\|\s*(Active|Healthy|Running|Live|Online)\s*\|)"
    r"|(?:\*\*\s*(Active|Healthy|Running|Online)\s*\*\*)"
    r"|(?:\b(green|amber|red)\b[^.\n]{0,16}\b(status|state|healthy|lamp|chip|component)\b)",
    re.I | re.M,
)

RANK_WORD = re.compile(
    r"\b(strongest|best[- ]performing|most promising|top[- ]performing|outperform\w*"
    r"|winning|superior)\b",
    re.I,
)
ARM_NOUN = re.compile(r"\b(arm|arms|route|routes|lane|lanes|pocket|pockets|strategy|strategies)\b", re.I)

NON_LOOPBACK_IP = re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}(?:\.\d{1,3}){3}\b")
BARE_PORT = re.compile(r":(\d{4,5})\b")

# Only hosts with a stated reason. A new entry needs one too.
ALLOWED_HOSTS = {
    "github.com": "the repository and its Actions badges",
    "raw.githubusercontent.com": "raw asset URLs GitHub itself rewrites to",
    "img.shields.io": "static badges, no data of ours leaves in the request",
    "www.w3.org": "the SVG and XML namespaces",
    "eclipse.example": "the reserved placeholder origin the site ships with, "
                       "documented in web/README.md and deliberately non-resolving",
}
URL_HOST = re.compile(r"https?://([A-Za-z0-9.\-]+)")

# --------------------------------------------------------------- file set

MD_TARGETS = ["README.md", "CONTRIBUTING.md", "SECURITY.md"]
MD_GLOBS = ["docs/public/**/*.md", "docs/maintenance/**/*.md", "docs/assets/*.md"]
SVG_GLOB = "docs/assets/*.svg"

# The checker's own source is scanned only for links, never for content: it
# necessarily contains every pattern it bans.
SELF = "docs/maintenance/tools/check_public_docs.py"


@dataclass
class Violation:
    path: str
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        snippet = self.text.strip()
        if len(snippet) > 96:
            snippet = snippet[:93] + "..."
        return f"{self.path}:{self.line}  [{self.rule}]  {snippet}"


def md_files(root: Path) -> list[Path]:
    out = [root / n for n in MD_TARGETS]
    for g in MD_GLOBS:
        out.extend(sorted(root.glob(g)))
    return [p for p in out if p.is_file()]


def svg_files(root: Path) -> list[Path]:
    return sorted(root.glob(SVG_GLOB))


def strip_inline_code(line: str) -> str:
    """A word inside backticks is being named, not asserted."""
    return re.sub(r"`[^`]*`", " ", line)


def publishable_paths(root: Path) -> set[str] | None:
    """Paths that would exist on the public remote after a commit.

    Tracked files plus untracked-but-not-ignored ones — a link to a file that is
    only in the working tree still resolves once it is committed, but a link to
    an ignored file never will. Returns None outside a git checkout.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except Exception:
        return None
    return set(out.splitlines())


# --------------------------------------------------------------- checks


def check_content(root: Path, path: Path) -> list[Violation]:
    v: list[Violation] = []
    rel = path.relative_to(root).as_posix()
    in_fence = False
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if any(pat.search(raw) for _, _, pat in ALLOWANCES):
            continue
        line = strip_inline_code(raw)

        # Inline code hides a *word*, not a *value*. Stripping backticks lets a
        # document write `Active` while naming the ban — the reason the strip
        # exists — but it also let a threshold triple ride into a published
        # document inside a code span, in the very page explaining that
        # thresholds must not be published. Only the health rule, which is
        # about labels, reads the stripped line; the value rules read the raw
        # one.
        for rule, pat in (
            ("performance-figure", PERF_FIGURE),
            ("threshold-rule", THRESHOLD_RULE),
            ("horizon", HORIZON),
            ("formula", FORMULA),
            ("health-claim", HEALTH_LABEL),
        ):
            if rule != "health-claim":
                m = pat.search(raw if in_fence else raw)
                if m:
                    v.append(Violation(rel, i, rule, m.group(0)))
                continue
            m = pat.search(line)
            if m:
                v.append(Violation(rel, i, rule, m.group(0)))

        if RANK_WORD.search(line) and ARM_NOUN.search(line):
            v.append(Violation(rel, i, "arm-ranking", RANK_WORD.search(line).group(0)))

        m = NON_LOOPBACK_IP.search(line)
        if m:
            v.append(Violation(rel, i, "network-detail", m.group(0)))
        m = BARE_PORT.search(line)
        if m and not re.search(r"127\.0\.0\.1|localhost", raw):
            v.append(Violation(rel, i, "network-detail", m.group(0)))

        for host in URL_HOST.findall(raw):
            if host not in ALLOWED_HOSTS:
                v.append(Violation(rel, i, "external-host", host))
    return v


HEADING = re.compile(r"^#{1,6}\s+(.*)$")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_SRC = re.compile(r'<img[^>]+src="([^"]+)"')


def anchor_of(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"`|\*|_", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")


def anchors_in(path: Path) -> set[str]:
    if not path.is_file() or path.suffix.lower() != ".md":
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEADING.match(line)
        if m:
            out.add(anchor_of(m.group(1)))
    return out


def check_links(root: Path, path: Path, tracked: set[str] | None) -> list[Violation]:
    v: list[Violation] = []
    rel = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        targets = MD_LINK.findall(line) + IMG_SRC.findall(line)
        for t in targets:
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            frag = ""
            if "#" in t:
                t, frag = t.split("#", 1)
            if not t:  # same-document anchor
                if frag and frag not in anchors_in(path):
                    v.append(Violation(rel, i, "dead-anchor", "#" + frag))
                continue
            target = (path.parent / t).resolve()
            try:
                trel = target.relative_to(root.resolve()).as_posix()
            except ValueError:
                v.append(Violation(rel, i, "link-escapes-repo", t))
                continue
            if not target.exists():
                v.append(Violation(rel, i, "dead-link", t))
                continue
            if tracked is not None and trel != SELF:
                # a directory resolves if anything publishable lives under it
                published = trel in tracked or (
                    target.is_dir() and any(p.startswith(trel + "/") for p in tracked)
                )
                if not published:
                    v.append(Violation(rel, i, "ignored-link", trel))
            if frag and frag not in anchors_in(target):
                v.append(Violation(rel, i, "dead-anchor", f"{t}#{frag}"))
    return v


CHARW = {"mono": 0.601, "sans": 0.52, "display": 0.47}


def _face(family: str | None) -> str:
    f = (family or "").lower()
    if "mono" in f or "consolas" in f:
        return "mono"
    if "bahnschrift" in f or "din alternate" in f:
        return "display"
    return "sans"


def check_svg(root: Path, path: Path) -> list[Violation]:
    v: list[Violation] = []
    rel = path.relative_to(root).as_posix()
    raw = path.read_text(encoding="utf-8", errors="replace")

    if re.search(r"<!--[^>]*--[^>]*-->", raw):
        v.append(Violation(rel, 1, "svg-comment", "'--' inside an XML comment"))

    try:
        root_el = ET.fromstring(raw)
    except ET.ParseError as e:
        v.append(Violation(rel, getattr(e, "position", (1, 0))[0], "svg-malformed", str(e)))
        return v

    def num(el, attr, default=0.0) -> float:
        try:
            return float(el.get(attr, default))
        except (TypeError, ValueError):
            return float(default)

    W, H = num(root_el, "width"), num(root_el, "height")
    if not W or not H:
        v.append(Violation(rel, 1, "svg-no-size", "missing width/height"))
        return v

    if root_el.get("role") != "img" or not root_el.findall(".//{*}title"):
        v.append(Violation(rel, 1, "svg-a11y", 'needs role="img" and a <title>'))

    ids = {el.get("id") for el in root_el.iter() if el.get("id")}
    refs: set[str] = set()
    for el in root_el.iter():
        for val in el.attrib.values():
            refs |= set(re.findall(r"url\(#([^)]+)\)", str(val)))
    for missing in sorted(refs - ids):
        v.append(Violation(rel, 1, "svg-bad-ref", f"url(#{missing})"))

    for el in root_el.iter():
        for key, val in el.attrib.items():
            if not isinstance(val, str):
                continue
            for host in URL_HOST.findall(val):
                if host != "www.w3.org":
                    v.append(Violation(rel, 1, "svg-external", f"{key}={val}"))

    for el in root_el.iter():
        tag = el.tag.split("}")[-1]
        if tag == "rect":
            x, y, w, h = num(el, "x"), num(el, "y"), num(el, "width"), num(el, "height")
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                v.append(Violation(rel, 1, "svg-out-of-bounds", f"rect {x},{y} {w}x{h}"))
        elif tag == "line":
            for xa, ya in (("x1", "y1"), ("x2", "y2")):
                x, y = num(el, xa), num(el, ya)
                if x < -1 or y < -1 or x > W + 1 or y > H + 1:
                    v.append(Violation(rel, 1, "svg-out-of-bounds", f"line {x},{y}"))
        elif tag == "text":
            if el.get("transform"):
                continue  # rotated: the estimator does not apply
            txt = "".join(el.itertext())
            fs, ls = num(el, "font-size", 12), num(el, "letter-spacing", 0)
            width = len(txt) * CHARW[_face(el.get("font-family"))] * fs
            width += max(0, len(txt) - 1) * ls
            x = num(el, "x")
            anchor = el.get("text-anchor", "start")
            x0 = x if anchor == "start" else (x - width if anchor == "end" else x - width / 2)
            if x0 < -1 or x0 + width > W + 2:
                v.append(
                    Violation(rel, 1, "svg-text-overflow", f"{txt[:56]!r} ends at {x0 + width:.0f}/{W:.0f}")
                )

    if re.search(r"\b\d+(?:\.\d+)?\s*bps\b", raw, re.I):
        v.append(Violation(rel, 1, "performance-figure", "bps figure in a diagram"))
    return v


# --------------------------------------------------------------- driver


def run(root: Path, *, use_git: bool = True) -> list[Violation]:
    tracked = publishable_paths(root) if use_git else None
    out: list[Violation] = []
    for p in md_files(root):
        out += check_content(root, p)
        out += check_links(root, p, tracked)
    for p in svg_files(root):
        out += check_svg(root, p)
    return out


# --------------------------------------------------------------- mutants

# (name, file, needle -> replacement, rule that must catch it)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    ("bps figure", "README.md", "## Why Eclipse exists",
     "## Why Eclipse exists\n\nThe route returned +41.2 bps on average.", "performance-figure"),
    ("bps figure, words first", "README.md", "## Validation",
     "## Validation\n\nMeasured cost was bps: 12 per leg.", "performance-figure"),
    ("win rate", "README.md", "## Status",
     "## Status\n\nThe long leg reached a win rate of 74.6% here.", "performance-figure"),
    ("win rate, number first", "README.md", "## Roadmap",
     "## Roadmap\n\nWe saw 81.6% win rate across the window.", "performance-figure"),
    ("p-value", "docs/public/RESEARCH_METHOD.md", "## 1. The ladder",
     "## 1. The ladder\n\nIt cleared p = 0.001 on permutation.", "performance-figure"),
    ("drawdown", "docs/public/REPRODUCIBILITY.md", "## 1. Determinism is a tested contract",
     "## 1. Determinism is a tested contract\n\nWorst drawdown was 18.4% there.", "performance-figure"),
    ("threshold inside a code span", "docs/public/REPRODUCIBILITY.md", "## 5. Frozen artifacts and content hashes",
     "## 5. Frozen artifacts and content hashes\n\nThe gate was `min_intensity >= 2500` at the time.", "threshold-rule"),
    ("threshold rule", "README.md", "## Module map",
     "## Module map\n\nThe gate fires when imbalance >= 0.5 holds.", "threshold-rule"),
    ("threshold, named", "README.md", "## Quick start",
     "## Quick start\n\nSet the entry threshold: 3500 for the sweep.", "threshold-rule"),
    ("horizon binding", "README.md", "## Testing and CI",
     "## Testing and CI\n\nEvaluated at h=120 for the pocket.", "horizon"),
    ("horizon suffix on a name", "docs/public/ARCHITECTURE_OVERVIEW.md", "## The six planes",
     "## The six planes\n\nOperational Control 4H is the reference arm.", "horizon"),
    ("formula", "docs/public/RESEARCH_METHOD.md", "## 2. Unobserved is not zero",
     "## 2. Unobserved is not zero\n\nvdepth_bps = (a - b) / a * 10000 defines it.", "formula"),
    ("formula in a fence", "README.md", "## Data and observability",
     "## Data and observability\n\n```\nlimit = ep * (1 + 0.5 * spread)\n```\n", "formula"),
    ("health label in a table", "README.md", "| Cross-market portability | `research` |",
     "| Cross-market portability | `research` |\n| Collector | Running |", "health-claim"),
    ("health label, bolded", "docs/public/ARCHITECTURE_OVERVIEW.md", "## Observability plane",
     "## Observability plane\n\nThe collector is **Active** right now.", "health-claim"),
    ("colour asserting state", "docs/public/README.md", "## Contributing",
     "## Contributing\n\nA green status means the component is well.", "health-claim"),
    ("arm ranking", "README.md", "## Research state machine",
     "## Research state machine\n\nThe echo route is the strongest arm we hold.", "arm-ranking"),
    ("non-loopback IP", "SECURITY.md", "## Secrets",
     "## Secrets\n\nThe collector reports to 10.4.19.22 on the mesh.", "network-detail"),
    ("bare port", "SECURITY.md", "## Publication safety",
     "## Publication safety\n\nThe operator surface answers on eclipse-node:8770.", "network-detail"),
    ("unknown external host", "docs/public/README.md", "## Going deeper",
     "## Going deeper\n\nMirror: https://pastebin.example/eclipse\n", "external-host"),
    ("dead link", "docs/public/README.md", "[`RESEARCH_METHOD.md`](RESEARCH_METHOD.md)",
     "[`RESEARCH_METHOD.md`](RESEARCH_METHOD_GONE.md)", "dead-link"),
    ("dead anchor", "README.md", "[Why](#why-eclipse-exists)",
     "[Why](#why-eclipse-does-not-exist)", "dead-anchor"),
    ("svg external reference", "docs/assets/01_eclipse_hero.svg", '<rect width="1200" height="320" fill="#07090D"/>',
     '<image href="https://cdn.example/bg.png" x="0" y="0" width="1200" height="320"/>', "svg-external"),
    ("svg broken gradient ref", "docs/assets/01_eclipse_hero.svg", 'fill="url(#wash)"',
     'fill="url(#nope)"', "svg-bad-ref"),
    ("svg out of bounds", "docs/assets/04_safety_stack.svg", '<rect width="1200" height="576" fill="#07090D"/>',
     '<rect width="1200" height="576" fill="#07090D"/>\n<rect x="1400" y="20" width="300" height="40" fill="#111"/>', "svg-out-of-bounds"),
    ("svg text overflow", "docs/assets/06_governance.svg", '<text x="40" y="88"',
     '<text x="1100" y="88"', "svg-text-overflow"),
    ("svg malformed", "docs/assets/03_research_lifecycle.svg", "</svg>", "<g>\n</svg>", "svg-malformed"),
    ("svg double hyphen comment", "docs/assets/02_system_architecture.svg", "<!-- the rail -->",
     "<!-- the rail -- bus -->", "svg-comment"),
    ("svg bps figure", "docs/assets/05_microstructure_concept.svg", "PRICE</text>",
     "PRICE 41.2 bps</text>", "performance-figure"),
]


def self_test(root: Path) -> int:
    base = run(root, use_git=True)
    if base:
        print("SELF-TEST ABORTED — the surface is not clean to begin with:\n")
        for x in base:
            print("  " + str(x))
        return 1

    missed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="eclipse-pubdocs-") as tmp:
        scratch = Path(tmp) / "surface"
        for rel in [*MD_TARGETS, "docs/public", "docs/maintenance", "docs/assets"]:
            src = root / rel
            dst = scratch / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)

        # The scratch copy holds only the public surface, so links out of it
        # (docs/INVARIANTS.md and friends) are legitimately dead there. Compare
        # every mutant against that baseline and require a *new* violation —
        # otherwise a mutant could pass on noise it did not cause.
        noise = {(x.path, x.rule, x.text) for x in run(scratch, use_git=False)}

        for name, rel, needle, replacement, rule in MUTANTS:
            target = scratch / rel
            original = target.read_text(encoding="utf-8")
            if needle not in original:
                missed.append(f"{name}: needle not found in {rel} — mutant is stale")
                continue
            target.write_text(original.replace(needle, replacement, 1), encoding="utf-8")
            found = run(scratch, use_git=False)
            target.write_text(original, encoding="utf-8")
            new = [x for x in found if (x.path, x.rule, x.text) not in noise]
            if not any(x.rule == rule for x in new):
                got = ", ".join(sorted({x.rule for x in new})) or "nothing new"
                missed.append(f"{name}: expected [{rule}] in {rel}, caught {got}")

        # The ignored-link rule cannot be mutation-tested in the scratch copy,
        # because the copy is not a git checkout and the rule is skipped there.
        # Exercise it directly instead, so it is not the rule that never fires.
        probe = scratch / "README.md"
        probe_original = probe.read_text(encoding="utf-8")
        probe.write_text(
            probe_original + "\n[ignored](docs/assets/README.md)\n", encoding="utf-8"
        )
        direct = check_links(scratch, probe, tracked=set())  # nothing publishable
        probe.write_text(probe_original, encoding="utf-8")
        if not any(x.rule == "ignored-link" for x in direct):
            missed.append("ignored-link: rule did not fire against an empty publishable set")

    total = len(MUTANTS)
    if missed:
        print(f"SELF-TEST FAILED — {len(missed)}/{total} mutants survived:\n")
        for m in missed:
            print("  " + m)
        return 1
    print(f"SELF-TEST PASSED — {total}/{total} mutants caught.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--self-test", action="store_true", help="inject mutants and require every one to be caught")
    ap.add_argument("--list", action="store_true", help="print the file set and exit")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "README.md").is_file():
        # allow running from docs/maintenance/tools/
        for parent in Path(__file__).resolve().parents:
            if (parent / "README.md").is_file() and (parent / "docs").is_dir():
                root = parent
                break

    if args.list:
        for p in md_files(root) + svg_files(root):
            print(p.relative_to(root).as_posix())
        return 0

    if args.self_test:
        return self_test(root)

    violations = run(root)
    if violations:
        print(f"{len(violations)} violation(s):\n")
        for x in violations:
            print("  " + str(x))
        return 1
    n = len(md_files(root)) + len(svg_files(root))
    print(f"clean — {n} files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
