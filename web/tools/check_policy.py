"""Static checker for the Eclipse public site.

The content rules in web/README.md keep regressing — the health-colour rule has
now failed four independent review passes, and twice the README's own statement
of it was factually wrong about the site it describes. Counting places by hand
does not work. This checks the rules against the files instead.

    python web/tools/check_policy.py          # from the repo root
    python tools/check_policy.py              # from web/

Exit code 0 = clean, 1 = at least one violation. No dependencies.

What it enforces
  1. Health colour (green/amber/red) may never assert the state of a component.
     Every occurrence must fall into one of the declared non-health categories
     below, or into the one fenced exception (the projected demo console).
  2. No label reads Active / Healthy / Running outside that fenced console.
  3. No performance figure: bps, win rate, profit factor, drawdown, p-value.
  4. No horizon suffix on an arm name (4H / 6H / +31m / 30m).
  5. No ranking vocabulary between arms (strongest / best / leading / promising).
  6. Structure: no unclosed tag, no duplicate id, no dead link or fragment, no
     undefined CSS class, a skip link and a #main on every page, no external
     request.
  7. One component, one state. The same component may not be Building on one
     page and Accepted on another — including inside a single page, where a
     bulb strip and the section it links to have disagreed.

Adding a genuine new non-health use of colour means adding a category here with
its reason — which is the point: the justification lives next to the check.
"""

from __future__ import annotations

import html
import os
import re
import sys
from html.parser import HTMLParser

# ---------------------------------------------------------------- categories

# Each entry: (name, why it is not a health claim, pattern the coloured text
# must match). The pattern is matched against the text of the element carrying
# the colour, lowercased.
COLOUR_CATEGORIES = [
    (
        "prohibition",
        "red marks something that is forbidden or wrong, never a component's state",
        re.compile(
            r"\b(denied|deny|never|no |not |cannot|refus|wrong|withheld|without)\b"
        ),
    ),
    (
        "custody",
        "amber marks what an agent holds, which is a scope statement",
        re.compile(r"\b(holds?|scope only)\b"),
    ),
    (
        "verdict",
        "the approve / reduce / reject legend is a decision, not a health state",
        re.compile(r"^(approve|reduce|reject)$"),
    ),
    (
        "progress",
        "a completed step in a pipeline is progress, not health",
        re.compile(r"^(complete|completed)$"),
    ),
]

# The single fenced exception: the projected demo console renders component
# status colours for agents that do not exist. It is allowed only while it is
# gated, labelled projected, and carries the sticky bar.
FENCED_FILES = {os.path.join("assets", "js", "master-center.js")}
FENCE_REQUIREMENTS = [
    ("proj-bar", "the sticky Projected bar"),
    ("Projected — nothing here is running", "the sticky bar's text"),
    ("projected", "panel headings that say projected"),
]

HEALTH_COLOUR = re.compile(
    r'(?:color|border-left-color|border-left)\s*:\s*(?:2px\s+solid\s+)?var\(--(green|amber|red)\)'
)

BANNED_LABEL = re.compile(r'>\s*(Active|Healthy|Running|Online)\s*<', re.I)
PERF_FIGURE = re.compile(
    # a figure, not a metric's name: the site is allowed to say what research
    # optimises against, it is not allowed to say what any of it measured
    r"\b\d+(?:\.\d+)?\s?bps\b|\bwin[- ]rate\s+(?:of\s+)?\d|\bprofit factor\s+\d"
    r"|\bdrawdown\s+(?:of\s+)?\d|\bp-value\b|\bsharpe\s+(?:ratio\s+)?(?:of\s+)?\d",
    re.I,
)
HORIZON_SUFFIX = re.compile(r"\b\d+\s?[Hh]\b(?!\w)|\+\d+\s?m\b")
RANKING = re.compile(
    r"\b(strongest|most promising|best[- ]performing|leading arm|outperform\w*)\b", re.I
)

# ---------------------------------------------------------------- components

# Names the site uses for the same thing, so a state claimed on one page can be
# compared with the same claim on another. Only unambiguous aliases belong here:
# the Shared SDK is a library, not an agent, and is deliberately absent.
COMPONENT_ALIASES = {
    "master center": "master-center",
    "event bus": "event-bus",
    "event bus & agent registry": "event-bus",
    "alpha": "alpha",
    "alpha engine": "alpha",
    "the scalper becomes the alpha agent": "alpha",
    "market": "market-intelligence",
    "market intelligence": "market-intelligence",
    "global market intelligence": "market-intelligence",
    "research": "research-engine",
    "research engine": "research-engine",
    "continuous research engine": "research-engine",
    "risk": "risk-governor",
    "risk governor": "risk-governor",
    "execution": "execution-gateway",
    "execution gateway": "execution-gateway",
    "security": "security-guardian",
    "security guardian": "security-guardian",
    "data": "data-guardian",
    "data guardian": "data-guardian",
    "pr guardian": "pr-guardian",
    "organization-wide pr guardian": "pr-guardian",
    "observability": "observability",
    "telemetry": "observability",
}

# A state word and the vocabulary term it means. Several surfaces spell the same
# state differently ("Not implemented" is `planned`), which is fine as long as
# they mean the same thing.
STATE_WORDS = {
    "accepted": "accepted",
    "building": "building",
    "design": "design",
    "next": "design",
    "planned": "planned",
    "not implemented": "planned",
}

VOID = {
    "meta", "link", "br", "hr", "img", "input", "source", "path", "circle",
    "rect", "line", "use", "stop", "polygon", "ellipse", "defs",
}


class Structure(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.ids: dict[str, int] = {}
        self.errors: list[str] = []
        self.links: list[tuple[str, int]] = []
        self.assets: list[tuple[str, int]] = []
        self.classes: set[str] = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids[a["id"]] = self.ids.get(a["id"], 0) + 1
        if "class" in a:
            self.classes.update(a["class"].split())
        if tag == "a" and "href" in a:
            self.links.append((a["href"], self.getpos()[0]))
        if tag in ("script", "img", "link"):
            url = a.get("src") or a.get("href")
            if url and a.get("rel") not in ("canonical",):
                self.assets.append((url, self.getpos()[0]))
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                unclosed = [t for t, _ in self.stack[i + 1:]]
                self.errors.append(
                    f"</{tag}> at line {self.getpos()[0]} closes over unclosed {unclosed}"
                )
                del self.stack[i:]
                return
        self.errors.append(f"stray </{tag}> at line {self.getpos()[0]}")


def element_scope(src: str, pos: int) -> str:
    """Visible text of the element whose start tag carries the declaration.

    A window of characters is not good enough: it swallowed the verdict legend
    into the prohibition category on the first attempt, because some neighbouring
    sentence happened to contain "not". So walk the actual element.
    """
    tag_start = src.rfind("<", 0, pos)
    tag_end = src.find(">", pos)
    if tag_start < 0 or tag_end < 0:
        return ""
    name = re.match(r"<\s*([A-Za-z][-\w]*)", src[tag_start:])
    if not name:
        return ""
    tag = name.group(1).lower()
    if tag in VOID or src[tag_end - 1] == "/":
        return ""
    depth, i = 1, tag_end + 1
    body_start = i
    while depth and i < len(src):
        nxt = src.find("<", i)
        if nxt < 0:
            break
        m = re.match(r"</?\s*([A-Za-z][-\w]*)", src[nxt:])
        if m and m.group(1).lower() == tag:
            depth += -1 if src[nxt + 1] == "/" else 1
            if depth == 0:
                return strip_tags(src[body_start:nxt])
        i = nxt + 1
    return strip_tags(src[body_start:body_start + 200])


def strip_tags(fragment: str) -> str:
    # unescape too: "Event bus &amp; agent registry" must compare equal to the
    # same name written elsewhere without the entity
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    return re.sub(r"\s+", " ", text).strip()


def nearby_label(src: str, pos: int) -> str:
    """The label that gives a coloured value its meaning.

    In a kv row the colour sits on the value while the key ("Denied", "Never
    versioned in place") carries the semantics; in a panel the heading does.
    """
    window = src[max(0, pos - 400):pos]
    for pattern in (r'<div class="kv-k">([^<]*)</div>(?!.*<div class="kv-k">)',
                    r'<div class="panel-hd">([^<]*)</div>(?!.*<div class="panel-hd">)'):
        found = re.findall(pattern, window, re.S)
        if found:
            return strip_tags(found[-1])
    return ""


def line_of(src: str, pos: int) -> int:
    return src.count(chr(10), 0, pos) + 1


def check_colour(path: str, src: str, problems: list[str]) -> list[tuple[str, str]]:
    classified = []
    for m in HEALTH_COLOUR.finditer(src):
        own = element_scope(src, m.start())
        label = nearby_label(src, m.start())
        for name, _why, pattern in COLOUR_CATEGORIES:
            subject = own if name in ("verdict", "progress") else f"{label} | {own}"
            if pattern.search(subject.lower()):
                classified.append((name, f"{path}:{line_of(src, m.start())}"))
                break
        else:
            problems.append(
                f"{path}:{line_of(src, m.start())} health colour --{m.group(1)} on "
                f"{(label + ': ' if label else '') + own[:60]!r} matches no declared "
                f"non-health category"
            )
    return classified


def collect_states(page: str, src: str) -> list[tuple[str, str, int]]:
    """(component, state, line) for every state claim the page makes."""
    found = []

    def add(name: str, state: str, pos: int) -> None:
        key = COMPONENT_ALIASES.get(strip_tags(name).lower().strip())
        word = STATE_WORDS.get(strip_tags(state).lower().strip())
        if key and word:
            found.append((key, word, line_of(src, pos)))

    # bulb strips: <span class="bulb-n">Name</span><span class="bulb-s">State</span>
    for m in re.finditer(
        r'class="bulb-n">([^<]+)</span>\s*<span class="bulb-s">([^<]+)<', src
    ):
        add(m.group(1), m.group(2), m.start())

    # roadmap rows: a phase name followed by its state chip
    for m in re.finditer(
        r'class="road-n">([^<]+)</span>.*?class="tag" data-s="[^"]*">([^<]+)<', src, re.S
    ):
        add(m.group(1), m.group(2), m.start())

    # section headings: <b>Name</b> ... <span class="tag" ...>State</span>
    for m in re.finditer(
        r'<p class="eyebrow">.*?<b>([^<]+)</b>(.*?)</p>', src, re.S
    ):
        tags = re.findall(r'<span class="tag"[^>]*>([^<]+)</span>', m.group(2))
        for t in tags:
            add(m.group(1), t, m.start())

    # tables: first cell is the component, last cell carries the chip
    for m in re.finditer(r"<tr>\s*<td>([^<]+)</td>(.*?)</tr>", src, re.S):
        chips = re.findall(r'<span class="tag"[^>]*>([^<]+)</span>', m.group(2))
        for c in chips:
            add(m.group(1), c, m.start())

    return found


def check_consistency(claims: dict, problems: list[str]) -> None:
    for component, seen in sorted(claims.items()):
        states = {state for state, _where in seen}
        if len(states) > 1:
            detail = "; ".join(f"{state} at {where}" for state, where in sorted(seen))
            problems.append(
                f"{component}: claimed in {len(states)} different states — {detail}"
            )


def check_fence(root: str, problems: list[str]) -> None:
    console = os.path.join(root, "status.html")
    js = os.path.join(root, "assets", "js", "master-center.js")
    if not (os.path.exists(console) and os.path.exists(js)):
        problems.append("fenced console files missing")
        return
    blob = open(console, encoding="utf-8").read() + open(js, encoding="utf-8").read()
    for needle, what in FENCE_REQUIREMENTS:
        if needle not in blob:
            problems.append(
                f"the demo console uses health colour but {what} is gone "
                f"({needle!r} not found) — remove the colours or restore the fence"
            )


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, "assets", "css", "eclipse.css")):
        print("cannot find web/ — run from the repo root or from web/")
        return 1

    css = open(os.path.join(root, "assets", "css", "eclipse.css"), encoding="utf-8").read()
    defined = set(re.findall(r"\.([A-Za-z][-\w]*)", css))

    pages = sorted(
        f for f in os.listdir(root) if f.endswith(".html")
    )
    problems: list[str] = []
    tally: dict[str, int] = {}
    claims: dict[str, set[tuple[str, str]]] = {}

    for page in pages:
        path = os.path.join(root, page)
        src = open(path, encoding="utf-8").read()
        text = re.sub(r"<[^>]+>", " ", src)

        parser = Structure()
        parser.feed(src)
        for err in parser.errors:
            problems.append(f"{page}: {err}")
        if parser.stack:
            problems.append(f"{page}: unclosed at EOF {[t for t, _ in parser.stack]}")
        for ident, count in parser.ids.items():
            if count > 1:
                problems.append(f"{page}: duplicate id {ident!r} ({count}x)")
        for href, line in parser.links:
            if href.startswith(("http", "mailto:", "javascript:")):
                continue
            if href.startswith("#"):
                if href[1:] not in parser.ids:
                    problems.append(f"{page}:{line} dead fragment {href}")
                continue
            target = href.split("#")[0].lstrip("/")
            if target and not os.path.exists(os.path.join(root, target)):
                problems.append(f"{page}:{line} dead link {href}")
        for url, line in parser.assets:
            if url.startswith("http"):
                problems.append(f"{page}:{line} external request {url}")
        for cls in sorted(parser.classes - defined):
            problems.append(f"{page}: class {cls!r} used but not defined in the stylesheet")
        if 'class="skip"' not in src:
            problems.append(f"{page}: no skip-to-content link")
        if 'id="main"' not in src:
            problems.append(f"{page}: no #main for the skip link to reach")

        for name, where in check_colour(page, src, problems):
            tally[name] = tally.get(name, 0) + 1

        for component, state, line in collect_states(page, src):
            claims.setdefault(component, set()).add((state, f"{page}:{line}"))

        for m in BANNED_LABEL.finditer(src):
            problems.append(
                f"{page}:{line_of(src, m.start())} label reads {m.group(1)!r} — "
                f"nothing here is running"
            )
        for pattern, what in (
            (PERF_FIGURE, "performance figure"),
            (HORIZON_SUFFIX, "horizon suffix"),
            (RANKING, "ranking between arms"),
        ):
            for m in pattern.finditer(text):
                problems.append(
                    f"{page}:{line_of(text, m.start())} {what}: {m.group(0)!r}"
                )

    check_consistency(claims, problems)
    check_fence(root, problems)

    def say(line: str) -> None:
        try:
            print(line)
        except UnicodeEncodeError:                      # cp1254 console
            print(line.encode("ascii", "replace").decode())

    say(f"pages checked: {len(pages)}  |  component state claims: "
        f"{sum(len(v) for v in claims.values())} across {len(claims)} components")
    if tally:
        say("health colour, by declared category:")
        for name, count in sorted(tally.items()):
            why = next(w for n, w, _ in COLOUR_CATEGORIES if n == name)
            say(f"  {name:<12} {count:>3}   {why}")
        say(f"  {'fenced':<12}   -   the projected demo console (status.html)")
    if problems:
        print(f"\n{len(problems)} violation(s):")
        for p in problems:
            say(f"  {p}")
        return 1
    print("\nno violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
