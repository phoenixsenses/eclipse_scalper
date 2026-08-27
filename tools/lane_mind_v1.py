# -*- coding: utf-8 -*-
"""LANE MIND V1 -- the recall layer beside the record.  IT WRITES NOTHING.

THE RULE THIS IMPLEMENTS
------------------------
`reports/atlas/_SHARED_LOG.md` is the RECORD and it stays SACRED: append-only,
never edited, never curated, never summarised in place.  Its own header says so.
That property is the only reason it was still alive on 2026-08-27 when every
mechanical surface beside it had gone stale.

This file is the RECALL.  It is DERIVED and DISPOSABLE: delete it and nothing is
lost, because it holds no state of its own.  It prints to stdout and creates NO
FILE.  That is deliberate --

    every surface that went stale on 2026-08-27 was a DERIVED FILE ON DISK.
      _ATLAS_INDEX.json        a day behind (DAY was hard-coded)
      ECLIPSE_BRAIN_V1.md      frozen at 2026-08-26 21:23
      ECLIPSE_CROSSWALK_V1.md  frozen at 2026-08-26 21:24
      ECLIPSE_WITHDRAWALS_V1   frozen at 2026-08-26 21:23

    a reader that writes nothing cannot go stale.  run it, read it, throw it away.

WHAT IT FIXES, EACH ONE A FAILURE MEASURED THAT DAY
---------------------------------------------------
  --who        D-E1 re-derived S101/section 437's frailty result a day after
               S101 established it, because nothing could answer "has anyone
               measured this before?".  `--who frailty multi-spell` answers it.
  --owed       47 messages were addressed to lane B and lane B wrote 1 block.
               The log made the ASKING countable; this makes the BACKLOG visible
               to the lane that owes it.
  --brief      what a lane should read at session start: everything appended
               since ITS OWN last block, and nothing else.  No state file -- the
               cursor is the lane's last block in the log itself.
  --ct         open contradictions, from the register, unfiltered by date.
  --check      format invariants of the record, so the record stays parseable.

IT NEVER FILTERS BY DAY.  `atlas_index_v1.py` carries `DAY = "2026-08-26"` and
therefore indexed zero of 2026-08-27's twenty-plus sections while printing a
clean summary.  A silent empty selection is worse than a crash.

IT KNOWS ALL FOUR ID SHAPES.  `A-S53`, `B-S114`, `C-T43`, `D-E5`.  Section
numbers collide across lanes by design (section 496 was used twice in one
minute); identity is the stable ID, never the number, and this tool never
renumbers anything.

Usage
-----
  python tools/lane_mind_v1.py --brief D          what lane D missed
  python tools/lane_mind_v1.py --who frailty      who has touched this before?
  python tools/lane_mind_v1.py --owed             the obligation matrix
  python tools/lane_mind_v1.py --ct               open contradictions
  python tools/lane_mind_v1.py --check            record format invariants
  python tools/lane_mind_v1.py --json --owed      machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:                      # so the corpus reader imports from anywhere
    sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "reports", "atlas", "_SHARED_LOG.md")
STATE = os.path.join(ROOT, "SYSTEM_STATE.md")
CTREG = os.path.join(ROOT, "CONTRADICTION_REGISTER.md")

BLOCK = re.compile(r"^### (?P<hdr>.+?)$\n```\n(?P<body>.*?)^```", re.S | re.M)
FIELD = re.compile(r"^(?P<k>[a-z][a-z ]*[A-D]?):(?P<v>.*?)(?=^[a-z][a-z ]*[A-D]?:|\Z)",
                   re.S | re.M)
SECT = re.compile(r"^## §(?P<num>\d+)\s*(?P<title>.*)$", re.M)
TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")
# every lane's id shape, in one place
STABLE_ID = re.compile(r"\b([ABCD]-(?:S\d{1,4}[a-z]?|T\d{1,4}|E\d{1,4}|L\d+))\b")
LANES = ("A", "B", "C", "D")
# DECLARED, not derived: a corpus term occurring more often than this carries no
# selectivity.  measured anchors -- `restricted` 77, `passage` 55, `marked` 51 are
# discriminating; `mean` 2446, `process` 4185, `first` 1730, `point` 1563 are not.
NON_DISCRIMINATING_AT = 500


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace")


# ------------------------------------------------------------------ the record
def blocks():
    """Parsed shared-log blocks, in file order.  Nothing is filtered."""
    txt = read(LOG)
    line_of = _line_index(txt)
    out = []
    for m in BLOCK.finditer(txt):
        hdr = m.group("hdr").strip()
        if hdr.startswith("<STABLE_ID>"):          # the format template
            continue
        body = m.group("body")
        fields = {f.group("k").strip(): f.group("v").strip()
                  for f in FIELD.finditer(body)}
        lm = re.search(r"lane ([A-D])", hdr) or re.search(r"by lane ([A-D])", hdr)
        sid = STABLE_ID.search(hdr)
        out.append({
            "header": hdr,
            "lane": lm.group(1) if lm else "?",
            "stable_id": sid.group(1) if sid else hdr.split("·")[0].strip(),
            "date": (re.search(r"(\d{4}-\d{2}-\d{2})", hdr) or [None, ""])[1]
                    if re.search(r"(\d{4}-\d{2}-\d{2})", hdr) else "",
            "line": line_of(m.start()),
            "fields": fields,
            "body": body,
        })
    return out


def _line_index(txt):
    starts = [0]
    for i, ch in enumerate(txt):
        if ch == "\n":
            starts.append(i + 1)

    def f(pos):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1
    return f


def sections():
    """SYSTEM_STATE sections.  Section NUMBERS collide; stable IDs do not."""
    txt = read(LOG and STATE)
    lines = txt.split("\n")
    heads = [(i + 1, m.group("num"), m.group("title").replace("**", "").strip())
             for i, l in enumerate(lines) for m in [SECT.match(l)] if m]
    out = []
    for k, (ln, num, title) in enumerate(heads):
        end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
        body = "\n".join(lines[ln - 1:end])
        sid = STABLE_ID.search(title)
        toks = []
        for blk in re.findall(r"```(.*?)```", body, re.S):
            toks += TOKEN.findall(blk)
        out.append({"line": ln, "section": int(num), "title": title,
                    "stable_id": sid.group(1) if sid else None,
                    "date": (re.search(r"(\d{4}-\d{2}-\d{2})", title) or ["", ""])[1],
                    "tokens": sorted(set(toks)), "body": body})
    return out


def contradictions():
    txt = read(CTREG)
    out = []
    for line in txt.split("\n"):
        m = re.match(r"^\|\s*(CT-[0-9A-Za-z\-]+)\s*\|(.*)$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        status = cells[-1] if cells else ""
        if not status and len(cells) > 1:
            status = cells[-2]
        openish = bool(re.search(r"\bA[ÇC]IK\b|\bOPEN\b|ÇÖZÜLMEDİ|COZULMEDI", status, re.I))
        closed = bool(re.search(r"\bKAPANDI\b|\bCLOSED\b|\bÇÖZÜLDÜ\b|COZULDU", status, re.I))
        out.append({"id": m.group(1), "open": openish and not closed,
                    "closed": closed,
                    "status_head": re.sub(r"\s+", " ", status)[:150],
                    "summary": re.sub(r"\s+", " ", cells[0])[:180] if cells else ""})
    # A RESOLUTION ROW CLOSES ITS PARENT.  The register keeps the original row and
    # APPENDS the resolution as `CT-016-R` -- append-only, like the log.  Reading
    # each row independently reported CT-016 as still open after C had closed it,
    # which would have sent a lane to re-open settled work.
    closers = {c["id"].rsplit("-", 1)[0] for c in out
               if re.search(r"-R\d*$|-RESOLVED$", c["id"]) and c["closed"]}
    for c in out:
        if c["id"] in closers and c["open"]:
            c["open"] = False
            c["closed_by"] = [x["id"] for x in out if x["id"].startswith(c["id"] + "-")]
    return out


# ------------------------------------------------------------------ citations
# Every source on the shelf, under every name this estate actually writes it by.
SOURCE_ALIASES = {
    "AALEN_BORGAN_GJESSING": ("abg", "aalen", "borgan", "gjessing"),
    "HERNAN_ROBINS_WHATIF": ("h&r", "hernan", "hernán", "robins", "whatif"),
    "BOUCHAUD_TQP": ("bouchaud", "tqp"),
    "SURVIVAL_STK4080": ("stk4080", "lindqvist"),
    "KISSELL_SATPM": ("kissell", "satpm"),
    "CARTEA_AHFT": ("cartea", "jaimungal", "penalva", "ahft"),
    "CHAN_AT": ("chan",),
    "HASBROUCK_EMM": ("hasbrouck",),
    "LOPEZDEPRADO_MLAM": ("lopez de prado", "lópez de prado", "mlam"),
    "ABERGEL_LOB": ("abergel",),
    "ECONOPHYS_ODM": ("econophys", "econophysics"),
    "HARRIS_TE": ("harris",),
    "HONORE_1993": ("honore", "honoré"),
}
_ALL_ALIASES = sorted({a for al in SOURCE_ALIASES.values() for a in al}, key=len, reverse=True)
# A citation needs a KEYWORD, a section mark, or a DOTTED locator.  A bare integer after a book
# name is a page, a count or a coincidence -- the checker's first run proved it, by flagging
# "(ABERGEL 138, CARTEA 443)", which is a tally of dash glyphs and not a citation at all.
_KW = (r"§|ch|chapter|technical point|tp|fig|figure|eq|equation|slides|app|appendix|"
       r"exercise|ex|section|b[oö]l[uü]m")
CITE = re.compile(
    r"(?P<src>" + "|".join(re.escape(a) for a in _ALL_ALIASES) + r")"
    r"[\s'’]*(?:(?P<kw>" + _KW + r")\.?\s*\(?\s*(?P<loc1>\d+(?:\.\d+){0,3})"
    r"|(?P<loc2>\d+\.\d+(?:\.\d+){0,2}))", re.I)


def _alias_to_source(a):
    a = a.lower()
    for src, al in SOURCE_ALIASES.items():
        if a in al:
            return src
    return None


def citations(text):
    """Every (source, locator) this text claims.  Mechanical, no judgement."""
    out = []
    for m in CITE.finditer(text):
        src = _alias_to_source(m.group("src"))
        if not src:
            continue
        raw = re.sub(r"\s+", " ", m.group(0)).strip()
        kind = ""
        low = raw.lower()
        for k in ("technical point", "tp", "figure", "fig", "chapter", "ch", "slides",
                  "equation", "eq", "appendix", "app", "exercise"):
            if k in low:
                kind = k
                break
        out.append({"source": src, "kind": kind,
                    "locator": m.group("loc1") or m.group("loc2"),
                    "as_written": raw})
    return out


def resolve_citations(cites):
    """Does the shelf actually carry what was cited?

    Two failure modes, both real and both mechanical:
      SOURCE_NOT_ON_SHELF   the book was cited and it is not here
      LOCATOR_NOT_FOUND     the book is here and the locator never appears in it
    A locator that DOES appear is RESOLVED, which is weaker than "the citation is correct" --
    section numbers recur -- and it is labelled that way.  For `Technical Point` and `Figure`
    the full phrase is searched and the resolution is EXACT.
    """
    try:
        from tools.corpus_text_v1 import bodies
    except Exception as e:
        return {"error": "corpus unreadable: %s" % e, "rows": [], "unresolved": []}
    B = bodies()
    seen, rows = set(), []
    for c in cites:
        key = (c["source"], c["kind"], c["locator"])
        if key in seen:
            continue
        seen.add(key)
        src = c["source"]
        if src not in B:
            rows.append(dict(c, status="SOURCE_NOT_ON_SHELF", hits=0, exact=False))
            continue
        body = B[src]
        if c["kind"] in ("technical point", "tp"):
            n = len(re.findall(r"technical point\s*" + re.escape(c["locator"]), body, re.I))
            exact = True
        elif c["kind"] in ("figure", "fig"):
            n = len(re.findall(r"figure\s*" + re.escape(c["locator"]), body, re.I))
            exact = True
        else:
            n = len(re.findall(r"(?<![\d.])" + re.escape(c["locator"]) + r"(?![\d])", body))
            exact = False
        rows.append(dict(c, status="RESOLVED" if n else "LOCATOR_NOT_FOUND",
                         hits=n, exact=exact))
    return {"rows": rows, "n": len(rows),
            "unresolved": [r for r in rows if r["status"] != "RESOLVED"],
            "caveat": "RESOLVED means the locator occurs in that source.  For section numbers "
                      "that is weak -- numbers recur.  For Technical Point and Figure the phrase "
                      "is searched and the resolution is exact."}


# ------------------------------------------------------------------ the recall
def owed(bl):
    """Who is addressed, who has answered since.  The 47-to-1 measurement."""
    last_block = {L: None for L in LANES}
    for i, b in enumerate(bl):
        if b["lane"] in LANES:
            last_block[b["lane"]] = i
    matrix = {a: {t: 0 for t in LANES} for a in LANES}
    inbox = {L: [] for L in LANES}
    for i, b in enumerate(bl):
        src = b["lane"]
        for tgt in LANES:
            v = b["fields"].get("to %s" % tgt, "").strip()
            if not v or v == "-":
                continue
            if src in LANES:
                matrix[src][tgt] += 1
            if last_block[tgt] is None or i > last_block[tgt]:
                inbox[tgt].append({"from": src, "id": b["stable_id"],
                                   "line": b["line"], "text": v})
    return {"matrix": matrix,
            "blocks_by_lane": {L: sum(1 for b in bl if b["lane"] == L) for L in LANES},
            "unread_since_own_last_block": {L: len(v) for L, v in inbox.items()},
            "inbox": inbox}


def who(terms, bl, sec):
    """Has anyone touched this before?  The S101-duplication preventer."""
    pats = [re.compile(re.escape(t), re.I) for t in terms]

    def hit(s):
        return all(p.search(s) for p in pats)

    res = []
    for s in sec:
        hay = s["title"] + " " + " ".join(s["tokens"])
        if hit(hay):
            res.append({"where": "SYSTEM_STATE", "ref": "§%d" % s["section"],
                        "stable_id": s["stable_id"], "line": s["line"],
                        "date": s["date"], "text": s["title"][:150]})
            continue
        tok = next((t for t in s["tokens"] if hit(t)), None)
        if tok:
            res.append({"where": "SYSTEM_STATE:token", "ref": "§%d" % s["section"],
                        "stable_id": s["stable_id"], "line": s["line"],
                        "date": s["date"], "text": tok})
            continue
        # THE BODY MUST BE SEARCHED TOO.  The first version of this function looked
        # only at titles and verdict tokens, and therefore FAILED THE ONE CASE IT
        # WAS BUILT FOR: section 437 / S101 carries "frailty" in its body, has no
        # fenced verdict block, and so was invisible to `--who frailty` -- the
        # exact query that would have prevented D-E1 from duplicating it.
        if hit(s["body"]):
            ln = next((l for l in s["body"].splitlines() if hit(l)), "")
            res.append({"where": "SYSTEM_STATE:body", "ref": "§%d" % s["section"],
                        "stable_id": s["stable_id"], "line": s["line"],
                        "date": s["date"],
                        "text": (s["title"][:60] + "  ~  " + re.sub(r"\s+", " ", ln).strip())[:170]})
    for b in bl:
        for k in ("verdict", "stands", "what"):
            v = b["fields"].get(k, "")
            if hit(v):
                snip = re.sub(r"\s+", " ", v)
                res.append({"where": "SHARED_LOG:%s" % k, "ref": b["stable_id"],
                            "stable_id": b["stable_id"], "line": b["line"],
                            "date": b["date"], "text": snip[:150]})
                break
    return res


def who_corpus(terms, window=1500, snip=170, max_per_source=3):
    """The other half of "has anyone touched this": what does the CORPUS say?

    Read through `corpus_text_v1` -- NUL-safe and ligature-normalised.  A plain reader misses up
    to 100% of hits on fi/fl terms and skips 3 of 13 files entirely, so this is not optional.

    PROXIMITY, not document-level AND.  Two words both appearing somewhere in a 500-page book is
    not a match; it is a coincidence.  Terms must co-occur within `window` characters.  The anchor
    is whichever term is RAREST in that source, so the scan is cheap and the snippet is centred on
    the informative word.

    An empty result here is `BEYOND_THE_SHELF`, which D-E5 established is a VERDICT and not an
    omission -- provided the terms are discriminating, which this function cannot check.
    """
    try:
        from tools.corpus_text_v1 import bodies
    except Exception as e:                                  # corpus absent is not a crash
        return {"error": "corpus unreadable: %s" % e, "per_source": {}, "total": 0}
    pats = [(t, re.compile(re.escape(t), re.I)) for t in terms]
    # HOW DISCRIMINATING IS EACH TERM?  A term that occurs thousands of times carries no
    # selectivity, and a proximity search anchored beside it returns coincidence.  Measured:
    # `--who restricted mean` returns 29 hits in 7 sources; `--who "restricted mean"` returns 5 in
    # 1, and the second is the right answer.  The whole difference is a pair of quotes, and the
    # cause is that `mean` occurs 2,446 times while `restricted` occurs 77.
    freq = {}
    for t, pp in pats:
        freq[t] = sum(len(pp.findall(b)) for b in bodies().values())
    weak = [t for t, n in freq.items() if n > NON_DISCRIMINATING_AT]
    out, total = {}, 0
    for name, body in sorted(bodies().items()):
        counts = {t: len(p.findall(body)) for t, p in pats}
        if min(counts.values()) == 0:
            continue                                        # a term absent -> no match here
        anchor_t = min(counts, key=counts.get)
        ap = dict(pats)[anchor_t]
        hits, snips = 0, []
        for m in ap.finditer(body):
            lo, hi = max(0, m.start() - window), m.start() + window
            seg = body[lo:hi]
            if all(p.search(seg) for _, p in pats):
                hits += 1
                if len(snips) < max_per_source:
                    a, b = max(0, m.start() - snip), m.start() + snip
                    nl = body.count(chr(10), 0, m.start()) + 1
                    snips.append({"line": nl,
                                  "text": re.sub(r"\s+", " ", body[a:b]).strip()})
        if hits:
            total += hits
            out[name] = {"hits": hits, "anchor_term": anchor_t,
                         "term_counts": counts, "snippets": snips}
    return {"total": total, "n_sources": len(out), "per_source": out,
            "proximity_window_chars": window,
            "term_frequency": freq,
            "non_discriminating_terms": weak,
            "threshold": NON_DISCRIMINATING_AT,
            "reliable": not weak,
            "note": ("a term above the threshold carries no selectivity and the proximity hits "
                     "beside it are coincidence.  quote a multi-word term to search it as ONE "
                     "phrase -- the shell splits it into separate terms otherwise.")
            if weak else ""}


def brief(lane, bl):
    """Everything appended since THIS lane's own last block.  No state file."""
    idx = [i for i, b in enumerate(bl) if b["lane"] == lane]
    cur = idx[-1] if idx else -1
    since = bl[cur + 1:]
    mine = bl[cur] if idx else None
    return {"lane": lane,
            "your_last_block": (mine["stable_id"] if mine else None),
            "your_last_block_line": (mine["line"] if mine else None),
            "blocks_since": [{"id": b["stable_id"], "lane": b["lane"],
                              "line": b["line"],
                              "what": re.sub(r"\s+", " ", b["fields"].get("what", ""))[:160],
                              "to_you": re.sub(r"\s+", " ",
                                               b["fields"].get("to %s" % lane, "").strip())}
                             for b in since],
            "addressed_to_you": sum(1 for b in since
                                    if b["fields"].get("to %s" % lane, "").strip() not in ("", "-")),
            # what the arriving blocks CITE.  C-T31's rule: verify a source in the state it is in
            # when you cite it -- and D-E17 is what happens when a lane does not.
            "citations_arriving": resolve_citations(
                [c for b in since for c in citations(b["body"])])}


def check(bl):
    """Format invariants of the RECORD, so the record stays parseable."""
    required = ("what", "verdict", "stands", "withdraws", "next")
    problems, ids = [], {}
    for b in bl:
        miss = [k for k in required if k not in b["fields"]]
        if miss:
            problems.append({"id": b["stable_id"], "line": b["line"],
                             "problem": "missing fields: %s" % ",".join(miss)})
        missing_to = [t for t in LANES if ("to %s" % t) not in b["fields"]]
        if len(missing_to) == 4:
            problems.append({"id": b["stable_id"], "line": b["line"],
                             "problem": "no `to X` lines at all"})
        if b["lane"] == "?":
            problems.append({"id": b["stable_id"], "line": b["line"],
                             "problem": "lane not parseable from header"})
        ids.setdefault(b["stable_id"], []).append(b["line"])
    dupes = {k: v for k, v in ids.items() if len(v) > 1}
    cites = resolve_citations([c for b in bl for c in citations(b["body"])])
    return {"blocks": len(bl), "problems": problems,
            "citation_resolution": cites,
            "repeated_stable_ids": dupes,
            "note": "a repeated stable id is legal (a correction block reuses it); "
                    "it is listed so a reader can see the thread, not flagged as an error"}


# ------------------------------------------------------------------ rendering
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brief", metavar="LANE", help="what LANE missed since its own last block")
    ap.add_argument("--who", nargs="+", metavar="TERM", help="who has touched these terms before?")
    ap.add_argument("--owed", action="store_true", help="the obligation matrix + per-lane inbox")
    ap.add_argument("--ct", action="store_true", help="open contradictions")
    ap.add_argument("--check", action="store_true", help="record format invariants")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--full", action="store_true", help="print inbox message text in --owed")
    ap.add_argument("--no-corpus", action="store_true",
                    help="--who: skip the corpus half (estate only)")
    a = ap.parse_args()
    if not any([a.brief, a.who, a.owed, a.ct, a.check]):
        ap.print_help()
        return 0

    bl = blocks()
    out = {}
    if a.brief:
        out["brief"] = brief(a.brief.upper(), bl)
    if a.who:
        out["who"] = who(a.who, bl, sections())
        if not a.no_corpus:
            cw = who_corpus(a.who)
            # THE HIGHEST-VALUE OBJECT IN THE SYSTEM: a corpus source that speaks to your terms
            # and that this estate has NEVER cited.  D-E2 found ABG ch.10 that way and D-E3 found
            # the restricted mean that way.
            cited = {c["source"] for b in bl for c in citations(b["body"])}
            for src, v in cw.get("per_source", {}).items():
                v["ever_cited_in_the_log"] = src in cited
            # THREE LEVELS, not two.  The first version reported "never cited" for anything with
            # no parseable locator, and that fires on 7 of 13 sources -- including HONORE_1993,
            # which lane D read end to end and built two rounds on.  It was measuring citation
            # FORMATTING, not neglect.  Short and unstructured sources have no section numbers to
            # cite, so they can never leave that bucket.
            mentioned = set()
            alltext = " ".join(b["body"] for b in bl)
            for src, al in SOURCE_ALIASES.items():
                if any(re.search(re.escape(a), alltext, re.I) for a in al):
                    mentioned.add(src)
            hits = list(cw.get("per_source", {}))
            weak_q = bool(cw.get("non_discriminating_terms"))
            cw["never_mentioned_sources"] = [] if weak_q else [x for x in hits
                                                               if x not in mentioned]
            cw["mentioned_not_cited_sources"] = [] if weak_q else [
                x for x in hits if x in mentioned and x not in cited]
            cw["flag_semantics"] = ("NEVER MENTIONED is the strong signal.  MENTIONED-NOT-CITED "
                                    "only means no locator was pinned, which short sources can "
                                    "never satisfy -- it is a weak hint, not neglect.")
            out["who_corpus"] = cw
    if a.owed:
        o = owed(bl)
        if not a.full:
            o = {k: v for k, v in o.items() if k != "inbox"}
        out["owed"] = o
    if a.ct:
        cs = contradictions()
        out["contradictions"] = {"total": len(cs), "open": [c for c in cs if c["open"]]}
    if a.check:
        out["check"] = check(bl)

    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 0

    print("LANE MIND V1 -- derived, disposable, writes nothing.  "
          "The record is reports/atlas/_SHARED_LOG.md and it is untouched.")
    print("=" * 100)
    if "brief" in out:
        b = out["brief"]
        print("\nBRIEF for lane %s" % b["lane"])
        print("  your last block: %s (log line %s)" % (b["your_last_block"], b["your_last_block_line"]))
        print("  appended since:  %d blocks, %d addressed to you\n" % (len(b["blocks_since"]), b["addressed_to_you"]))
        ca = b.get("citations_arriving", {})
        if ca.get("rows"):
            print("  citations arriving in these blocks (verify, do not inherit -- C-T31):")
            for r in ca["rows"]:
                print("    [%-16s] %-26s %-8s %s"
                      % (r["status"], r["source"], r["locator"], r["as_written"]))
            print("")
        for x in b["blocks_since"]:
            print("  [%s] %-10s line %-6d %s" % (x["lane"], x["id"], x["line"], x["what"]))
            if x["to_you"] and x["to_you"] != "-":
                for ln in x["to_you"].split(". "):
                    if ln.strip():
                        print("        -> %s" % ln.strip()[:150])
    if "who" in out:
        print("\nWHO HAS TOUCHED THIS BEFORE  (%d hits)" % len(out["who"]))
        for h in out["who"]:
            print("  %-22s %-8s %-8s line %-7s %s"
                  % (h["where"], h["ref"], h["stable_id"] or "-", h["line"], h["text"]))
        if not out["who"]:
            print("  none in the estate -- and an empty result here is a CLAIM, not a default.")
            print("  This estate writes in Turkish AND English, often in the same section.")
            print("  Try the other language and a discriminating synonym before")
            print("  concluding nobody has: `cok-spell` / `multi-spell`, `sure` / `duration`.")
    if "who_corpus" in out:
        cw = out["who_corpus"]
        print("")
        print("WHAT THE CORPUS SAYS  (proximity <= %s chars, NUL-safe + ligature-normalised)"
              % cw.get("proximity_window_chars", "?"))
        if cw.get("error"):
            print("  %s" % cw["error"])
        elif not cw["total"]:
            print("  ZERO hits across all 13 sources.")
            print("  That is BEYOND_THE_SHELF -- a VERDICT, not an omission -- PROVIDED your terms")
            print("  are discriminating.  Name the terms you used when you publish it.")
        else:
            print("  %d hits in %d of 13 sources" % (cw["total"], cw["n_sources"]))
            fq = cw.get("term_frequency", {})
            print("  term frequency in the corpus: %s"
                  % ", ".join("%s=%d" % (t, n) for t, n in sorted(fq.items(),
                                                                 key=lambda x: -x[1])))
            if cw.get("non_discriminating_terms"):
                print("  !! NOT DISCRIMINATING (> %d): %s"
                      % (cw["threshold"], ", ".join(cw["non_discriminating_terms"])))
                print("     the hits beside such a term are COINCIDENCE, and the never-cited")
                print("     flag is SUPPRESSED for this query.  quote a multi-word term to")
                print("     search it as ONE phrase -- the shell splits it otherwise.")
                print("     measured: --who restricted mean -> 29 hits / 7 sources;")
                print("               --who \"restricted mean\" -> 5 hits / 1 source.")
            for src, v in sorted(cw["per_source"].items(), key=lambda x: -x[1]["hits"]):
                print("  %-26s %4d hits   anchor=%s   %s"
                      % (src, v["hits"], v["anchor_term"], v["term_counts"]))
                for sn in v["snippets"]:
                    print("      L%-7d %s" % (sn["line"], sn["text"]))
            if cw.get("never_mentioned_sources"):
                print("  >> NEVER MENTIONED IN THE LOG: %s"
                      % ", ".join(cw["never_mentioned_sources"]))
                print("     STRONG signal -- a source that speaks to your terms and that no lane")
                print("     has ever named.  D-E2 found ABG chapter 10 this way.")
            if cw.get("mentioned_not_cited_sources"):
                print("  -- mentioned but no locator pinned: %s"
                      % ", ".join(cw["mentioned_not_cited_sources"]))
                print("     WEAK hint only.  short sources have no section numbers to cite and can")
                print("     never leave this bucket -- HONORE_1993 sits here after two rounds")
                print("     built on it.  this measures citation FORMATTING, not neglect.")
            if cw["n_sources"] == 1:
                print("  NOTE: all hits sit in ONE source.  Check the SENSE in the snippet before")
                print("  citing it -- a term can be a homonym across books (A-S66: `saturat` was")
                print("  45 hits of \"saturated model\", a different meaning entirely).")

    if "owed" in out:
        o = out["owed"]
        print("\nOBLIGATION MATRIX   (rows = author, cols = addressed to)")
        print("     " + "".join("%6s" % t for t in LANES) + "   blocks written")
        for s in LANES:
            print("  %s  " % s + "".join("%6d" % o["matrix"][s][t] for t in LANES)
                  + "%14d" % o["blocks_by_lane"][s])
        print("\n  unread since that lane's own last block:")
        for L in LANES:
            print("    lane %s: %d" % (L, o["unread_since_own_last_block"][L]))
    if "contradictions" in out:
        c = out["contradictions"]
        print("\nCONTRADICTIONS  %d total, %d open" % (c["total"], len(c["open"])))
        for x in c["open"]:
            print("  %-10s %s" % (x["id"], x["summary"]))
    if "check" in out:
        c = out["check"]
        print("\nRECORD CHECK  %d blocks, %d problems" % (c["blocks"], len(c["problems"])))
        cc = c.get("citation_resolution", {})
        if cc.get("error"):
            print("  citations: %s" % cc["error"])
        elif cc:
            bad = cc["unresolved"]
            print("  citations: %d distinct, %d UNRESOLVED" % (cc["n"], len(bad)))
            for r in bad:
                print("    %-18s %-24s %-10s %s"
                      % (r["status"], r["source"], r["locator"], r["as_written"]))
            if not bad:
                print("    every cited locator occurs in the source it names")
        for p in c["problems"]:
            print("  %-12s line %-7d %s" % (p["id"], p["line"], p["problem"]))
        if c["repeated_stable_ids"]:
            print("  threads (same stable id, more than one block):")
            for k, v in c["repeated_stable_ids"].items():
                print("    %-12s lines %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
