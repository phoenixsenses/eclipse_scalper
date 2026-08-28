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

BLOCK = re.compile(r"^### (?P<hdr>[^\n]+)\n```\n(?P<body>.*?)^```", re.S | re.M)
# The header is LINE-BOUNDED on purpose.  With `.+?` under re.S it expands across lines until a
# fence happens to line up, and the FORMAT TEMPLATE at the top of the record contains a literal
# `### <STABLE_ID>` line -- so the template match swallowed the first real block (A-S45), which
# was then discarded along with the template.  Measured D-E25.  A header that cannot span lines
# cannot swallow anything.
FIELD = re.compile(r"^(?P<k>[a-z][a-z ]*[A-D]?):(?P<v>.*?)(?=^[a-z][a-z ]*[A-D]?:|\Z)",
                   re.S | re.M)
SECT = re.compile(r"^(?:## |\*\*)§(?P<num>\d+)\s*(?P<title>.*)$", re.M)
# A-S90, VERIFIED AT SOURCE AND IT WAS RIGHT.  The old pattern matched `## §N` only, and
# SYSTEM_STATE carries TWO header shapes: 309 `## §` starting at line 33,419 and 374 `**§`
# starting at line 9,030.  So `sections()` returned 309 of 683 and 54.8% of the file was
# invisible to the ESTATE half of `--who` -- the half whose whole purpose is answering "has
# anyone measured this before?".  The blind half is the OLDER half, which is exactly where a
# prior measurement would live.
TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")
# every lane's id shape, in one place
STABLE_ID = re.compile(r"^([ABCD]-[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*)")
# widened 2026-08-27 (D-E25) from the four canonical shapes to what the record ACTUALLY carries:
# lane C writes `C-KULLIYAT-T55`, which the old pattern REJECTED, and that rejection is what
# triggered the silent fallback.  A convention the tool refuses to read is a defect in the tool,
# not in the lane.  Anchored with ^ so a CITED id inside a header can never win.
LANES = ("A", "B", "C", "D")
# DECLARED, not derived: a corpus term occurring more often than this carries no
# selectivity.  measured anchors -- `restricted` 77, `passage` 55, `marked` 51 are
# discriminating; `mean` 2446, `process` 4185, `first` 1730, `point` 1563 are not.
NON_DISCRIMINATING_AT = 500


def read(path):
    """Read a record file with line endings NORMALISED.

    THE RECORD IS MIXED CRLF AND LF, and `BLOCK` requires a bare newline after the header line.
    On a CRLF block the header match fails and backtracks under `re.S` until the header SWALLOWS
    THE BODY -- after which the ID pattern grabs the first ID-shaped string it finds, which is
    normally a CITATION of another lane.  Measured 2026-08-27 (D-E25): `C-KULLIYAT-T55` was filed
    under `D-E22`, an ID belonging to a different lane, purely because that block cited it.
    Third CRLF-caused defect in one day, after the corpus probe and the hyphen fold.
    """
    with open(path, "rb") as fh:
        t = fh.read().decode("utf-8", "replace")
    return t.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))


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
        sid = STABLE_ID.match(hdr.strip())   # match, NOT search: a cited id must never win
        out.append({
            "header": hdr,
            "lane": lm.group(1) if lm else "?",
            "stable_id": sid.group(1) if sid else hdr.split("·")[0].strip(),
            # A SILENT FALLBACK HERE HAS THE SHAPE A-S77 NAMED: success reported over a wrong
            # selection, exactly like the hard-coded DAY that blinded atlas_index_v1.  An id that
            # cannot be parsed is now MARKED and surfaced by --check, never quietly replaced.
            "id_parse": "OK" if sid else "UNPARSEABLE",
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


def _lane_of(sid, title=""):
    """Lane letter from a stable id like `D-E39`, or from a section title's `[D-E40]` stamp.

    SYSTEM_STATE sections do not carry the id in `stable_id`; they carry it in the TITLE as
    `[D-E40]`.  Reading only `stable_id` classified this lane's OWN sections as another writer's,
    so `--who "inverse gaussian"` reported 1 INDEPENDENT_PRIOR that was section 582 -- a section
    written by this lane one round earlier.  A classifier that counts my own work as independent
    prior work is the exact failure it was built to prevent.
    """
    m = re.match(r"^([ABCD])-", str(sid or ""))
    if m:
        return m.group(1)
    m = re.search(r"\[([ABCD])-[A-Za-z0-9_\-]+\]", str(title or ""))
    return m.group(1) if m else None


def _provenance(lane, hits):
    """A HIT IS NOT PRIOR WORK.  Classify every estate hit by WHO wrote it and WHEN.

    `--who > 0` was being read as "someone measured this before", and it is not.  C-T68 measured
    that 67-91% of hits on a distinctive term are the SEARCHER'S OWN blocks; on lane D's own terms
    it is worse -- 100% on six of eight (`restricted mean`, `cause-specific hazard`,
    `inverse gaussian`, `local dependence`, `never_alive`, `edge_gone`).  A tool that answers
    "has anyone measured this?" by showing you your own work is answering a different question.

    C-T67 named the other half: a record that quotes its own controls eventually contaminates
    them.  In four lanes that read each other, a hit written AFTER this lane first raised a term
    may be an ECHO of it rather than independent prior art, and counting it as prior art is
    circular.

    Four classes, and only one of them is evidence:
      SELF                this lane's own writing.  Not independent.  Not evidence of prior work.
      INDEPENDENT_PRIOR   another writer, and written BEFORE this lane first mentioned the term.
      ECHO_RISK           another writer, but written AFTER this lane first mentioned it.  It may
                          be a genuine independent result or a response to this lane; the record
                          cannot tell them apart, so it is reported as a risk, not as evidence.
      CORPUS              the shelf.  A separate leg entirely, never mixed into these counts.

    A hit is independent prior only when it is provably before EVERY SELF hit.  Within one file,
    line order proves that.  Across SYSTEM_STATE and the shared log, line numbers have no common
    axis, so only a strictly earlier ISO date proves it; equal or missing dates remain ECHO_RISK.
    This is conservative in one stated direction: it can over-count ECHO_RISK, never manufacture
    INDEPENDENT_PRIOR.
    """
    # SYSTEM_STATE and the shared log are DIFFERENT FILES whose line numbers share no axis.  The
    # first repair kept one cut per file, but A-S93 found the remaining hole: a later SELF hit in
    # the log made earlier log hits look prior even though a SYSTEM_STATE SELF hit had already
    # raised the term.  Pairwise proof against EVERY SELF hit closes both defects.
    def fam(h):
        return "SYSTEM_STATE" if str(h.get("where", "")).startswith("SYSTEM_STATE") else "LOG"

    self_hits = [h for h in hits if h.get("writer_lane") == lane]
    cuts = {}
    for h in self_hits:
        f = fam(h)
        cuts[f] = min(cuts.get(f, h["line"]), h["line"])

    def provably_before(hit, own):
        if fam(hit) == fam(own):
            return hit["line"] < own["line"]
        hd, od = str(hit.get("date") or ""), str(own.get("date") or "")
        return bool(hd and od and hd < od)

    # A-S93 IS RIGHT AND THE FIX DID NOT CLOSE IT -- THE DEFECT MOVED.  A-S93 reported that
    # `f not in cuts` called every hit in a file prior work when the asker had no SELF hit there.
    # That branch is gone, but `all(provably_before(h, own) for own in self_hits)` is VACUOUSLY
    # TRUE when `self_hits` is empty, so the same hits get the same label by a different route.
    # Measured over ten terms: 27% of all INDEPENDENT_PRIOR comes from that case
    # (`spurious regression` 6 of 6, `oracle ceiling` 10 of 10).
    #
    # BUT THE LABEL IS NOT WRONG, ONLY UNMARKED.  With no self hit anywhere in the record, this
    # lane never raised the term, so nobody can be echoing it -- the hits ARE work by others that
    # this lane has not done, which is exactly what the caller needs to know.  Turning them into
    # ECHO_RISK "to be safe" would HIDE genuine prior work, which is the failure this tool exists
    # to prevent.  So the ordering is marked instead: PROVEN when it was actually established,
    # VACUOUS_NO_SELF_HIT when there was nothing to order against.
    for h in hits:
        if h.get("writer_lane") == lane:
            h["provenance"] = "SELF"
            h["ordering"] = "-"
        elif not self_hits:
            h["provenance"] = "INDEPENDENT_PRIOR"
            h["ordering"] = "VACUOUS_NO_SELF_HIT"
        elif all(provably_before(h, own) for own in self_hits):
            h["provenance"] = "INDEPENDENT_PRIOR"
            h["ordering"] = "PROVEN"
        else:
            h["provenance"] = "ECHO_RISK"
            h["ordering"] = "-"
    return hits, cuts


def who(terms, bl, sec, lane=None):
    """Has anyone touched this before?  The S101-duplication preventer.

    `lane` enables provenance classification -- see `_provenance`.  Without it every hit is
    returned unclassified, which is what the first year of this tool did and what made a raw hit
    count read as prior work.
    """
    # WHITESPACE IS NOT LITERAL HERE EITHER.  The corpus call site learned this in D-E22, but the
    # estate call site kept `re.escape(term)`: SYSTEM_STATE section 281 contains "The fixed" at
    # one line end and "design uses" on the next, while `--who "fixed design"` returned ZERO.
    # A fix lands in a call site, not in a concept.  Single-word terms are unchanged.
    pats = [re.compile(r"\s+".join(re.escape(w) for w in t.split()), re.I) for t in terms]

    def hit(s):
        return all(p.search(s) for p in pats)

    res = []
    for s in sec:
        hay = s["title"] + " " + " ".join(s["tokens"])
        if hit(hay):
            res.append({"where": "SYSTEM_STATE", "ref": "§%d" % s["section"],
                        "stable_id": s["stable_id"], "line": s["line"],
                        "writer_lane": _lane_of(s["stable_id"], s["title"]),
                        "date": s["date"], "text": s["title"][:150]})
            continue
        tok = next((t for t in s["tokens"] if hit(t)), None)
        if tok:
            res.append({"where": "SYSTEM_STATE:token", "ref": "§%d" % s["section"],
                        "stable_id": s["stable_id"], "line": s["line"],
                        "writer_lane": _lane_of(s["stable_id"], s["title"]),
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
                        "writer_lane": _lane_of(s["stable_id"], s["title"]),
                        "date": s["date"],
                        "text": (s["title"][:60] + "  ~  " + re.sub(r"\s+", " ", ln).strip())[:170]})
    # EVERY FIELD, NOT THREE OF NINE.  Measured D-E42: reading only `what`, `verdict` and
    # `stands` left 50.8% of the shared log's own text invisible -- `withdraws` (53k chars),
    # `next` (28k) and ALL FOUR `to X` lines (277k combined).  The `to X` lines are where lanes
    # hand each other findings, so the field most likely to answer "has anyone told me about
    # this?" was the one never searched.  Same shape as A-S90's header defect one round earlier:
    # searching a subset of the record while reporting on all of it.
    for b in bl:
        # A HARDCODED FIELD LIST GOES STALE, SO THERE IS NO LIST.  D-E42 widened this from three
        # fields to nine and A-S94 found the tenth one round later: `corpus`, present in 45 blocks
        # and 40,678 characters -- the field where lanes record WHAT THE SHELF SAID, invisible to
        # the command that asks what the shelf said.  An `also` field exists too, in one block,
        # which nobody would have thought to add.  Every field the record actually carries is read
        # now, in a stable order so the reported `where` stays deterministic.
        for k in sorted(b["fields"]):
            v = b["fields"].get(k, "")
            if hit(v):
                snip = re.sub(r"\s+", " ", v)
                res.append({"where": "SHARED_LOG:%s" % k, "ref": b["stable_id"],
                            "stable_id": b["stable_id"], "line": b["line"],
                            "writer_lane": b["lane"], "date": b["date"], "text": snip[:150]})
                break
    # THE CONTRADICTION REGISTER IS ESTATE TOO.  28k of text that --who never opened, and it
    # is the file that answers "has anyone flagged a CONFLICT about this?" -- a different
    # question from "has anyone measured it", and one this tool was silently not answering.
    # Rows carry no lane stamp, so they classify by the same rule as any unstamped writer.
    # NO SILENT SWALLOW.  The first version wrapped this in a bare `except Exception: pass` and
    # named a constant that does not exist (`CT` rather than `CTREG`), so every query returned
    # ZERO register rows and reported success -- the exact silent-fallback shape this lane has
    # catalogued three times.  A missing register is now a VISIBLE row, not an empty result.
    if os.path.exists(CTREG):
        for i, ln in enumerate(read(CTREG).split(chr(10)), 1):
            if len(ln.strip()) > 20 and hit(ln):
                m = re.search(r"CT-\d+", ln)
                res.append({"where": "CONTRADICTION_REGISTER",
                            "ref": m.group(0) if m else "CT-?",
                            "stable_id": None, "line": i, "writer_lane": None,
                            "date": "", "text": re.sub(r"\s+", " ", ln).strip()[:150]})
    else:
        res.append({"where": "CONTRADICTION_REGISTER", "ref": "UNREADABLE",
                    "stable_id": None, "line": 0, "writer_lane": None, "date": "",
                    "text": "register not found at %s -- this row exists so the absence is "
                            "visible instead of silent" % CTREG})
    if lane:
        res, _cut = _provenance(lane.upper(), res)
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
        from tools.corpus_text_v1 import bodies, normalise
    except Exception as e:                                  # corpus absent is not a crash
        return {"error": "corpus unreadable: %s" % e, "per_source": {}, "total": 0}
    terms = list(terms)
    if not terms:
        raise ValueError("corpus query requires at least one term")
    # One query, one shelf snapshot.  The old path re-read and re-normalised all 13 books once
    # per term for frequencies and then once again for results.  Besides wasting minutes in the
    # acceptance suite, a mid-query shelf change could make diagnostics and hits describe
    # different corpus states.
    corpus = bodies()
    # WHITESPACE IS NOT LITERAL.  `re.escape("funding rate")` demands EXACTLY one space, but PDF
    # text carries a NEWLINE wherever the phrase straddles a line break, and column layout gives
    # runs of spaces.  Measured on eight control phrases known to be in the shelf: `limit order`
    # 1639 -> 1744, `order book` 1143 -> 1218, `market impact` 603 -> 642.  6.0% of real phrase
    # hits were INVISIBLE to this function.  Every word boundary is now `\s+`.
    pats = []
    for t in terms:
        words = normalise(t).split()
        if not words:
            raise ValueError("corpus query term is empty after normalisation")
        pats.append((t, re.compile(r"\s+".join(re.escape(w) for w in words), re.I)))
    # HOW DISCRIMINATING IS EACH TERM?  A term that occurs thousands of times carries no
    # selectivity, and a proximity search anchored beside it returns coincidence.  Measured:
    # `--who restricted mean` returns 29 hits in 7 sources; `--who "restricted mean"` returns 5 in
    # 1, and the second is the right answer.  The whole difference is a pair of quotes, and the
    # cause is that `mean` occurs 2,446 times while `restricted` occurs 77.
    freq = {}
    for t, pp in pats:
        freq[t] = sum(len(pp.findall(b)) for b in corpus.values())
    weak = [t for t, n in freq.items() if n > NON_DISCRIMINATING_AT]
    out, total = {}, 0
    for name, body in sorted(corpus.items()):
        counts = {t: len(p.findall(body)) for t, p in pats}
        if min(counts.values()) == 0:
            continue                                        # a term absent -> no match here
        anchor_t = min(counts, key=counts.get)
        ap = dict(pats)[anchor_t]
        hits, snips, embedded = 0, [], 0
        for m in ap.finditer(body):
            lo, hi = max(0, m.start() - window), m.start() + window
            seg = body[lo:hi]
            if all(p.search(seg) for _, p in pats):
                hits += 1
                # IS THE MATCH EMBEDDED IN A LONGER WORD?  `overlapping returns` matches
                # `NON-overlapping returns` -- the OPPOSITE concept -- and that spurious hit was
                # the only thing separating a 2-hit from a 3-hit result in a live query.  Word
                # boundaries are NOT imposed, because D-E17 searched the stem `identifiab` on
                # purpose and every one of those hits is embedded.  So this is REPORTED, not
                # filtered: a stem query is legitimately near 100%, a phrase query should be 0%.
                if ((m.start() > 0 and (body[m.start() - 1].isalnum()
                                        or body[m.start() - 1] == "-"))
                        or (m.end() < len(body) and body[m.end()].isalnum())):
                    embedded += 1
                if len(snips) < max_per_source:
                    a, b = max(0, m.start() - snip), m.start() + snip
                    nl = body.count(chr(10), 0, m.start()) + 1
                    snips.append({"line": nl,
                                  "text": re.sub(r"\s+", " ", body[a:b]).strip()})
        if hits:
            total += hits
            out[name] = {"hits": hits, "anchor_term": anchor_t, "embedded_hits": embedded,
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

    # THE MID-ROUND RACE, measured D-E29.  A cursor that is "my own last block" silently drops
    # everything that arrives AFTER a lane runs --brief and BEFORE it appends its block: writing
    # the block jumps the cursor past those arrivals and they are never shown again.  Measured on
    # this record: seven blocks landed between D-E24 and D-E25; four were visible when that round
    # opened and THREE -- A-S78, C-KULLIYAT-T56, C-T58 -- were not, and stayed invisible for four
    # rounds.  C-T58 was a live defect report on the canonical corpus reader.
    # The window (my block n-1, my block n) is exactly what round n could not have seen.  Showing
    # it once at the start of round n+1 is stateless, never repeats, and leaves no gap.
    prev = idx[-2] if len(idx) > 1 else -1
    during = [b for b in bl[prev + 1:cur]
              if b["lane"] != lane
              and b["fields"].get("to %s" % lane, "").strip() not in ("", "-")]
    return {"lane": lane,
            "arrived_during_your_last_round": [
                {"id": b["stable_id"], "lane": b["lane"], "line": b["line"],
                 "to_you": re.sub(r"\s+", " ", b["fields"].get("to %s" % lane, "").strip())}
                for b in during],
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


def inbox(lane, bl):
    """EVERY message addressed to `lane`, over the WHOLE record, IGNORING the cursor.

    `--brief` is cursor-based: it shows what arrived since the lane's own last block.  That is
    right for a running lane and WRONG after a reader repair.  D-E25 recovered 13 blocks the
    parser had been dropping, 7 of them addressed to lane D -- and every one sits BEFORE the
    current cursor, so `--brief` will never surface them.  A repair that restores the record but
    not the delivery is half a repair.

    Run once per lane after any parser change.  Cheap, and it cannot go stale: it derives from
    the record every time and writes nothing.
    """
    out = []
    for b in bl:
        if b["lane"] == lane:
            continue
        v = b["fields"].get("to %s" % lane, "").strip()
        if v and v != "-":
            out.append({"from": b["lane"], "id": b["stable_id"], "line": b["line"], "text": v})
    return out


def promises(lane, bl):
    """Did the lane DO what its own `next:` line said, or did it just write it again?

    A-S80 named the shape and proposed the machine-checkable form: a finding that runs against a
    lane's own plan gets recorded as a caveat and never becomes the next test.  The tractable core
    of that is narrower and entirely inside this record: a lane's `next:` line IS a promise, and
    the block after it either takes it up or it does not.

    HEURISTIC, AND SAID SO.  It matches content words from the promise against the following
    block's `what` and `stands`.  A promise restated in different words reads as unmet; a promise
    deliberately abandoned reads the same as one forgotten.  It FLAGS, it does not judge -- the
    lane says which it was.  Reported for every lane; edited for none.
    """
    stop = {"which", "there", "their", "these", "those", "would", "could", "about", "after",
            "before", "where", "while", "still", "other", "another", "whether", "rather",
            "something", "anything", "nothing", "lane", "round", "block", "record"}
    # A LANE IS NOT ALWAYS ONE WRITER.  C-T62 and C-KULLIYAT-T60 both reported this and it is
    # confirmed: lane C is written by TWO agents with different id stems -- 42 blocks under `C-T`
    # and 19 under `C-KULLIYAT-T` -- so 33 of 60 consecutive C pairs (55%) matched one agent's
    # `next:` against the OTHER agent's block.  D-E32's "lane C, 40 of 56 unmet" was measured that
    # way and is withdrawn.  Lanes A and D carry a single stem each, which is why the defect only
    # ever showed on C.  Promises are now grouped by ID STEM, and the stem is printed so a reader
    # can see which writer is being judged.
    def stem(sid):
        m = re.match(r"^([A-Za-z][A-Za-z\- ]*[A-Za-z])", sid or "")
        return m.group(1) if m else (sid or "?")

    idx_all = [i for i, b in enumerate(bl) if b["lane"] == lane]
    groups = {}
    for i in idx_all:
        groups.setdefault(stem(bl[i]["stable_id"]), []).append(i)
    idx = [i for g in groups.values() for i in g]          # kept for the null's body list
    out = []
    for g in groups.values():
      for a, nx in zip(g, g[1:]):
          nxt = bl[a]["fields"].get("next", "").strip()
          if not nxt or nxt == "-":
              continue
          terms = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z_\-]{4,}", nxt)]
          terms = [t for t in dict.fromkeys(terms) if t not in stop][:10]
          body = " ".join([bl[nx]["fields"].get(k, "") for k in ("what", "verdict", "stands")]).lower()
          hit = [t for t in terms if t in body]
          need = max(1, len(terms) // 3)
          out.append({"promised_in": bl[a]["stable_id"], "judged_at": bl[nx]["stable_id"],
                      "stem": stem(bl[a]["stable_id"]),
                      "line": bl[a]["line"], "next": re.sub(r"\s+", " ", nxt)[:150],
                      "terms": terms, "matched": hit, "kept": len(hit) >= need})

    # NULL CALIBRATION, BEFORE THE RATE IS READABLE.  A content-word matcher scores a lane's
    # WRITING STYLE as much as its follow-through, so the raw rate is not comparable across lanes
    # and must not be quoted as one.  Measured D-E32: lane A's observed rate sits z = +0.91 from a
    # permutation null -- INDISTINGUISHABLE FROM CHANCE -- while C is +2.47 and D is +2.94.
    # Publishing A's "34 of 36 unmet" as a finding would have been a fabricated indictment of
    # another lane from a probe nobody had calibrated.  So the rate is withheld unless the lane's
    # own null says it carries information; the flagged INSTANCES are always listed, because a
    # human reading one costs nothing and the instance is the product.
    if len(out) >= 4:
        import random as _r
        import statistics as _st
        rng = _r.Random(20260827)
        bodies = [" ".join([bl[i]["fields"].get(k, "") for k in ("what", "verdict", "stands")])
                  .lower() for i in idx]
        obs = sum(1 for x in out if x["kept"]) / len(out)
        nulls = []
        for _ in range(400):
            sh = bodies[:]
            rng.shuffle(sh)
            k = 0
            for j, x in enumerate(out):
                need2 = max(1, len(x["terms"]) // 3)
                body = sh[min(j + 1, len(sh) - 1)]
                if sum(1 for t in x["terms"] if t in body) >= need2:
                    k += 1
            nulls.append(k / len(out))
        m, sd = _st.mean(nulls), _st.pstdev(nulls)
        z = (obs - m) / sd if sd > 0 else float("nan")
        # A z-GATE ALONE IS NOT ENOUGH AT THESE COUNTS.  Regrouping lane A by id stem moved a
        # SINGLE pair and its z went +0.91 -> +2.20, because the whole statistic rested on 3 kept
        # promises out of 37.  A rate built on a handful of successes is not a property of a lane
        # whatever its z says, so the gate now needs BOTH |z| > 2 AND at least 5 kept.
        kept_n = sum(1 for x in out if x["kept"])
        cal = {"observed_kept_rate": round(obs, 3), "null_mean": round(m, 3),
               "kept_n": kept_n,
               "z": round(z, 2), "informative": bool(abs(z) > 2 and kept_n >= 5),
               "note": ("the rate scores writing style as much as follow-through; it is NOT "
                        "comparable across lanes and is withheld when the null says so")}
    else:
        cal = {"informative": False, "note": "too few blocks to calibrate"}
    return {"items": out, "calibration": cal}


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
    for b in bl:
        if b.get("id_parse") == "UNPARSEABLE":
            problems.append({"id": b["stable_id"], "line": b["line"],
                             "problem": "header is not a stable ID; not silently replaced"})

    # A REPAIRED DEFECT MUST NOT READ LIKE A LIVE ONE.  The record is append-only, so a malformed
    # block is corrected by a LATER block in the same thread and the original stays malformed
    # forever.  Reporting both identically made every reader -- the operator included -- conclude
    # the defect was never fixed.  D-E22 lacked `to X` lines and D-E22-R lacked `withdraws`;
    # D-E22-R2 supplies BOTH, and section 546 records that.  So a problem whose thread later
    # supplies what it lacked is moved to SUPERSEDED, naming the block that supplied it.  It is
    # never deleted: the history is the point of an append-only record.
    def thread_of(sid):
        return re.sub(r"(-R\d*)+$", "", sid or "")

    by_thread = {}
    for b in bl:
        by_thread.setdefault(thread_of(b["stable_id"]), []).append(b)

    open_probs, superseded = [], []
    for pr in problems:
        later = [b for b in by_thread.get(thread_of(pr["id"]), []) if b["line"] > pr["line"]]
        fixer = None
        for b in later:
            if "missing fields:" in pr["problem"]:
                missing = pr["problem"].split(":", 1)[1].strip().split(",")
                if all(m.strip() in b["fields"] for m in missing):
                    fixer = b
                    break
            elif "no `to X`" in pr["problem"]:
                if any(k.startswith("to ") for k in b["fields"]):
                    fixer = b
                    break
        if fixer:
            superseded.append(dict(pr, content_supplied_by=fixer["stable_id"],
                                   superseded_by=fixer["stable_id"],
                                   superseded_at=fixer["line"]))
        else:
            open_probs.append(pr)

    dupes = {k: v for k, v in ids.items() if len(v) > 1}
    cites = resolve_citations([c for b in bl for c in citations(b["body"])])
    # A-S77: a silent fallback reports success over a wrong selection.  An id the parser could
    # not read is a PROBLEM, not a detail -- D-E25 measured 13 blocks lost to exactly that shape.

    return {"blocks": len(bl), "problems": open_probs,
            "superseded": superseded,
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
    ap.add_argument("--promises", metavar="LANE",
                    help="did this lane do what its own next: line said?")
    ap.add_argument("--inbox", metavar="LANE",
                    help="EVERY message addressed to LANE, ignoring the cursor")
    ap.add_argument("--owed", action="store_true", help="the obligation matrix + per-lane inbox")
    ap.add_argument("--ct", action="store_true", help="open contradictions")
    ap.add_argument("--check", action="store_true", help="record format invariants")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--full", action="store_true", help="print inbox message text in --owed")
    ap.add_argument("--no-corpus", action="store_true",
                    help="--who: skip the corpus half (estate only)")
    a = ap.parse_args()
    # Provenance is defined only for the four actual lanes.  Previously a typo such as
    # `--brief E --who frailty` found no SELF cut and therefore classified every estate hit as
    # INDEPENDENT_PRIOR.  An invalid identity must fail closed before any evidence is rendered.
    for flag, value in (("--brief", a.brief), ("--inbox", a.inbox),
                        ("--promises", a.promises)):
        if value is not None and value.upper() not in LANES:
            ap.error("%s lane must be one of %s" % (flag, ", ".join(LANES)))
    # These two human-only renderers predate the JSON contract and currently print prose followed
    # by `{}`, which is neither JSON nor useful machine data.  Adding new keys is forbidden by the
    # handover.  Reject the unsupported combination explicitly instead of emitting counterfeit
    # machine output; valid JSON commands and all existing keys remain unchanged.
    if a.json and (a.promises or a.inbox):
        ap.error("--json is not supported for --promises or --inbox; no schema is frozen")
    if not any([a.brief, a.who, a.owed, a.ct, a.check, a.inbox, a.promises]):
        ap.print_help()
        return 0

    bl = blocks()
    out = {}
    if a.brief:
        out["brief"] = brief(a.brief.upper(), bl)
    if a.who:
        # provenance needs to know WHOSE lane is asking; --brief supplies it, and
        # without it every hit comes back unclassified, which is the old behaviour.
        _asker = (a.brief or a.inbox or a.promises or "")[:1].upper() or None
        out["who"] = who(a.who, bl, sections(), lane=_asker)
        out["who_asker_lane"] = _asker
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
    if a.promises:
        lane = a.promises.upper()
        res = promises(lane, bl)
        pr, cal = res["items"], res["calibration"]
        unmet = [x for x in pr if not x["kept"]]
        print("PROMISES for lane %s   %d `next:` lines" % (lane, len(pr)))
        if cal.get("informative"):
            print("  NULL-CALIBRATED: observed %.3f vs permutation null %.3f, z %+.2f -- informative"
                  % (cal["observed_kept_rate"], cal["null_mean"], cal["z"]))
            print("  %d not taken up by the following block" % len(unmet))
        else:
            print("  RATE WITHHELD: z %s against a permutation null -- NOT distinguishable from"
                  % cal.get("z"))
            print("  chance for this lane, so the count would score writing style, not follow-up.")
            print("  the flagged instances are still listed; read them, do not count them.")
        print("  heuristic: content-word overlap.  it FLAGS, it does not judge -- a promise")
        print("  deliberately abandoned reads the same as one forgotten.  say which it was.")
        for x in unmet:
            print("")
            print("  %s -> %s   line %d" % (x["promised_in"], x["judged_at"], x["line"]))
            print("     promised: %s" % x["next"])
            print("     matched : %s" % (x["matched"] or "nothing"))
        if not unmet:
            print("  none -- every `next:` was taken up by the block after it.")
        print("")

    if a.inbox:
        lane = a.inbox.upper()
        msgs = inbox(lane, bl)
        print("INBOX for lane %s   %d messages, WHOLE record, cursor ignored" % (lane, len(msgs)))
        print("  --brief only shows what arrived since your own last block.  this shows all of it,")
        print("  which is what you need after a reader repair (D-E25 recovered 13 dropped blocks).")
        for m in msgs:
            print("")
            print("  [%s] %s   line %d" % (m["from"], m["id"], m["line"]))
            for ln2 in re.sub(r"\s+", " ", m["text"]).strip().split(". "):
                if ln2.strip():
                    print("     %s" % ln2.strip())
        if not msgs:
            print("  none -- and an empty inbox is a CLAIM, not a default.")
    
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
        dur = b.get("arrived_during_your_last_round") or []
        if dur:
            print("  ARRIVED DURING YOUR LAST ROUND -- you could not have seen these when it opened:")
            print("  (writing your block jumps the cursor past them; shown once, here)")
            for m in dur:
                print("    [%s] %s   line %d" % (m["lane"], m["id"], m["line"]))
                print("       %s" % m["to_you"][:300])
            print("")
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
        asker = out.get("who_asker_lane")
        prov = {}
        for _h in out.get("who", []):
            prov[_h.get("provenance","UNCLASSIFIED")] = prov.get(_h.get("provenance","UNCLASSIFIED"), 0) + 1
        if not asker:
            # Without the asking lane there is no SELF cut and therefore no chronology-based
            # provenance classification.  The old renderer nevertheless converted a bag of
            # UNCLASSIFIED hits into "INDEPENDENT PRIOR WORK: 0" and then published the strong
            # negative "NO INDEPENDENT PRIOR WORK".  That was a fabricated absence claim.
            print("  PROVENANCE UNCLASSIFIED: pass a lane with --brief, --inbox, or --promises.")
            print("  No independent-prior count is identifiable without the asking lane.")
        elif prov:
            _ip = prov.get("INDEPENDENT_PRIOR", 0)
            _vac = sum(1 for _h in out.get("who", [])
                       if _h.get("ordering") == "VACUOUS_NO_SELF_HIT")
            print("  INDEPENDENT PRIOR WORK: %d      (self %d, echo-risk %d)"
                  % (_ip, prov.get("SELF", 0), prov.get("ECHO_RISK", 0)))
            if _vac:
                print("     of which %d rest on VACUOUS ordering: this lane has NO hit anywhere in"
                      % _vac)
                print("     the record for this term, so nothing was ordered against.  The label")
                print("     still means real work by others -- it does NOT mean a proven sequence.")
            # THE ASYMMETRY HAS A STRUCTURAL REASON, NOT JUST AN EMPIRICAL ONE (D-E41).
            # C-T68 measured that a zero is strong and a non-zero is weak.  H&R 8.6 says WHY:
            # "conditioning on the common effect Y of two independent causes A and E ALWAYS
            # induces a conditional association between A and E in at least one of the strata
            # of Y", while a special situation leaves the OTHER stratum conditionally
            # independent.  Here the causes are (A) the topic is worth studying and (E) this
            # lane raised it; the common effect is (Y) another lane wrote about it.  So the
            # HIT-EXISTS stratum always carries induced association -- that is ECHO_RISK --
            # and the NO-HIT stratum is the one that can stay clean.  H&R 8.5: adjusting means
            # treating selection as a treatment and requires POSITIVITY for it, which fails
            # here because a term only this lane uses has no chance of being written by
            # another lane at all.  So the class is REPORTED, never adjusted away.
            if _ip == 0:
                print("  >> NO INDEPENDENT PRIOR WORK.  a hit count is not prior work: on lane D")
                print("     terms six of eight distinctive terms were 100% the searcher's own")
                print("     blocks (C-T68 measured 67-91%).  SELF is not evidence, and ECHO_RISK")
                print("     may be a response to this lane rather than independent of it.")
                print("     A ZERO HERE IS THE STRONG READING (H&R 8.6): a hit is a common effect,")
                print("     and conditioning on it always induces association in that stratum; the")
                print("     no-hit stratum is the one that can stay clean.")
            if prov.get("ECHO_RISK"):
                print("     ECHO_RISK = another writer, but AFTER this lane first raised the term.")
                print("     The record cannot separate an independent result from a reply (C-T67).")
        for h in out["who"]:
            print("  %-22s %-8s %-8s line %-7s %-18s %s"
                  % (h["where"], h["ref"], h["stable_id"] or "-", h["line"],
                     h.get("provenance", "UNCLASSIFIED"), h["text"]))
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
                emb = v.get("embedded_hits", 0)
                print("  %-26s %4d hits   anchor=%s   %s%s"
                      % (src, v["hits"], v["anchor_term"], v["term_counts"],
                         ("   [%d/%d EMBEDDED in a longer word -- normal for a STEM query, "
                          "spurious for a PHRASE]" % (emb, v["hits"])) if emb else ""))
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
        print("")
        print("RECORD CHECK  %d blocks, %d OPEN, %d superseded"
              % (c["blocks"], len(c["problems"]), len(c.get("superseded", []))))
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
        for p in c.get("superseded", []):
            print("  [content supplied later] %-14s line %-6d %s  -> in %s"
                  % (p["id"], p["line"], p["problem"], p["superseded_by"]))
        if c.get("superseded"):
            print("  A-S85 IS RIGHT AND THE LABEL IS NOW PRECISE: the ORIGINAL BLOCK IS NEVER")
            print("  REPAIRED.  Under append-only it stays malformed forever, and the gap is CONTENT,")
            print("  not identity, so no alias fixes it.  A later block SUPPLIES the missing content")
            print("  in the same thread, so the information is not lost -- that is all this row says.")
            print("  Machine consumers: parse --json, not this text.  The human format changed once")
            print("  today and broke another lane's gate (C-KULLIYAT-T63); --json did not change.")
        for p in c["problems"]:
            print("  %-12s line %-7d %s" % (p["id"], p["line"], p["problem"]))
        if c["repeated_stable_ids"]:
            print("  threads (same stable id, more than one block):")
            for k, v in c["repeated_stable_ids"].items():
                print("    %-12s lines %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
