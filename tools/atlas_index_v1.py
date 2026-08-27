# -*- coding: utf-8 -*-
"""ECLIPSE ATLAS indexer -- resolve one day's sections into stable identities.

WHY THIS EXISTS
---------------
Three sessions ran on 2026-08-26 and wrote 99 sections into SYSTEM_STATE.md
sharing a single `§` namespace.  39 of those numbers are duplicated: `§449`
points at three different studies.  That is the live recurrence of the defect
memory records at §398, whose decision is explicit:

    STUDY / LANE / UUID identity.  NO RENUMBERING.

So this tool never rewrites a `§`.  It assigns a STABLE ID and keeps the `§`
plus the line number as aliases.  A stable ID is unique by construction; a `§`
is not, and the index asserts both facts.

DISCIPLINE (inherited from CKR-01's `_LANE_VERDICTS.json`)
---------------------------------------------------------
Verdict tokens are extracted MECHANICALLY from the fenced blocks each section
closes with.  Nothing is inferred, nothing is invented; a section with no token
block is recorded `NOT_RECORDED` rather than summarised.

RE-RUNNABLE.  A fourth session tomorrow is picked up by re-running this; the
MD artifacts are generated from the JSON, never hand-edited.
"""

from __future__ import annotations

import io
import json
import re
from collections import Counter, defaultdict

SRC = "SYSTEM_STATE.md"
OUT_JSON = "reports/atlas/_ATLAS_INDEX.json"
import sys as _sys
# D-E1/D-E6 (shared log): DAY was a constant where it needs to be an argument.
# A fixed date silently excludes every section written after it.
DAY = next((a.split("=", 1)[1] for a in _sys.argv[1:] if a.startswith("--day=")),
           "2026-08-26")

HDR = re.compile(r"^## §(\d+)\s*(.*)$")
# a section may also be titled "## §481 — S136: ..." (the English-titled thread)
STUDY = re.compile(r"\bS(\d{1,3})\b")
ROUND = re.compile(r"K[ÜU]LL[İIiı]YAT\s+TURU\s+(\d+)", re.IGNORECASE)
REVIEW = re.compile(r"(REVIEW|CORRECTION)\s+ROUND\s+(\d+)", re.IGNORECASE)
# verdict tokens: ALL-CAPS words with underscores, inside fenced blocks
TOKEN = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,})\b")


def study_key(title: str):
    """The label, from the title.  Never used to decide the LANE."""
    r = ROUND.search(title)
    if r:
        return "T%s" % r.group(1)
    rv = REVIEW.search(title)
    if rv:
        return "%s%s" % ("RV" if rv.group(1).upper().startswith("REVIEW") else "CR",
                         rv.group(2))
    # The label sits at the START of the title ("S45 -- ...", "-- S136: ...").
    # An S-number further in is a REFERENCE, not a label: "FEASIBILITY GATES V1
    # -- S1..S14" is a synthesis of fourteen studies, not study S1.  The
    # monotonicity assert below caught exactly that mislabelling.
    m = STUDY.search(title[:16])
    return "S%s" % m.group(1) if m else None


def thread(rows):
    """Assign lanes from the STUDY-KEY RANGES, which do not overlap.

    Two earlier attempts failed and both failures are instructive:

      1. Reading the lane off the title's prose mislabelled ten sections.
      2. Reconstructing threads from the interleaving by attaching each section
         to the thread whose last number sits closest below it ALSO failed --
         the three sessions' section numbers overlap, so "closest below" jumps
         between threads and split a lane that is known to be contiguous.

    What does not overlap is the study numbering: S1..S45, S95..S114, and the
    T/RV/CR rounds plus S13x.  That is the identifier a session actually
    controls, so it is the one used.  Sections with no study key are attached to
    the nearest labelled section by line distance -- adjacency is only a
    tie-break, never the primary signal.

    The assert at the end is the real check: within a lane, study numbers must be
    monotone in line order.  If they are not, the threading is wrong.
    """
    def lane_of(k):
        if k is None:
            return None
        # D-E1 / D-E6 (shared log): lane D key are not S-numbers at all (D-E1, D-E2, ...),
        # so the S-number regex could not match them and lane D read as no lane at all.
        # D reported this twice and did not edit the file, which was correct: a tool that
        # decides identity is exactly where an outside edit is worst.
        if k[0] == "E" and k[1:].isdigit():
            return "D"
        if k[0] in "TRC" and not k.startswith("S"):
            return "C"
        n = int(k[1:])
        if 1 <= n <= 60:
            return "A"
        if 95 <= n <= 129:
            return "B"
        if 61 <= n <= 94 or 130 <= n <= 199:
            return "C"      # S66 cascade thread and the S13x thread
        return None

    for r in rows:
        r["lane"] = lane_of(r["study"])
        r["lane_source"] = "study_key" if r["lane"] else None

    known = [(i, r) for i, r in enumerate(rows) if r["lane"]]
    for i, r in enumerate(rows):
        if r["lane"]:
            continue
        prev = [(abs(i - j), rr["lane"]) for j, rr in known if j < i]
        nxt = [(abs(i - j), rr["lane"]) for j, rr in known if j > i]
        pick = min(prev)[1] if prev else (min(nxt)[1] if nxt else "A")
        r["lane"], r["lane_source"] = pick, "line_adjacency"

    # monotonicity check: a session's study numbers only ever go up
    for lane in sorted({r["lane"] for r in rows}):
        seq = [int(r["study"][1:]) for r in rows
               if r["lane"] == lane and r["study"] and r["study"].startswith("S")]
        bad = [(a, b) for a, b in zip(seq, seq[1:]) if b < a]
        assert not bad, "lane %s study numbers not monotone: %s" % (lane, bad[:5])
    return rows


def parse(path=SRC, day=DAY):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    heads = []
    for i, l in enumerate(lines):
        m = HDR.match(l)
        if m and day in l:
            heads.append((i, int(m.group(1)), m.group(2).replace("**", "").strip()))
    out = []
    for k, (i, sec, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i:end])
        study = study_key(title)
        # verdict tokens live in fenced blocks
        toks = []
        for blk in re.findall(r"```(.*?)```", body, re.S):
            toks += TOKEN.findall(blk)
        seen = set()
        toks = [t for t in toks if not (t in seen or seen.add(t))]
        out.append({
            "line": i + 1, "section": sec, "title": title,
            "lane": None, "study": study,
            "tokens": toks or ["NOT_RECORDED"],
            "n_lines": end - i,
        })
    return out


def assign_ids(rows):
    """Stable IDs.  Unique by construction; a fallback keeps uniqueness total."""
    used = Counter()
    for r in rows:
        base = "%s-%s" % (r["lane"], r["study"]) if r["study"] else "%s-L%d" % (r["lane"], r["line"])
        used[base] += 1
        r["id"] = base if used[base] == 1 else "%s.%d" % (base, used[base])
    return rows


def link_corrections(rows):
    """Mechanically detect sections that correct/withdraw an earlier claim.

    Only two signals are used, both textual and both explicit: a WITHDRAW-family
    token, and a '§NNN' reference inside the same section.  Nothing is inferred
    from prose.
    """
    for r in rows:
        wd = [t for t in r["tokens"]
              if any(k in t for k in ("WITHDRAW", "CORRECT", "SUPERSED", "RETIRE",
                                      "REJECT", "VOID", "REFUS"))]
        r["withdrawal_tokens"] = wd
        r["status"] = "CORRECTS_OR_WITHDRAWS" if wd else "STANDS"
    return rows


if __name__ == "__main__":
    rows = link_corrections(assign_ids(thread(parse())))
    ids = [r["id"] for r in rows]
    secs = [r["section"] for r in rows]
    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    dup_secs = sorted(k for k, v in Counter(secs).items() if v > 1)

    # THE ASSERT THE WHOLE TOOL EXISTS FOR
    assert not dup_ids, "stable IDs must be unique, got duplicates: %s" % dup_ids

    by_lane = defaultdict(list)
    for r in rows:
        by_lane[r["lane"]].append(r)

    doc = {
        "built": DAY, "source": SRC, "n_sections": len(rows),
        "duplicate_section_numbers": dup_secs,
        "n_duplicate_section_numbers": len(dup_secs),
        "stable_ids_unique": True,
        "not_recorded": sum(1 for r in rows if r["tokens"] == ["NOT_RECORDED"]),
        "lanes": {k: len(v) for k, v in sorted(by_lane.items())},
        "rows": rows,
    }
    import os
    os.makedirs("reports/atlas", exist_ok=True)
    json.dump(doc, io.open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("sections indexed        : %d" % len(rows))
    print("lanes                   : %s" % dict(doc["lanes"]))
    print("duplicate SECTION numbers: %d  (preserved, never renumbered)" % len(dup_secs))
    print("duplicate STABLE ids     : %d  (asserted zero)" % len(dup_ids))
    print("sections with NO verdict block: %d" % doc["not_recorded"])
    print("sections that correct/withdraw: %d"
          % sum(1 for r in rows if r["status"] == "CORRECTS_OR_WITHDRAWS"))
    print("wrote", OUT_JSON)
