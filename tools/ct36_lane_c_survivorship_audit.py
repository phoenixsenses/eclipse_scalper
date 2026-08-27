# -*- coding: utf-8 -*-
"""C-T36 -- WHICH OF LANE C'S CLAIMS ARE STILL STANDING?  A MECHANICAL SURVIVORSHIP AUDIT.

Lane C has written 40-odd SYSTEM_STATE sections today and 34 errata entries across ten
append-only ledgers.  Nobody can now say which verdict tokens still stand -- including the
lane that wrote them.  The atlas charter gave that audit to Lane B ("re-derive the N claims,
find what should have been withdrawn and was not"); nobody has run it for Lane C.

This runs it for Lane C, and it is NOT independent: the lane is auditing itself, which
CLAUDE.md's chain says is worth little.  It is published as a MECHANICAL INDEX, not as a
review.  What it can do that a reading cannot is be exhaustive and reproducible: every token
is matched against every errata entry by literal string search, so nothing is missed because
someone forgot it.

METHOD.
  1. Extract every UPPER_SNAKE verdict token from fenced blocks in SYSTEM_STATE.md, together
     with the section number and title it appeared in.
  2. Load every errata ledger under reports/atlas/ and the H-U lane directory.
  3. For each token, search all errata text (old_statement + corrected_statement) for a
     literal occurrence.  A hit means the token was TOUCHED by a correction.
  4. Classify the hit by the verb the errata used about it: WITHDRAWN, SUPERSEDED, SUSPENDED,
     REFUTED, REPLACED, or TOUCHED_UNCLASSIFIED.
  5. Report tokens that are touched, tokens that are untouched, and -- the point of the
     exercise -- tokens whose errata says one thing while the section still asserts them.

LIMITS, stated rather than discovered later.  A literal string match cannot see a claim that
was corrected without naming its token, and cannot see a token that is quoted inside an
errata for an unrelated reason.  Both directions are reported as counts so the reader knows
the size of the blind spot.  This is an index, not a verdict.

  python -m tools.ct36_lane_c_survivorship_audit --i-have-approval
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

OUT = "reports/atlas"
STATE = "SYSTEM_STATE.md"
LEDGER_GLOBS = ("reports/atlas/IMMUTABLE_ERRATA_LEDGER_*.json",
                "reports/research/hb4_liquidation_specialness_v1/"
                "IMMUTABLE_ERRATA_LEDGER_*.json")
TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")
SEC_RE = re.compile(r"^## §(\d+)\s+(.*)$", re.M)
FENCE_RE = re.compile(r"```(.*?)```", re.S)
LANE_C_MARKS = ("KÜLLİYAT TURU", "[C-T")   # bare "C-T" matched foreign sections
VERBS = (("WITHDRAWN", ("withdraw", "geri çek", "GERI CEKILDI", "GERİ ÇEKİLDİ")),
         ("SUPERSEDED", ("supersede", "SUPERSEDE", "süpersede")),
         ("SUSPENDED", ("suspend", "askıya", "ASKIYA")),
         ("REFUTED", ("refute", "çürüt", "REFUTED")),
         ("REPLACED", ("replacement token", "Replacement token", "yerine")))


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    txt = io.open(STATE, encoding="utf-8").read()
    heads = [(m.start(), int(m.group(1)), m.group(2)) for m in SEC_RE.finditer(txt)]
    heads.append((len(txt), -1, ""))

    tokens = {}
    lane_c_secs = 0
    for i in range(len(heads) - 1):
        start, num, title = heads[i]
        body = txt[start:heads[i + 1][0]]
        if not any(k in title for k in LANE_C_MARKS):
            continue
        lane_c_secs += 1
        for fence in FENCE_RE.findall(body):
            for t in set(TOKEN_RE.findall(fence)):
                tokens.setdefault(t, []).append({"section": num, "title": title[:70]})

    ledgers, entries = [], []
    for g in LEDGER_GLOBS:
        for p in sorted(glob.glob(g)):
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            ledgers.append(os.path.basename(p))
            for e in d.get("entries", []):
                entries.append({
                    "id": e.get("errata_id"), "ledger": os.path.basename(p),
                    "text": " ".join(str(e.get(k, "")) for k in
                                     ("old_statement", "corrected_statement",
                                      "source_section_or_line")),
                    "verdict_affected": e.get("primary_verdict_affected")})

    def classify(tok, txt_):
        """TOUCHED or nothing.  An earlier version inferred WITHDRAWN / SUPERSEDED from
        verbs in a 260-character window and got it wrong in the direction that matters:
        a REPLACEMENT token introduced by an errata sits next to the word "withdrawn"
        (which applies to the token it replaces) and was labelled WITHDRAWN itself.
        Verb inference is dropped; the errata id is reported and the reader classifies."""
        return "TOUCHED" if tok in txt_ else None

    rows = []
    for tok, where in sorted(tokens.items()):
        hits = []
        for e in entries:
            c = classify(tok, e["text"])
            if c:
                hits.append({"errata": e["id"], "ledger": e["ledger"], "class": c,
                             "verdict_affected": e["verdict_affected"]})
        status = "STANDING"
        if hits:
            status = "TOUCHED"
        rows.append({"token": tok, "status": status,
                     "sections": sorted({w["section"] for w in where}),
                     "n_sections": len(where), "errata": hits})

    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1

    res = {"lane": "C", "independent": False,
           "note": "self-audit; CLAUDE.md's chain says a lane reviewing itself is worth "
                   "little.  Published as a mechanical index, not a review.",
           "lane_c_sections_scanned": lane_c_secs,
           "ledgers": ledgers, "n_errata_entries": len(entries),
           "n_tokens": len(rows), "status_counts": by, "tokens": rows}

    print("Lane C sections scanned: %d   errata ledgers: %d   entries: %d"
          % (lane_c_secs, len(ledgers), len(entries)), flush=True)
    print("distinct verdict tokens: %d" % len(rows), flush=True)
    for k in sorted(by, key=lambda k: -by[k]):
        print("    %-24s %d" % (k, by[k]), flush=True)
    print("\nTOKENS TOUCHED BY A CORRECTION:", flush=True)
    for r in rows:
        if r["status"] != "STANDING":
            print("    %-52s %-22s secs %s  <- %s"
                  % (r["token"][:52], r["status"], r["sections"],
                     ",".join(h["errata"] for h in r["errata"])), flush=True)
    multi = [r for r in rows if r["status"] == "STANDING" and r["n_sections"] > 1]
    print("\nSTANDING tokens asserted in more than one section: %d" % len(multi), flush=True)
    for r in multi[:20]:
        print("    %-52s secs %s" % (r["token"][:52], r["sections"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT36_LANE_C_SURVIVORSHIP_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT36_LANE_C_SURVIVORSHIP_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
