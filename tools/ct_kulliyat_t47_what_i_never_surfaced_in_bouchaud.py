# -*- coding: utf-8 -*-
"""C-KULLIYAT-T47 -- THE PASSAGES I NEVER SURFACED, IN THE BOOK I COULD ALWAYS READ.

Two rounds of source-checking (C-KULLIYAT-T44 Abergel, T45 Hernan & Robins) both ended the
same way: the citation held, and the source then supplied clauses the citing summary had
dropped.  Both were NUL-byte files -- invisible to grep entirely.

BOUCHAUD_TQP is the opposite case and it is the one this lane leans on hardest: 0 NUL bytes,
so grep could always OPEN it, but 2,246 ligatures, so grep could not always FIND in it.  The
failure mode is therefore different and worse to detect: I never misquoted a passage I found;
I never learned about the passages I did not find.

C-KULLIYAT-T46 established that ligature loss is SAME WORD, SAME BOOK -- pure loss with no
polysemy escape.  So every ligature-only hit in Bouchaud is a passage this lane could have
used and did not.

SCOPE.  This lane's load-bearing economic conclusion is about MAKER PROFITABILITY: no queue
position clears the real fee (C-T15/T16), the fee binds rather than adverse selection or
impact (C-T41), h_c exists only on the large-tick symbol and only at zero fee.  That argument
was built from Sec 17.3 and Sec 21.4, found by grepping terms like "queue position" and
"profitab".  So the question with the highest stakes is narrow and answerable:

    WHICH BOUCHAUD PASSAGES ABOUT MARKET-MAKER PROFITABILITY ARE LIGATURE-ONLY, AND WOULD
    THEREFORE NEVER HAVE APPEARED IN ANY SEARCH THIS LANE RAN?

Measured: per term, hits in BOUCHAUD only, raw vs normalised, and the ligature-only passages
printed in full so their content can be judged rather than counted.

No DB.  My own scope, my own artifact.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t47_what_i_never_surfaced_in_bouchaud --i-have-approval
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

from tools.corpus_text_v1 import normalise

SRC = os.path.join("data", "literature_v2", "text", "BOUCHAUD_TQP.txt")
OUT = "reports/atlas"

TERMS = ["profitab", "profit", "market-making", "market making", "efficient",
         "efficiency", "benefit", "compensat", "sufficient", "offset"]
CONTEXT = 300
MAX_SHOW = 3


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    raw = open(SRC, "rb").read().decode("utf-8", "replace")
    nrm = normalise(raw)
    print("BOUCHAUD_TQP  raw chars %d  ligatures %d  NUL %d"
          % (len(raw), sum(raw.count(c) for c in "ﬀﬁﬂﬃﬄﬅﬆ"),
             open(SRC, "rb").read().count(b"\x00")), flush=True)
    print(flush=True)

    rows = {}
    for term in TERMS:
        rx = re.compile(re.escape(term), re.I)
        n_true = len(rx.findall(nrm))
        n_raw = len(rx.findall(raw))
        rows[term] = {"true": n_true, "raw_grep": n_raw, "ligature_only": n_true - n_raw,
                      "pct_invisible": (100.0 * (n_true - n_raw) / n_true) if n_true else None}
        print("    %-14s true %5d   grep-visible %5d   LIGATURE-ONLY %5d  (%.1f%% invisible)"
              % (term, n_true, n_raw, n_true - n_raw,
                 rows[term]["pct_invisible"] or 0.0), flush=True)

    # the passages themselves: hits that exist ONLY in the normalised text
    print("\n=== LIGATURE-ONLY PASSAGES, printed so content can be judged ===", flush=True)
    shown = {}
    for term in ("profitab", "market-making", "compensat", "offset"):
        rx = re.compile(re.escape(term), re.I)
        raw_pos = set()
        # positions in the normalised text whose raw counterpart is not a match
        out = []
        for m in rx.finditer(nrm):
            i = m.start()
            # a hit is "ligature-only" if the same window in raw does not contain the term
            window_raw = raw[max(0, i - 40): i + len(term) + 40]
            if not rx.search(window_raw):
                seg = " ".join(nrm[max(0, i - CONTEXT): i + CONTEXT].split())
                out.append(seg)
        shown[term] = out[:MAX_SHOW]
        print("\n  --- %s : %d ligature-only hits ---" % (term, len(out)), flush=True)
        for s in out[:MAX_SHOW]:
            print("      ..." + s[:430], flush=True)

    res = {"source": SRC, "terms": rows, "sampled_passages": shown,
           "why": "ligature loss is same word same book (C-KULLIYAT-T46), so every "
                  "ligature-only hit is a passage this lane could have used and did not",
           "tokens": ["BOUCHAUD_IS_READABLE_BUT_NOT_FULLY_SEARCHABLE_BY_GREP",
                      "THE_FAILURE_IS_PASSAGES_NEVER_SURFACED_NOT_PASSAGES_MISQUOTED"],
           "ceiling": "MEASUREMENT_FIDELITY"}
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T47_UNSURFACED_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T47_UNSURFACED_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
