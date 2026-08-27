# -*- coding: utf-8 -*-
"""S66 -- every corpus read lane A made this session, re-run with the correct reader.

WHY
---
`tools/corpus_text_v1.py` (lane C, this estate) documents two extraction defects in
`data/literature_v2/text/*.txt` and measures what a naive `grep` loses to them:

    identifiability   0 / 78     100.0% missed
    competing risks  46 / 64      28.1%
    ...
    "An ABSENCE CLAIM made over this corpus with a raw reader is therefore worthless,
     and absence claims are exactly what this corpus is used for."

**Lane A read the corpus with `grep` for its entire session** -- A-S48 (saturation),
A-S49 (the square-root law), A-S50 (timing risk), A-S51 (the PSR), A-S52 (zero-sum),
A-S57 (direct trading costs), A-S58 (adverse selection).  Every one of those reads is
subject to the defect, and one of them produced a live ABSENCE CLAIM:

    A-S51:  "Kulliyatta Lo'nun seri-korelasyon duzeltmesi yok"
            (the corpus has no Lo-style serial-correlation correction to the Sharpe ratio)

That claim was made from `grep -n -i "annualiz\\|annualis"`.  If it is wrong, A-S51's
framing -- that the corpus offered the PSR and nothing else -- is wrong with it.

WHAT THIS DOES
--------------
1  Re-runs each term lane A actually searched, with the correct reader, and reports the
   ratio the naive read would have returned.
2  Re-tests the single absence claim with discriminating terms in BOTH the corpus's
   languages, as the module and the onboarding prompt both require.

Fenced: reads only.  `corpus_text_v1.py` is lane C's file and is imported, not edited.
"""

import io
import json
import sys

sys.path.insert(0, ".")
from tools.corpus_text_v1 import count           # lane C's reader, imported not modified

OUT = "reports/research/h2_response_shape_v1/S66_CORPUS_READS_AUDITED_V1.json"

# every term lane A grepped this session, with the study it fed
MINE = [
    ("saturat", "A-S48  the saturation statement that gave p = -1/2"),
    ("square-root law", "A-S49  the impact law"),
    ("execution styles", "A-S49  'holds for all execution styles'"),
    ("timing risk", "A-S50  Kissell's other half"),
    ("adverse selection", "A-S50/A-S58  the passive payoff term"),
    ("deton", "A-S47  LdP detoning"),
    ("Marcenko", "A-S47  the MP theorem"),
    ("annualiz", "A-S51  THE ABSENCE CLAIM was built on this"),
    ("annualis", "A-S51  the British spelling, also searched"),
    ("zero-sum", "A-S52  Harris's thesis"),
    ("why losers trade", "A-S52  Harris's imperative"),
    ("direct trading costs", "A-S57  the 0.1-1 bps calibration"),
    ("opportunity cost", "A-S50/A-S57  TQP 21.4's event (vi)"),
    ("queue position", "A-S58  the priority prediction"),
]

# the absence claim, re-tested with discriminating terms in both languages
ABSENCE = [
    "serial correlation", "autocorrelation", "autocorrelated",
    "Lo (2002)", "Lo, A", "iid returns", "independently and identically",
    "Sharpe ratio", "annualized Sharpe", "square root of time", "sqrt(T) rule",
    "time aggregation", "overlapping returns",
]


def naive(term):
    """What a raw grep would have returned: NUL files skipped, ligatures unmatched."""
    import os
    tot = 0
    for fn in sorted(os.listdir("data/literature_v2/text")):
        p = "data/literature_v2/text/" + fn
        raw = io.open(p, "rb").read()
        if b"\x00" in raw:                      # grep declares BINARY and skips
            continue
        tot += raw.decode("utf-8", "replace").lower().count(term.lower())
    return tot


def main():
    print("EVERY CORPUS READ LANE A MADE THIS SESSION, RE-RUN WITH THE CORRECT READER")
    print("  %-24s %8s %8s %8s   %s" % ("term", "correct", "naive", "recall", "fed"))
    rows = []
    worst = None
    for term, why in MINE:
        c = count(term)
        tot = sum(c.values()) if isinstance(c, dict) else int(c)
        nv = naive(term)
        rec = (100.0 * nv / tot) if tot else float("nan")
        flag = "  <-- LOST" if tot and nv < tot else ""
        print("  %-24s %8d %8d %7.1f%%   %s%s" % (term, tot, nv, rec, why, flag))
        rows.append({"term": term, "correct": tot, "naive": nv, "recall_pct": rec,
                     "study": why})
        if tot and (worst is None or rec < worst[1]):
            worst = (term, rec)

    print()
    if worst:
        print("  worst recall among lane A's own reads: %s at %.1f%%" % worst)
    lost = [r for r in rows if r["correct"] and r["naive"] < r["correct"]]
    print("  terms where the naive read lost hits: %d of %d" % (len(lost), len(rows)))

    print()
    print("THE ABSENCE CLAIM, RE-TESTED  (A-S51: 'the corpus has no Lo-style")
    print("serial-correlation correction to the Sharpe ratio')")
    print("  %-30s %8s %s" % ("discriminating term", "hits", "where"))
    hits = {}
    for t in ABSENCE:
        c = count(t)
        d = {k: v for k, v in c.items() if v} if isinstance(c, dict) else {}
        n = sum(d.values())
        hits[t] = d
        print("  %-30s %8d %s" % (t, n, ", ".join(sorted(d)[:4]) if d else "-"))

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S66_CORPUS_READS_AUDITED", "reads": rows,
         "absence_retest": {k: v for k, v in hits.items()}}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
