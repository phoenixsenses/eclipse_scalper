# -*- coding: utf-8 -*-
"""LANE D / D-E3 -- WHAT THE CORPUS ASKS US.

Mechanical extraction of every interrogative sentence in the 13-source corpus, so
that the lane's SELECTION of questions can be audited against the full population
rather than taken on trust.

Extraction is mechanical.  Selection is not -- the ledger in
`D_E3_CORPUS_QUESTION_LEDGER_V1.md` is a lane-D judgement over this output and
says so.  This file exists so anyone can re-filter it differently.

Reads only `data/literature_v2/text/*.txt` through `corpus_text_v1` (NUL-safe,
ligature-normalised).  Touches no market data.

Usage:  python tools/d_e3_corpus_question_extract_v1.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.corpus_text_v1 import bodies

OUT = "reports/atlas/D_E3_CORPUS_QUESTIONS_ALL_V1.json"

# A capitalised run of 15..200 chars ending in '?'.  Deliberately simple: a
# cleverer parser would be a selection step wearing an extraction costume.
QUESTION = re.compile(r"([A-Z][^.?!]{15,200}\?)")

# Terms that make a question a DEMAND ON THE ANALYST rather than exposition.
DESIGN = re.compile(
    r"\b(at risk|risk set|time scale|time zero|follow-?up|censor|truncat|"
    r"competing|recurrent|renewal|frailty|cluster|intensity|rate function|"
    r"martingale|proportional hazard|identif|assumption|estimat|unbiased|"
    r"horizon|duration|half-?life|how long|mean survival|median survival|"
    r"restricted mean|compensator|predictable)\b", re.I)


def key(q):
    return re.sub(r"[^a-z]", "", q.lower())[:45]


def main():
    out, counts = {}, {}
    for name, text in bodies().items():
        flat = re.sub(r"\s+", " ", text)
        seen, uniq = set(), []
        for q in QUESTION.findall(flat):
            k = key(q)
            if k in seen:
                continue
            seen.add(k)
            uniq.append({"q": q.strip(), "design_relevant": bool(DESIGN.search(q))})
        out[name] = uniq
        counts[name] = {"unique_questions": len(uniq),
                        "design_relevant": sum(1 for x in uniq if x["design_relevant"])}

    doc = {"study": "D-E3", "lane": "D",
           "method": "mechanical extraction; SELECTION into the ledger is a lane-D "
                     "judgement and is not mechanical",
           "reader": "corpus_text_v1 (NUL-safe, ligature-normalised)",
           "totals": {"unique_questions": sum(c["unique_questions"] for c in counts.values()),
                      "design_relevant": sum(c["design_relevant"] for c in counts.values())},
           "per_source": counts, "questions": out}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1, ensure_ascii=False))
    print("%-26s %8s %8s" % ("source", "unique", "design"))
    for k, v in sorted(counts.items(), key=lambda x: -x[1]["design_relevant"]):
        print("%-26s %8d %8d" % (k, v["unique_questions"], v["design_relevant"]))
    print("%-26s %8d %8d" % ("TOTAL", doc["totals"]["unique_questions"],
                             doc["totals"]["design_relevant"]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
