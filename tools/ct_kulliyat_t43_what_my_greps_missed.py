# -*- coding: utf-8 -*-
"""C-KULLIYAT-T43 -- WHAT A DAY OF RAW `grep` COST THIS LANE, MEASURED.

Lane D's onboarding prompt names a defect this lane committed all day:

    "NEVER `grep` data/literature_v2/text.  Use tools/corpus_text_v1.py -- 3 of 13 files carry
     NUL bytes and grep skips them silently; 10 of 13 carry ligatures.  A plain grep for
     'identifiability' finds 0 of 78 hits."

Every corpus search this lane ran today was a raw `grep`.  Two consequences, and the second is
worse than the first:

  NUL BYTES     ABERGEL_LOB.txt carries 1018 of them.  `grep` reported "Binary file matches"
                once, in this lane's own output, and the lane read that line and moved on.  So
                every multi-file search silently EXCLUDED Abergel -- the source that H-T6 / C-T9
                is built on.
  LIGATURES     a search for a term containing ff/fi/fl silently under-returns.

And the claim that depended on it: C-T40 swept four files with a raw grep for open-question
phrasings and published that the corpus names THREE open questions.  Lane C's other session
(C-T43 in the shared log) swept all thirteen sources properly and tabulated FIFTEEN demands.
Three against fifteen is the cost, and it is an ABSENCE CLAIM -- exactly the kind the module's
own docstring says is "worthless" from a raw reader.

MEASURED HERE, with tools/corpus_text_v1.py:
  (1) per-source NUL and ligature counts, so the exclusion is quantified rather than asserted;
  (2) raw-grep recall against the correct reader for the terms THIS LANE actually searched
      today, not the module's illustrative list;
  (3) C-T40's open-question sweep re-run over all thirteen sources with the correct reader,
      and the count compared against the three it published.

No new market data.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t43_what_my_greps_missed --i-have-approval
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

from tools.corpus_text_v1 import bodies, load, normalise

OUT = "reports/atlas"
TEXT_DIR = os.path.join("data", "literature_v2", "text")

# the terms this lane actually grepped today, verbatim
MY_TERMS = ["refill", "fill probability", "signature plot", "queue position",
            "open question", "deserves further", "remains unclear", "efficiency",
            "identifiability", "different", "confirm", "first"]

# C-T40's sweep, as it was run: four files, raw grep, these phrasings
CT40_PHRASES = ["open question", "remains to be", "is still debated", "not yet been",
                "deserves further", "would be interesting to", "remains unclear",
                "is an open", "still poorly understood", "call for further"]
CT40_FILES = ["BOUCHAUD_TQP", "CARTEA_AHFT", "ABERGEL_LOB", "HASBROUCK_EMM"]
CT40_PUBLISHED_COUNT = 3
OTHER_LANE_COUNT = 15


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    res = {"defect": "raw grep over data/literature_v2/text",
           "named_by": "LANE_ONBOARDING_PROMPTS_V1 section 6", "ceiling":
           "MEASUREMENT_FIDELITY"}

    print("=== (1) per-source NUL bytes and ligatures ===", flush=True)
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    src = {}
    for p in files:
        raw = open(p, "rb").read()
        nul = raw.count(b"\x00")
        txt = raw.decode("utf-8", "replace")
        lig = sum(txt.count(c) for c in "ﬀﬁﬂﬃﬄﬅﬆ")
        stem = os.path.basename(p)[:-4]
        src[stem] = {"nul": nul, "ligatures": lig, "bytes": len(raw)}
        flag = ("GREP SKIPS THIS FILE" if nul else "")
        print("    %-22s nul %6d  ligatures %6d   %s" % (stem, nul, lig, flag), flush=True)
    res["sources"] = src
    n_nul = sum(1 for v in src.values() if v["nul"])
    n_lig = sum(1 for v in src.values() if v["ligatures"])
    print("    %d of %d files invisible to grep; %d of %d carry ligatures"
          % (n_nul, len(src), n_lig, len(src)), flush=True)

    print("\n=== (2) raw-grep recall on the terms THIS LANE searched ===", flush=True)
    B = bodies()
    recall = {}
    for term in MY_TERMS:
        true_n = sum(len(re.findall(re.escape(term), t, re.I)) for t in B.values())
        # what a raw grep would have seen: skip NUL files, no ligature folding
        raw_n = 0
        for p in files:
            b = open(p, "rb").read()
            if b.count(b"\x00"):
                continue
            raw_n += len(re.findall(re.escape(term), b.decode("utf-8", "replace"), re.I))
        pct = (100.0 * (1 - raw_n / true_n)) if true_n else None
        recall[term] = {"true": true_n, "raw_grep": raw_n, "missed_pct": pct}
        print("    %-20s true %5d   raw grep %5d   MISSED %s"
              % (term, true_n, raw_n, ("%.1f%%" % pct) if pct is not None else "n/a"),
              flush=True)
    res["recall_on_my_terms"] = recall

    print("\n=== (3) C-T40's open-question sweep, re-run correctly ===", flush=True)
    hits_all, hits_ct40 = {}, {}
    for stem, t in B.items():
        n_all = sum(len(re.findall(re.escape(ph), t, re.I)) for ph in CT40_PHRASES)
        if n_all:
            hits_all[stem] = n_all
        if stem in CT40_FILES:
            hits_ct40[stem] = n_all
    tot_all = sum(hits_all.values())
    tot_ct40 = sum(hits_ct40.values())
    print("    correct reader, ALL 13 sources : %d passages across %d sources"
          % (tot_all, len(hits_all)), flush=True)
    for k in sorted(hits_all, key=lambda k: -hits_all[k]):
        print("        %-22s %4d" % (k, hits_all[k]), flush=True)
    print("    correct reader, C-T40's 4 files: %d" % tot_ct40, flush=True)
    print("    C-T40 PUBLISHED: %d open questions.  the other lane's proper sweep: %d demands."
          % (CT40_PUBLISHED_COUNT, OTHER_LANE_COUNT), flush=True)
    res["ct40_resweep"] = {"all_sources_passages": tot_all, "by_source": hits_all,
                           "ct40_four_files_passages": tot_ct40,
                           "ct40_published_count": CT40_PUBLISHED_COUNT,
                           "other_lane_count": OTHER_LANE_COUNT}

    res["tokens"] = [
        "EVERY_CORPUS_SEARCH_THIS_LANE_RAN_TODAY_WAS_A_RAW_GREP",
        "ABERGEL_WAS_SILENTLY_EXCLUDED_FROM_EVERY_MULTI_FILE_SEARCH",
        "THE_BINARY_FILE_MATCHES_LINE_APPEARED_IN_MY_OWN_OUTPUT_AND_I_READ_PAST_IT",
        "CT40_OPEN_QUESTION_SWEEP_IS_AN_ABSENCE_CLAIM_FROM_A_RAW_READER",
        "THREE_AGAINST_FIFTEEN",
    ]
    print("\n=== TOKENS ===", flush=True)
    for t in res["tokens"]:
        print("    " + t, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T43_GREP_COST_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T43_GREP_COST_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
