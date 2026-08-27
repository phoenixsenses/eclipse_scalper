# -*- coding: utf-8 -*-
"""C-KULLIYAT-T46 -- A-S67 LANDS ON MY RECALL TABLE: SPLIT THE LOSS BY CHANNEL.

C-KULLIYAT-T43 published a recall table -- "raw grep missed 73% of `efficiency`, 80% of
`confirm`, 67% of `first`" -- as the cost of a day of raw `grep` over the corpus.  A-S67 then
measured something that lands directly on it:

    THE_45_MISSING_SATURAT_HITS_WERE_ALL_A_HOMONYM_IN_HERNAN_ROBINS
    A_RAW_RECALL_STATISTIC_OVERSTATES_LOSS_ON_A_POLYSEMOUS_TERM

A raw recall number counts occurrences, not usable occurrences.  If the recovered hits are a
different sense of the word, the practical loss is smaller than the percentage says.

BUT THE CAVEAT DOES NOT APPLY EVENLY, and that is what this round adds.  The loss has TWO
channels and they are qualitatively different:

    LIGATURE LOSS   the hit is in a file grep CAN read; it was missed only because the glyph is
                    `ﬁ` rather than `fi`.  Same word, same book, same page.  There is no
                    polysemy question here at all -- it is pure, unimpeachable loss.

    NUL-FILE LOSS   the hit is in one of ABERGEL_LOB / HERNAN_ROBINS / SURVIVAL_STK4080, which
                    grep skips whole.  The recovered hits are in a DIFFERENT BOOK, so the sense
                    can differ -- and A-S67's `saturat` case is exactly this: all 45 in Hernan &
                    Robins, all a homonym.  This is where the caveat bites.

So the fix to my table is not a discount; it is a DECOMPOSITION.  Per term:
    loss_ligature + loss_nul = true_total - raw_grep     (checked, not assumed)
and the NUL component is reported PER SOURCE so a reader can judge sense at a glance instead of
taking a percentage on trust.

No DB, no market data.  My own artifact.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t46_recall_loss_by_channel --i-have-approval
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

from tools.corpus_text_v1 import normalise

TEXT_DIR = os.path.join("data", "literature_v2", "text")
OUT = "reports/atlas"

TERMS = ["refill", "fill probability", "signature plot", "queue position",
         "open question", "deserves further", "remains unclear", "efficiency",
         "identifiability", "different", "confirm", "first", "saturat"]


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    raw_txt, norm_txt, has_nul = {}, {}, {}
    for p in files:
        b = open(p, "rb").read()
        stem = os.path.basename(p)[:-4]
        has_nul[stem] = b.count(b"\x00") > 0
        d = b.decode("utf-8", "replace")
        raw_txt[stem] = d
        norm_txt[stem] = normalise(d)

    nul_files = [s for s in has_nul if has_nul[s]]
    print("NUL-bearing sources (grep skips these whole): %s" % ", ".join(sorted(nul_files)),
          flush=True)
    print(flush=True)
    print("%-18s %7s %7s | %8s %8s | %s"
          % ("term", "true", "rawgrep", "lig_loss", "nul_loss", "nul loss by source"),
          flush=True)

    rows = {}
    for term in TERMS:
        rx = re.compile(re.escape(term), re.I)
        true_total = sum(len(rx.findall(norm_txt[s])) for s in norm_txt)
        raw_grep = sum(len(rx.findall(raw_txt[s])) for s in raw_txt if not has_nul[s])
        # ligature loss: readable files, hits that only the normalised form finds
        lig = sum(len(rx.findall(norm_txt[s])) - len(rx.findall(raw_txt[s]))
                  for s in norm_txt if not has_nul[s])
        # NUL loss: everything in the skipped files
        by_src = {s: len(rx.findall(norm_txt[s])) for s in nul_files
                  if len(rx.findall(norm_txt[s]))}
        nul = sum(by_src.values())
        ok = (lig + nul) == (true_total - raw_grep)
        rows[term] = {"true": true_total, "raw_grep": raw_grep,
                      "ligature_loss": lig, "nul_loss": nul, "nul_by_source": by_src,
                      "decomposition_exact": ok,
                      "pct_missed": (100.0 * (1 - raw_grep / true_total)) if true_total else None,
                      "loss_channel": ("NUL_DOMINATED" if nul > lig else
                                       "LIGATURE_DOMINATED" if lig > nul else
                                       "NONE" if (lig + nul) == 0 else "MIXED")}
        src = "  ".join("%s %d" % (s.split("_")[0], n) for s, n in sorted(by_src.items()))
        print("%-18s %7d %7d | %8d %8d | %s%s"
              % (term, true_total, raw_grep, lig, nul, src, "" if ok else "   [SUM MISMATCH]"),
              flush=True)

    print(flush=True)
    lig_only = [t for t, r in rows.items() if r["loss_channel"] == "LIGATURE_DOMINATED"]
    nul_dom = [t for t, r in rows.items() if r["loss_channel"] == "NUL_DOMINATED"]
    print("LIGATURE-DOMINATED (same word, same book -- A-S67's caveat does NOT apply): %s"
          % ", ".join(lig_only), flush=True)
    print("NUL-DOMINATED (recovered from a different book -- sense must be checked): %s"
          % ", ".join(nul_dom), flush=True)

    res = {"trigger": "A-S67: a raw recall statistic overstates loss on a polysemous term",
           "corrects": "C-KULLIYAT-T43 recall table (SYSTEM_STATE 516)",
           "nul_sources": sorted(nul_files), "terms": rows,
           "ligature_dominated": lig_only, "nul_dominated": nul_dom,
           "all_decompositions_exact": all(r["decomposition_exact"] for r in rows.values()),
           "tokens": [
               "RECALL_LOSS_HAS_TWO_CHANNELS_AND_ONLY_ONE_IS_POLYSEMY_EXPOSED",
               "LIGATURE_LOSS_IS_SAME_WORD_SAME_BOOK_AND_UNIMPEACHABLE",
               "NUL_LOSS_COMES_FROM_A_DIFFERENT_BOOK_AND_NEEDS_A_SENSE_CHECK",
               "THE_DECOMPOSITION_IS_EXACT_NOT_APPORTIONED"],
           "ceiling": "MEASUREMENT_FIDELITY"}
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T46_RECALL_CHANNELS_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T46_RECALL_CHANNELS_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
