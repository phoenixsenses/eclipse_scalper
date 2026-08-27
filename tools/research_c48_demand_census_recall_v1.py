r"""LANE C, round 48 -- redo my own demand census with the reader that was built afterwards.

C-T43 swept all thirteen corpus sources for demand constructions ("must be", "requires that",
"cannot be ... unless", "care must be taken", "the key question") and published 437 methodological
passages and a fifteen-item table of what the corpus asks this estate.

C-T46 then measured that the loader C-T43 used was WRONG: hand-rolled, carrying 4 of the 6
ligatures and no hyphen normalisation. I recorded the 437 as a LOWER BOUND and moved on. This
round pays that debt, because the estate's rule is that an absence claim names its reader, and a
census is a presence claim plus an implicit absence.

A SECOND DEFECT OF C-T43, WHICH IS WHY THIS FILE EXISTS AT ALL. The sweep was an ad-hoc heredoc.
Nothing was saved. A census whose extractor is not on disk cannot be re-run by anyone, including
its author, which is the same failure the estate's derived-index rule exists to prevent. The
regex is reproduced here VERBATIM as C-T43 used it, so both readers can be run over one extractor
and the difference attributed to the reader alone.

`--who corpus demand census` returns nothing in English and, in Turkish, returns D-E3 (914
interrogative sentences) and this lane's own C-T43. D-E3's object is QUESTIONS and mine is
DEMANDS -- complementary, not duplicative, so nothing is inherited. But D-E3 also predates
corpus_text_v1, so the recall figure measured here is reported to lane D rather than applied to
their extraction: their file is not touched.

READERS COMPARED
    C-T43's own      NUL-stripped, 4 ligatures (ff fi fl ffi), no hyphen normalisation
    corpus_text_v1   6 ligatures (ff fi fl ffi ffl st) + hyphen/soft-hyphen normalisation
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import corpus_text_v1 as CT                                    # noqa: E402

TEXT_DIR = ROOT / "data" / "literature_v2" / "text"
OUT = ROOT / "reports" / "atlas"

# --- reproduced VERBATIM from the C-T43 heredoc
DEMAND = re.compile(
    r'\b(must be|one must|one should|it is (?:essential|crucial|important|necessary)|'
    r'requires? (?:that|knowledge|the|a|an)|cannot be .{0,40}(?:unless|without)|'
    r'care must be taken|the key question|the crucial|should always|'
    r'it is (?:therefore )?(?:vital|critical)|'
    r'needs? to be (?:measured|estimated|checked|tested)|before (?:one|we|any) can)\b', re.I)
METH = re.compile(
    r'\b(data|sample|estimat|measure|test|assum|model|bias|null|significan|infer|'
    r'identif|calibrat|volatil|impact|liquid|cost|risk|price|order|trade)\b', re.I)

# --- C-T43's own loader, reproduced verbatim
OLD_LIG = {'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi'}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_c_t43(path):
    t = open(path, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    for k, v in OLD_LIG.items():
        t = t.replace(k, v)
    return re.sub(r"\s+", " ", t)


def load_correct(path):
    return re.sub(r"\s+", " ", CT.load(path))


def sweep(text):
    hits = []
    for m in DEMAND.finditer(text):
        s = text[max(0, m.start() - 260):m.end() + 320]
        if METH.search(s):
            hits.append((m.start(), m.group(1).lower(), s.strip()))
    return hits


def key(h):
    """dedupe key: the demand phrase plus a normalised slice of its context"""
    return (h[1], re.sub(r"[^a-z ]", "", h[2].lower())[:90])


def main() -> int:
    per, tot_old, tot_new = {}, 0, 0
    only_new_examples = []
    for p in sorted((TEXT_DIR).glob("*.txt")):
        name = p.stem
        a = sweep(load_c_t43(str(p)))
        b = sweep(load_correct(str(p)))
        ka, kb = {key(x) for x in a}, {key(x) for x in b}
        missed = kb - ka
        per[name] = {"c_t43_reader": len(a), "correct_reader": len(b),
                     "delta": len(b) - len(a),
                     "unique_to_correct_reader": len(missed),
                     "recall_of_c_t43": (round(len(ka & kb) / len(kb), 4) if kb else None)}
        tot_old += len(a)
        tot_new += len(b)
        if missed:
            for x in b:
                if key(x) in missed and len(only_new_examples) < 8:
                    only_new_examples.append({"source": name, "phrase": x[1],
                                              "passage": x[2][:420]})
    art = {"study": "C-T48", "lane": "C", "utc": _utc(),
           "what": ("C-T43's demand census re-run through corpus_text_v1; the extractor is "
                    "identical and reproduced verbatim, so the difference is the READER alone"),
           "debt_paid": "C-T46 recorded C-T43's 437 as a lower bound and did not re-run it",
           "second_defect_fixed": ("C-T43's sweep was an ad-hoc heredoc and was never saved; "
                                   "a census whose extractor is not on disk cannot be re-run"),
           "readers": {"c_t43": "NUL-stripped, 4 ligatures, no hyphen normalisation",
                       "correct": "corpus_text_v1: 6 ligatures + hyphen/soft-hyphen"},
           "totals": {"c_t43_reader": tot_old, "correct_reader": tot_new,
                      "delta": tot_new - tot_old,
                      "c_t43_recall": (round(tot_old / tot_new, 4) if tot_new else None)},
           "per_source": per, "examples_only_the_correct_reader_finds": only_new_examples}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C48_DEMAND_CENSUS_RECALL_V1.json").write_text(json.dumps(art, indent=2),
                                                          encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("%-28s %12s %12s %8s %12s" % ("source", "C-T43 reader", "correct", "delta", "C-T43 recall"))
    for k, v in sorted(per.items()):
        w("%-28s %12d %12d %8s %12s" % (k, v["c_t43_reader"], v["correct_reader"],
                                        ("+%d" % v["delta"]) if v["delta"] else "-",
                                        v["recall_of_c_t43"]))
    t = art["totals"]
    w("%-28s %12d %12d %8s %12s" % ("TOTAL", t["c_t43_reader"], t["correct_reader"],
                                    "+%d" % t["delta"], t["c_t43_recall"]))
    w("")
    w("PASSAGES ONLY THE CORRECT READER FINDS (first few):")
    for x in only_new_examples[:5]:
        w("  [%s] (%s) ...%s..." % (x["source"][:14], x["phrase"], x["passage"][:300]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
