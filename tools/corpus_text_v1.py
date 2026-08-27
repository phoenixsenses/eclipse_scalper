# -*- coding: utf-8 -*-
"""CORPUS_TEXT_V1 -- the only correct way to read `data/literature_v2/text/*.txt`.

Two extraction defects make plain `grep` / `open().read()` unreliable on this
corpus, and both were measured and published before this module existed:

  NUL BYTES     3 of 13 files carry them (ABERGEL_LOB 1018, HERNAN_ROBINS 240,
                SURVIVAL_STK4080 79).  `grep` calls such a file BINARY and skips
                it, and `grep -c` returns 1 whatever the content.
  LIGATURES     10 of 13 files carry `ff fi fl ffi ffl st` as single glyphs --
                13,146 of them.  A search for "identifiability" therefore misses
                `identifiability`, and a search for "efficient" misses
                `efficient`.

Measured recall of a naive grep against the true (normalised) count, 2026-08-27:

    identifiability   0 / 78     100.0% missed
    positivity        1 / 160     99.4%
    confidence       24 / 350     93.1%
    coefficient      45 / 218     79.4%
    specific        176 / 658     73.3%
    first           573 / 1730    66.9%
    effect         1175 / 3365    65.1%
    flow            241 / 674     64.2%
    efficient       233 / 592      60.6%
    significant     100 / 241      58.5%
    profit          357 / 579      38.3%
    competing risks  46 / 64       28.1%

An ABSENCE CLAIM made over this corpus with a raw reader is therefore worthless,
and absence claims are exactly what this corpus is used for.

The corrective already existed twice in the repo and neither copy was reusable:
`research_s100_corpus_absence_claim_audit_v1.py` is NUL-safe but ligature-BLIND,
and `research_s120_cross_lane_claim_audit_v1.py` handles both -- privately, in
its own file.  This module is S120's map lifted into one importable place so the
next reader cannot get it wrong by default.  Neither script is modified here;
S100's ligature blindness is reported, not fixed (lane separation).

Usage:
    from tools.corpus_text_v1 import load, bodies, count, absence
    n = count("identifiability")              # {source: hits} over the corpus
    absence(["queue-reactive"])               # refuses to answer from raw text
"""
from __future__ import annotations

import glob
import os

TEXT_DIR = os.path.join("data", "literature_v2", "text")

# Lifted verbatim from research_s120_cross_lane_claim_audit_v1.py so the two
# cannot drift.  ﬅ/ﬆ are the long-s ligatures; they occur in scanned front matter.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
             "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
DASHES = {"‐": "-", "‑": "-", "­": ""}


def normalise(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    for k, v in DASHES.items():
        text = text.replace(k, v)
    return text


def load(path: str) -> str:
    """NUL-safe read + ligature and hyphen normalisation.  Never use open().read()."""
    with open(path, "rb") as fh:
        return normalise(fh.read().decode("utf-8", "replace"))


def bodies(text_dir: str = TEXT_DIR) -> dict:
    """{source_stem: normalised_text} for the whole corpus."""
    return {os.path.basename(p)[:-4]: load(p)
            for p in sorted(glob.glob(os.path.join(text_dir, "*.txt")))}


def count(term: str, text_dir: str = TEXT_DIR, case_sensitive: bool = False) -> dict:
    """Per-source occurrence count of `term`, on normalised text."""
    out = {}
    for name, body in bodies(text_dir).items():
        hay = body if case_sensitive else body.lower()
        needle = term if case_sensitive else term.lower()
        n = hay.count(needle)
        if n:
            out[name] = n
    return out


def absence(terms, text_dir: str = TEXT_DIR) -> dict:
    """Evidence for an 'the corpus does not treat X' claim.

    Returns the per-term, per-source counts and a boolean.  An absence claim is
    only publishable when `supported` is True AND the terms are discriminating --
    this function checks the first and cannot check the second.
    """
    hits = {t: count(t, text_dir) for t in terms}
    total = sum(sum(v.values()) for v in hits.values())
    return {"terms": list(terms), "hits": hits, "total_hits": total,
            "supported": total == 0,
            "reader": "corpus_text_v1 (NUL-safe, ligature-normalised)",
            "caveat": "discriminating-term choice is NOT checked here"}


if __name__ == "__main__":
    import sys
    probe = sys.argv[1:] or ["identifiability", "competing risks", "positivity"]
    for t in probe:
        c = count(t)
        print("%-24s total=%-6d %s" % (t, sum(c.values()), c))
