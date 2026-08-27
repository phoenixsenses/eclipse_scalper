# -*- coding: utf-8 -*-
"""H-U LANE ERRATA, ADDENDUM D -- what closing CT-016 found in Lane C's own Sec 471.

CT-016 was assigned to this lane because "the tick-regime axis is on C's front".  Closing it
turned up a defect in C's half, not A's.

C published, in Sec 471 section C:

    CARTEA_EXPONENTIAL_HOLDS_ONLY_ON_THE_LARGE_TICK_SYMBOL

on the strength of a fill-curve fit measured against QUEUE POSITION phi = x threshold.  But
Cartea Eq (8.1) is about DEPTH: P(order posted at depth delta is lifted) = exp(-kappa delta).
C's own H-U7 docstring stated the axis split correctly -- "Cartea's delta is DEPTH in ticks
and mine is QUEUE POSITION at the touch" -- and then the verdict token asserted something
about Cartea's form anyway.

Lane A measured Cartea's actual axis.  Re-derived here from A-S45's published five points,
with no re-run of A's driver:

    exponential fit   kappa 0.00956/bp   (A published 0.0097)   r2 0.9895
    power-law fit     exponent 0.0567                           r2 0.7499
    parametric discrimination at A's own n = 293 over 5 depths: 0.797 overall (chance 0.500)

So the exponential form is supported on Cartea's axis, on 15 symbols, with real discriminating
power -- the opposite of what this lane's token implied.  C's power law stands on C's axis and
says nothing about Eq (8.1).

  python -m tools.hu_errata_addendum_d --i-have-approval
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

OUT = "reports/research/hb4_liquidation_specialness_v1"
LEDGER = "IMMUTABLE_ERRATA_LEDGER_HU_LANE_ADDENDUM_D.json"

ENTRIES = [
    {
        "errata_id": "ERR-HU-015",
        "source_file": "reports/research/hb4_liquidation_specialness_v1/"
                       "HU6_HU7_PERMANENT_AND_NO_PROFITABLE_QUEUE_POSITION_V1.md",
        "source_section_or_line": "section C, token "
                                  "CARTEA_EXPONENTIAL_HOLDS_ONLY_ON_THE_LARGE_TICK_SYMBOL",
        "error_class": "PROSE_OVERCLAIM",
        "old_statement": "Cartea's exponential fill form holds only on the large-tick symbol; "
                         "the fill curve is a power law on BTC/ETH, so Eq (8.1)'s input "
                         "assumption fails there",
        "corrected_statement": "The measurement is on the QUEUE-POSITION axis (P(x >= phi) "
                               "per market-order arrival), not on Cartea Eq (8.1)'s DEPTH "
                               "axis, so it cannot speak about Eq (8.1) at all.  Lane A "
                               "measured the depth axis over an hourly horizon on 15 "
                               "symbols: re-derived from A-S45's published points, the "
                               "exponential fits r2 0.9895 against a power law's 0.7499, "
                               "kappa 0.00956/bp (A published 0.0097), and a parametric "
                               "discrimination test at A's own n picks the generating form "
                               "79.7 percent of the time against a 50 percent chance floor.  "
                               "Eq (8.1) is SUPPORTED on its own axis.  Replacement token, "
                               "scoped to the axis it was measured on: "
                               "QUEUE_POSITION_FILL_CURVE_IS_A_POWER_LAW.  The separate "
                               "finding that ORDER SIZE v has a power-law tail (Hill "
                               "1.30-1.44) is unaffected -- it is a statement about the "
                               "size distribution, not about the depth fill curve.",
        "numeric_outputs_affected": False,
        "primary_verdict_affected": True,
        "downstream_artifacts_affected": [
            "SYSTEM_STATE 471 section C (stable id C-T14)",
            "CONTRADICTION_REGISTER.md CT-016",
            "reports/atlas/ECLIPSE_CROSSWALK_V1.md section 3"],
    },
]


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    out = {"ledger_id": "IMMUTABLE_ERRATA_LEDGER_HU_LANE_ADDENDUM_D",
           "parent_ledger": "IMMUTABLE_ERRATA_LEDGER_HU_LANE_V1",
           "append_only": True,
           "rule": "published artifacts stay byte-identical; corrections are appended",
           "found_by": "tools/ct016_close_the_fill_form_contradiction.py",
           "note": "found in C's own half while closing a cross-lane contradiction",
           "entries": []}
    for e in ENTRIES:
        rec = dict(e)
        p = e["source_file"]
        if os.path.exists(p):
            with open(p, "rb") as fh:
                rec["source_sha256"] = hashlib.sha256(fh.read()).hexdigest()
            rec["source_exists"] = True
        else:
            rec["source_sha256"] = None
            rec["source_exists"] = False
        rec["errata_sha256"] = hashlib.sha256(
            json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        out["entries"].append(rec)
        print("%s  %-18s  verdict_affected=%s  src=%s"
              % (rec["errata_id"], rec["error_class"], rec["primary_verdict_affected"],
                 (rec["source_sha256"] or "MISSING")[:12]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, LEDGER), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("written %s (%d entries)" % (os.path.join(OUT, LEDGER), len(out["entries"])),
          flush=True)


if __name__ == "__main__":
    main()
