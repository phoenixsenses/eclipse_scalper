# -*- coding: utf-8 -*-
"""H-U / LANE C ERRATA, ADDENDUM E -- C-T24 supersedes C-T21's kappa-chi, and that
UNDOES C-T23's closure of the fine balance.

C-T24 ran the corpus's own estimator -- Kyle's lambda from the INNER region,
Lambda(T) ~ T^-(kappa-chi), Sec 11.4 -- on Lane C's data pipeline, i.e. A-S30's estimator on
C's data.  Result, T >= 20, inner-median cut:

                this run    A-S30   gap      C-T21 (collapse)   gap
    BTCUSDT      0.2245     0.255   0.031          0.300        0.076
    ETHUSDT      0.3786     0.361   0.018          0.250        0.129
    SOLUSDT      0.2032     0.193   0.010          0.100        0.103

Three of three land on A.  The cross-lane disagreement recorded in Sec 486 was
ESTIMATOR-driven, and C-T21's global collapse grid was the outlier -- the same defect
ERR-HU-012 recorded for zeta, now found in kappa-chi too.

AND THE CORRECTION MAKES SEC 486 WORSE, NOT BETTER.  With C's value replaced:

    symbol   gamma   beta_pred=(1-gamma)/2   kx pooled (A + corrected C)   gap
    BTC      0.373        0.3135                    0.2398               0.074
    ETH      0.369        0.3155                    0.3698               0.054

Before the correction the gaps were 0.036 and 0.010.  The earlier agreement was partly luck:
C-T21's contaminated value pulled BTC up and ETH down, toward the prediction.  Worse, BTC and
ETH now have gamma agreeing to 0.004 -- so the fine balance predicts essentially the SAME beta
for both -- while their measured kappa-chi differ by 0.13.  That is a tension the earlier
table concealed.

  python -m tools.hu_errata_addendum_e --i-have-approval
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

OUT = "reports/atlas"
LEDGER = "IMMUTABLE_ERRATA_LEDGER_HU_LANE_ADDENDUM_E.json"

ENTRIES = [
    {
        "errata_id": "ERR-HU-016",
        "source_file": "tools/hu14_is_the_ladder_just_one_scaling_function.py",
        "source_section_or_line": "collapse grid search producing kappa and chi separately",
        "error_class": "CODE_ERROR",
        "old_statement": "kappa - chi = 0.300 (BTC) / 0.250 (ETH) / 0.100 (SOL), published "
                         "as KAPPA_MINUS_CHI_CONFIRMED_A_THIRD_TIME in C-T21",
        "corrected_statement": "kappa - chi is defined by Sec 11.4 from the INNER region: "
                               "Lambda(T) is the slope of R on dV as |dV| -> 0, and "
                               "Lambda(T) ~ Lambda(1) T^-(kappa-chi).  A global collapse over "
                               "the whole binned curve mixes F's linear and concave parts -- "
                               "the defect ERR-HU-012 already recorded for zeta.  Re-run with "
                               "the book's estimator on the SAME data (C-T24): 0.2245 / "
                               "0.3786 / 0.2032, which lands within 0.010-0.031 of A-S30's "
                               "0.255 / 0.361 / 0.193 on all three symbols.  C-T21's values "
                               "are SUPERSEDED.  The cross-lane gap in Sec 486 was "
                               "estimator-driven.",
        "numeric_outputs_affected": True,
        "primary_verdict_affected": True,
        "downstream_artifacts_affected": [
            "SYSTEM_STATE 483 (C-T21)", "SYSTEM_STATE 486 (C-T23)",
            "reports/atlas/EXPONENT_RECONCILIATION_V1.json"],
    },
    {
        "errata_id": "ERR-HU-017",
        "source_file": "tools/ct_reconcile_the_five_exponents.py",
        "source_section_or_line": "fine_balance_table; token "
                                  "FINE_BALANCE_CLOSED_WITHOUT_MEASURING_G_OF_L",
        "error_class": "PROSE_OVERCLAIM",
        "old_statement": "the fine balance kappa-chi = (1-gamma)/2 is satisfied to within "
                         "0.010-0.036 by two lanes and two statistics; Sec 481's open item "
                         "is closed",
        "corrected_statement": "That table used C-T21's superseded kappa-chi (ERR-HU-016).  "
                               "With C's corrected value the gaps GROW to 0.074 (BTC) and "
                               "0.054 (ETH), and a sharper tension appears: gamma agrees "
                               "between BTC and ETH to 0.004, so (1-gamma)/2 predicts "
                               "essentially the same beta for both, while the measured "
                               "kappa-chi now differ by 0.13 (0.2398 vs 0.3698).  The "
                               "earlier agreement was partly an artefact of the contaminated "
                               "value pulling both symbols toward the prediction.  "
                               "FINE_BALANCE_CLOSED... is WITHDRAWN; the item returns to "
                               "OPEN, better specified than before: "
                               "FINE_BALANCE_REOPENED_SAME_GAMMA_DIFFERENT_KAPPA_MINUS_CHI.",
        "numeric_outputs_affected": True,
        "primary_verdict_affected": True,
        "downstream_artifacts_affected": ["SYSTEM_STATE 486 (C-T23)", "SYSTEM_STATE 481"],
    },
    {
        "errata_id": "ERR-HU-018",
        "source_file": "SYSTEM_STATE.md",
        "source_section_or_line": "A-S30 section 2 table vs C-T24 T_min table",
        "error_class": "NUMERIC_ERROR",
        "old_statement": "no single lane published a wrong number here; this entry RECORDS an "
                         "unexplained cross-lane discrepancy rather than correcting one",
        "corrected_statement": "sigma_tilde(1), the volatility per trade, differs about "
                               "two-fold between lanes on the same instruments: A-S30 "
                               "reports 0.2058 / 0.2559 / 0.4074, C-T24 measures 0.0991 / "
                               "0.1436 / 0.2870.  It matters because T_min = "
                               "(tick/sigma_tilde)^2 is QUADRATIC in it: A's SOL T_min is "
                               "10.4, C's is 22.4, so A's T >= 20 cutoff sits ABOVE SOL's "
                               "floor on A's numbers and BELOW it on C's.  SOL's kappa-chi "
                               "is separately unstable in C-T24 across the inner-cut choice "
                               "(+0.203 median vs -0.155 quartile).  Both SOL figures should "
                               "be treated as UNSUPPORTED until sigma_tilde(1) is reconciled. "
                               " Neither lane's number is asserted to be the wrong one here.",
        "numeric_outputs_affected": False,
        "primary_verdict_affected": True,
        "downstream_artifacts_affected": ["SYSTEM_STATE 458 (A-S30)", "SYSTEM_STATE 486"],
    },
]


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    out = {"ledger_id": "IMMUTABLE_ERRATA_LEDGER_HU_LANE_ADDENDUM_E",
           "parent_ledger": "IMMUTABLE_ERRATA_LEDGER_HU_LANE_V1",
           "append_only": True,
           "rule": "published artifacts stay byte-identical; corrections are appended",
           "found_by": "tools/ct24_kyle_lambda_the_books_way.py",
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
        print("%s  %-16s  verdict_affected=%s  src=%s"
              % (rec["errata_id"], rec["error_class"], rec["primary_verdict_affected"],
                 (rec["source_sha256"] or "MISSING")[:12]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, LEDGER), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("written %s (%d entries)" % (os.path.join(OUT, LEDGER), len(out["entries"])),
          flush=True)


if __name__ == "__main__":
    main()
