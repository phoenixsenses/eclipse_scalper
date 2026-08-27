# -*- coding: utf-8 -*-
"""C-KULLIYAT-T50 -- ARE THE TWO SIDES OF MY FINE BALANCE MEASURED IN THE SAME UNIT?

C-T51, from the other session on this letter, published a unit warning: on BTC 10.75% of orders
occupy several consecutive aggTrade rows in the same direction at median depth 7, so any
statistic over consecutive aggTrades is partly measuring the INSIDE OF ONE ORDER.  Collapsing
(ts_ms, side) runs changed a probability by 0.41 there.

That warning was addressed to lane A.  It lands harder here, because this lane's headline test
is an IDENTITY BETWEEN TWO EXPONENTS and an inspection of my own sources shows they were
estimated under DIFFERENT EVENT DEFINITIONS:

    kappa - chi   C-T27, line "new = (ts != ts) | (eps != eps)"  ->  (ts_ms, side) ORDER COLLAPSE
    gamma         C-T19, recorded in C-T28 as                    ->  200 ms MERGE

C-T29 then combined them into `difference = (kappa-chi) - (1-gamma)/2` and rejected the fine
balance on 2 of 3 symbols; C-KULLIYAT-T49 published a corpus bridge that reads the SIGN of that
difference.  If gamma is index-dependent -- and it must be, since a lag of 10 ORDERS and a lag
of 10 x 200 ms BINS are different amounts of time -- then the identity was tested across two
clocks and the difference carries a unit artefact of unknown size.

THIS ROUND MEASURES THAT AND NOTHING ELSE.  Same days, same lag grid, same fit range, same
estimator, same debiasing; the ONLY thing varied is the event definition:

    A  200 ms merge            (as C-T19 / C-T28)
    B  (ts_ms, side) collapse  (as C-T27, the unit kappa-chi is in)

PREREGISTERED, fixed before any number is read:
  Q1  gamma_A vs gamma_B per symbol, and the gap in units of the recovery sd (0.023-0.029)
  Q2  the fine balance recomputed with gamma_B, i.e. BOTH SIDES IN ONE UNIT
  Q3  does the VERDICT of C-T29 (rejected on BTC and SOL) survive the unit repair?
  Q4  C(1) under both definitions, since H-T8 struck C(1) as convention-dependent and
      C-KULLIYAT-T48 declared Bouchaud Eq (17.14) uncomputable for exactly that reason

I WILL NOT READ Q3's FLAG BEFORE Q1 AND Q2 ARE PRINTED.  A change of verdict is a finding
either way; an unchanged verdict is a robustness result and NOT a licence to call the direction
established -- gamma's fit-range dependence is a separate, still-open defect.

DB is opened READ-ONLY through the existing drivers.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t50_is_my_headline_test_a_unit_mismatch --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import hb4_is_a_liquidation_special as B4

DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
MERGE_MS = 200
OUT = "reports/atlas"
CT29 = "reports/atlas/CT29_FINE_BALANCE_V1.json"
# C-T28's measured shrinkage of the gamma estimator toward 0.45, applied identically to both arms
SHRINK_TARGET = 0.45
SHRINK_SD = {"BTCUSDT": 0.02278, "ETHUSDT": 0.02513, "SOLUSDT": 0.02865}


def acf_sums(x, lags):
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    return {L: (float(np.sum(xc[L:] * xc[:-L])), den) for L in lags if len(xc) > L + 10}


def fit_gamma(cs):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and cs[L] > 0]
    if len(ls) < 3:
        return None
    a = np.polyfit(np.log(ls), np.log([cs[L] for L in ls]), 1)
    return -float(a[0])


def debias(g, sym):
    """Invert C-T28's measured shrinkage toward 0.45.  Identical on both arms, so it cannot
    manufacture a difference between them; carried only so the numbers sit on C-T29's scale."""
    d = json.load(io.open(CT29, encoding="utf-8"))["per_symbol"][sym]
    k = (d["gamma_debiased"] - SHRINK_TARGET) / (d["gamma_fitted"] - SHRINK_TARGET)
    return SHRINK_TARGET + k * (g - SHRINK_TARGET)


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    ct29 = json.load(io.open(CT29, encoding="utf-8"))
    res = {"days": list(DAYS), "lags": list(LAGS), "fit_range": [FIT_LO, FIT_HI],
           "merge_ms": MERGE_MS, "varied": "EVENT DEFINITION ONLY",
           "arms": {"A": "200 ms DEAD-TIME THINNING (C-T19's actual operation, tools/ct35 line 150)",
                    "B": "(ts_ms, side) collapse (as C-T27, the unit kappa-chi is in)"},
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    print("=== Q1 / Q4  gamma and C(1) under both event definitions ===", flush=True)
    print("%-9s | %-28s | %-28s" % ("symbol", "A  200 ms merge", "B  (ts_ms,side) collapse"),
          flush=True)
    print("%-9s | %8s %8s %9s | %8s %8s %9s" %
          ("", "n", "C(1)", "gamma", "n", "C(1)", "gamma"), flush=True)

    for sym in H2.SYMBOLS:
        acc = {"A": {L: [0.0, 0.0] for L in LAGS}, "B": {L: [0.0, 0.0] for L in LAGS}}
        n_tot = {"A": 0, "B": 0}
        for day in DAYS:
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            if ts is None or len(ts) < 10000:
                continue
            # ARM B -- (ts_ms, side) run collapse, exactly C-T27's line
            newb = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            sb = eps[np.flatnonzero(newb)].astype(float)
            # ARM A -- C-T19's actual operation, read out of tools/ct35 line 150:
            # a 200 ms DEAD-TIME THINNING of the order series, not a bin merge.
            ots0 = ts[np.flatnonzero(newb)]
            keep = np.concatenate([[True], np.diff(ots0) >= MERGE_MS])
            sa = eps[np.flatnonzero(newb)][keep].astype(float)
            del ts, px, eps, qty, newb, ots0, keep
            for arm, s in (("A", sa), ("B", sb)):
                n_tot[arm] += len(s)
                for L, (nu, de) in acf_sums(s, LAGS).items():
                    acc[arm][L][0] += nu
                    acc[arm][L][1] += de
            del sa, sb

        row = {}
        for arm in ("A", "B"):
            cs = {L: (acc[arm][L][0] / acc[arm][L][1])
                  for L in LAGS if acc[arm][L][1] > 0}
            g = fit_gamma(cs)
            row[arm] = {"n": n_tot[arm], "C1": cs.get(1),
                        "gamma_fitted": g,
                        "gamma_debiased": debias(g, sym) if g is not None else None,
                        "C_of_l": cs}
        res["per_symbol"][sym] = row
        print("%-9s | %8d %+8.4f %9.4f | %8d %+8.4f %9.4f"
              % (sym, row["A"]["n"], row["A"]["C1"], row["A"]["gamma_debiased"],
                 row["B"]["n"], row["B"]["C1"], row["B"]["gamma_debiased"]), flush=True)

    print("\n=== Q1  the gap, in units of the gamma recovery sd ===", flush=True)
    for sym in H2.SYMBOLS:
        r = res["per_symbol"][sym]
        gap = r["B"]["gamma_debiased"] - r["A"]["gamma_debiased"]
        sd = SHRINK_SD[sym]
        r["gamma_gap_B_minus_A"] = gap
        r["gamma_gap_in_sd"] = gap / sd
        print("    %-9s  gamma_A %.4f  gamma_B %.4f  gap %+.4f  = %+.2f sd"
              % (sym, r["A"]["gamma_debiased"], r["B"]["gamma_debiased"], gap, gap / sd),
              flush=True)

    print("\n=== Q2  the fine balance with BOTH SIDES IN ONE UNIT (arm B) ===", flush=True)
    print("    kappa-chi is already arm B; only gamma moves.", flush=True)
    print("    %-9s %10s %10s %10s %8s %s"
          % ("symbol", "kappa-chi", "beta_pred", "difference", "z", "vs C-T29"), flush=True)
    for sym in H2.SYMBOLS:
        old = ct29["per_symbol"][sym]
        r = res["per_symbol"][sym]
        gB = r["B"]["gamma_debiased"]
        beta = (1.0 - gB) / 2.0
        # gamma sd carried from C-T28's recovery; beta sd = gamma sd / 2
        sd_b = SHRINK_SD[sym] / 2.0
        diff = old["kappa_minus_chi"] - beta
        sd_d = float(np.hypot(old["kappa_minus_chi_sd"], sd_b))
        z = diff / sd_d
        r["repaired"] = {"beta_predicted": beta, "difference": diff,
                         "sd_of_difference": sd_d, "z": z,
                         "rejects_at_2sigma": abs(z) > 2.0}
        print("    %-9s %10.4f %10.4f %+10.4f %+8.2f  (was %+.4f, z %+.2f)"
              % (sym, old["kappa_minus_chi"], beta, diff, z,
                 old["difference"], old["z"]), flush=True)

    print("\n=== Q3  does C-T29's verdict survive the unit repair? ===", flush=True)
    old_rej = sorted(s for s in H2.SYMBOLS
                     if ct29["per_symbol"][s]["rejects_composite_at_2sigma"])
    new_rej = sorted(s for s in H2.SYMBOLS
                     if res["per_symbol"][s]["repaired"]["rejects_at_2sigma"])
    old_side = {s: ("TRENDING_HARD" if ct29["per_symbol"][s]["difference"] < 0
                    else "MEANREVERT_BOON") for s in H2.SYMBOLS}
    new_side = {s: ("TRENDING_HARD" if res["per_symbol"][s]["repaired"]["difference"] < 0
                    else "MEANREVERT_BOON") for s in H2.SYMBOLS}
    print("    rejected before: %s" % (old_rej or ["none"]), flush=True)
    print("    rejected after : %s" % (new_rej or ["none"]), flush=True)
    print("    sides before   : %s" % old_side, flush=True)
    print("    sides after    : %s" % new_side, flush=True)
    flipped = [s for s in H2.SYMBOLS if old_side[s] != new_side[s]]
    print("    symbols whose SIDE flipped: %s" % (flipped or ["none"]), flush=True)
    res["verdict_survives"] = (old_rej == new_rej)
    res["sides_unchanged"] = (not flipped)
    res["rejected_before"], res["rejected_after"] = old_rej, new_rej
    res["side_flips"] = flipped

    res["tokens"] = ["THE_FINE_BALANCE_WAS_TESTED_ACROSS_TWO_EVENT_DEFINITIONS",
                     "KAPPA_CHI_IS_ORDER_COLLAPSED_AND_GAMMA_WAS_200MS_MERGED",
                     "ONLY_THE_EVENT_DEFINITION_WAS_VARIED",
                     "VERDICT_SURVIVES" if res["verdict_survives"] else "VERDICT_CHANGES",
                     "SIDES_UNCHANGED" if res["sides_unchanged"] else "A_SIDE_FLIPPED",
                     "GAMMA_FIT_RANGE_DEPENDENCE_REMAINS_A_SEPARATE_OPEN_DEFECT"]
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T50_UNIT_MISMATCH_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T50_UNIT_MISMATCH_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
