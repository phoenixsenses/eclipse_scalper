# -*- coding: utf-8 -*-
"""LANE C CHARTER ITEM 2 -- RECONCILE zeta, gamma, delta, kappa-chi AND p IN ONE TABLE.

The charter asks one question outright: is Lane A's p ~= -0.5 the same quantity as Lane C's
kappa - chi?  Answering it needs the corpus identity that links them, and Sec 13.4.3 states
it in closed form:

    "the propagator model predicts chi = 1 - beta ~= 0.75 and kappa = 1.  Note, however,
     that the model correctly predicts the DIFFERENCE chi - kappa = 0.25 which governs the
     scaling of Kyle's lambda with T."

so  kappa - chi = 1 - (1 - beta) = beta,  and the fine balance (Eq 13.17) gives
beta = (1 - gamma)/2.  The chain is  kappa - chi = beta = (1 - gamma)/2.

That closes something this lane left open.  Sec 481 / ERR-HU-010 recorded
FINE_BALANCE_REMAINS_OPEN because a merge scan cannot identify the signature plot.  But the
fine balance does not have to be reached through G(l) at all: gamma is measured from the
sign autocorrelation decay (C-T19, stable under de-fragmentation), kappa - chi is measured
from aggregate impact scaling (A-S30 and C-T21, two lanes, two clocks, two estimators), and
the identity ties them.  Two independent statistics, one prediction.

ALL NUMBERS BELOW ARE PUBLISHED VALUES, taken from the two lanes' own sections.  Nothing is
re-measured; no DB is opened.  This is arithmetic on the record plus the corpus identity.

WHAT p IS, AND WHY IT IS NOT kappa - chi.  A-S40 defines
    f(h) = E[sign(eps_hat) * dm(t -> t+h)] / E[|dm(t -> t+h)|]
and measures f ~ h^p.  Its -1/2 comes from the DENOMINATOR: once the response R saturates,
E|dm| keeps growing as sqrt(h), so f ~ 1/sqrt(h).  A said so in its own section.  That
exponent would be -1/2 in any diffusive market with ANY saturating response -- it carries no
impact-law information by itself.  kappa - chi is an impact-law exponent and equals beta.
They are different objects, opposite in sign, and different in magnitude.

Where p DOES carry beta is in its DEVIATION from -1/2 at long horizons: if R(h) ~ h^(-a)
after the peak and E|dm| ~ h^(1/2), then p_long = -a - 1/2, so a = -p_long - 1/2.  Reported
as a consistency check, not a test -- different statistic, different horizon.

ESTIMATION.  Ceiling: RECONCILIATION.  Read-only.

  python -m tools.ct_reconcile_the_five_exponents --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
OUT = "reports/atlas"

# ---- published values, with their source ids -------------------------------
GAMMA_C = {"BTCUSDT": 0.373, "ETHUSDT": 0.369, "SOLUSDT": None}   # C-T19, 200 ms merge
GAMMA_C_SRC = "C-T19 (SYSTEM_STATE 481); SOL struck as unstable 0.286 -> 0.740"

KX_A = {"BTCUSDT": 0.255, "ETHUSDT": 0.361, "SOLUSDT": 0.193}     # A-S30, T>=20
KX_A_SRC = "A-S30 (SYSTEM_STATE 458), trade time, T>=20, r2 0.974/0.988/0.785"
KX_C = {"BTCUSDT": 0.300, "ETHUSDT": 0.250, "SOLUSDT": 0.100}     # C-T21
KX_C_SRC = "C-T21 (SYSTEM_STATE 483), market-order time, collapse grid search"

ZETA_A = {"BTCUSDT": 0.416, "ETHUSDT": 0.439, "SOLUSDT": 0.495}   # A-S30, outer region
ZETA_A_SRC = "A-S30 section 4, OUTER-region exponent, flat over a 50x range in T"
ZETA_C_STATUS = ("C-T20 fitted the WHOLE binned range and got a rising psi(T); "
                 "withdrawn by ERR-HU-012.  A's outer-region estimator is the correct one.")

P_A = {"BTCUSDT": -0.4085, "ETHUSDT": -0.4952, "SOLUSDT": -0.5075}
P_A_LONG = {"BTCUSDT": -0.6719, "ETHUSDT": -0.9276, "SOLUSDT": None}
P_A_SRC = "A-S40 (SYSTEM_STATE 468), capture ratio on a holdout predictor"

BOOK = {"kappa_minus_chi_band": (0.25, 0.30),
        "identity_1": "kappa - chi = beta            (Sec 13.4.3, explicit)",
        "identity_2": "beta = (1 - gamma)/2          (Eq 13.17, the fine balance)",
        "delta": "delta = gamma for a METAORDER      (Eq 16.16)"}


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rows = []
    print("=== kappa - chi  vs  the fine balance, two lanes ===", flush=True)
    print("    sym       gamma(C)  beta_pred  kx(A)   kx(C)   kx_mean   |mean-pred|",
          flush=True)
    for s in SYMS:
        g = GAMMA_C[s]
        pred = (1.0 - g) / 2.0 if g is not None else None
        a, c = KX_A[s], KX_C[s]
        mean = (a + c) / 2.0
        gap = abs(mean - pred) if pred is not None else None
        rows.append({"symbol": s, "gamma_C": g, "beta_predicted": pred,
                     "kappa_minus_chi_A": a, "kappa_minus_chi_C": c,
                     "kappa_minus_chi_mean": mean, "abs_gap": gap})
        print("    %-9s %8s  %9s  %6.3f  %6.3f  %7.3f   %s"
              % (s, "%.3f" % g if g else "struck",
                 "%.4f" % pred if pred else "n/a", a, c, mean,
                 "%.4f" % gap if gap is not None else "n/a"), flush=True)

    vals = [KX_A[s] for s in SYMS] + [KX_C[s] for s in SYMS]
    lo, hi = min(vals), max(vals)
    print("    ALL SIX kappa-chi measurements: min %.3f  max %.3f  spread %.1fx  mean %.3f"
          % (lo, hi, hi / max(lo, 1e-9), sum(vals) / len(vals)), flush=True)
    print("    book band %.2f-%.2f: centre replicates, TIGHTNESS DOES NOT"
          % BOOK["kappa_minus_chi_band"], flush=True)

    print("=== is p the same quantity as kappa - chi? ===", flush=True)
    pa = []
    for s in SYMS:
        pl = P_A_LONG[s]
        a_exp = (-pl - 0.5) if pl is not None else None
        pa.append({"symbol": s, "p_all": P_A[s], "p_long": pl,
                   "implied_R_decay_a": a_exp,
                   "beta_predicted": (1.0 - GAMMA_C[s]) / 2.0 if GAMMA_C[s] else None})
        print("    %-9s p(all) %+.4f   p(long) %s   => R ~ h^-a with a = %s   "
              "(beta_pred %s)"
              % (s, P_A[s], "%+.4f" % pl if pl else "n/a",
                 "%.3f" % a_exp if a_exp is not None else "n/a",
                 "%.4f" % ((1 - GAMMA_C[s]) / 2) if GAMMA_C[s] else "n/a"), flush=True)
    print("    p's -1/2 is the DENOMINATOR's diffusion (E|dm| ~ sqrt(h)), not an impact law.",
          flush=True)
    print("    ANSWER: NO.  p ~ -0.5 and kappa-chi ~ +0.27 are different objects, opposite "
          "in sign.", flush=True)

    print("=== zeta and delta ===", flush=True)
    for s in SYMS:
        print("    %-9s zeta(A, outer region) %.3f    delta: NOT MEASURABLE on public data"
              % (s, ZETA_A[s]), flush=True)
    print("    ZETA_IS_NOT_DELTA was published by A-S30 and re-derived by C in ERR-HU-013 "
          "-- the same conclusion paid for twice.", flush=True)

    out = {"book": BOOK,
           "sources": {"gamma": GAMMA_C_SRC, "kappa_minus_chi_A": KX_A_SRC,
                       "kappa_minus_chi_C": KX_C_SRC, "zeta_A": ZETA_A_SRC,
                       "zeta_C": ZETA_C_STATUS, "p": P_A_SRC},
           "fine_balance_table": rows,
           "kappa_minus_chi_all_six": {"min": lo, "max": hi,
                                       "mean": sum(vals) / len(vals)},
           "p_table": pa,
           "zeta_A_outer": ZETA_A,
           "answer_to_charter": "p is NOT kappa - chi.  p's -1/2 comes from the diffusive "
                                "denominator E|dm| ~ sqrt(h); kappa - chi is an impact-law "
                                "exponent and equals beta by Sec 13.4.3.",
           "tokens": ["KAPPA_MINUS_CHI_EQUALS_BETA_BY_THE_BOOK",
                      "FINE_BALANCE_CLOSED_WITHOUT_MEASURING_G_OF_L",
                      "TWO_LANES_TWO_STATISTICS_ONE_PREDICTION",
                      "P_IS_NOT_KAPPA_MINUS_CHI",
                      "KAPPA_MINUS_CHI_CENTRE_REPLICATES_TIGHTNESS_DOES_NOT",
                      "ZETA_IS_NOT_DELTA_PAID_FOR_TWICE",
                      "DELTA_IS_NOT_MEASURABLE_ON_PUBLIC_DATA"],
           "ceiling": "RECONCILIATION"}
    print("=== TOKENS ===", flush=True)
    for t in out["tokens"]:
        print("    " + t, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "EXPONENT_RECONCILIATION_V1.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print("written %s/EXPONENT_RECONCILIATION_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
