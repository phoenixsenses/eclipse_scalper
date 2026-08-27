# -*- coding: utf-8 -*-
"""CT-016 -- CLOSING THE FILL-CURVE CONTRADICTION BETWEEN LANE A AND LANE C.

The crosswalk records:

    A-S45  CARTEA_EXPONENTIAL_FORM_CONFIRMED, CARTEA_KAPPA_IS_NEARLY_FLAT_OVER_AN_HOUR
           kappa ~= 0.0097/bp, 293 attempts, 15 symbols, one day
    C-T14  FILL_CURVE_IS_A_POWER_LAW_ON_SMALL_TICK
           power law r2 0.998 vs exponential 0.906 on BTC/ETH

and marks it OPEN: "may be the same curve read at two scales, or a real disagreement about
functional form.  Neither lane saw the other."

This lane (C) wrote one of the two claims, so the review is not independent about C's half.
What IS independent, and is all that is attempted here, is a re-derivation of A-S45 from its
own PUBLISHED NUMBERS -- no re-run of A's driver, no new data, arithmetic only.  A-S45's
Madde 1 table gives the fill rates in full:

    depth bps   0.0    2.0    5.0   10.0   20.0
    fill        0.993  0.983  0.962  0.918  0.823      over 293 attempts total

THREE THINGS ARE CHECKED.

(1) ARE THE TWO CURVES THE SAME RANDOM VARIABLE?  A's fill is Sec 198 reachability over an
    HOUR: did any trade print at a price D bps from the mid.  That is the survival function
    of the HOURLY PRICE EXCURSION.  C's fill is P(x >= phi) per MARKET ORDER ARRIVAL: is
    this one order large enough to reach queue position phi at the touch.  That is the
    survival function of RELATIVE ORDER SIZE.  Different variables, and only A's is on
    Cartea Eq (8.1)'s axis, which is DEPTH.  C's own H-U7 docstring flagged the axis split
    in advance; the crosswalk did not carry it.

(2) DOES A'S RANGE DISCRIMINATE BETWEEN FORMS AT ALL?  kappa = 0.0097/bp over delta in
    [0, 20] gives kappa*delta in [0, 0.194]: the curve falls 17 percent end to end.  With
    293 attempts spread over 5 depths, each point carries a binomial standard error.  Both
    forms are fitted to A's five points and compared, and then a parametric discrimination
    test is run: generate data from each model at A's own n, refit both, and count how often
    the r2 comparison picks the generating model.  If that is near chance, the form was not
    established -- by A's numbers, not by opinion.

(3) FOR CONTRAST, over what range did C fit?  BTC 0.206 -> 0.095 across phi in [0.05, 1]:
    a 54 percent drop, and deliberately away from the origin where every form is flat.

Nothing here re-runs or re-judges A's DATA.  The fill rates are taken as published and
correct; only the inference from them to a functional form is re-derived.

ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.  Read-only; no DB, no repo state touched.

  python -m tools.ct016_close_the_fill_form_contradiction --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "reports/atlas"

# A-S45, SYSTEM_STATE section 473 (stable id A-S45), Madde 1 table, verbatim
A_DEPTH = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
A_FILL = np.array([0.993, 0.983, 0.962, 0.918, 0.823])
A_N_TOTAL = 293
A_KAPPA_PUBLISHED = 0.0097

# C-T14 / H-U7, BTCUSDT, the maker-relevant range
C_PHI = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 1.00])
C_FILL = np.array([0.2058, 0.1386, 0.1212, 0.1106, 0.1031, 0.0952])
C_R2_POWER, C_R2_EXP = 0.998, 0.906

N_SIM = 4000
RNG_SEED = 20260827


def fit_exp(d, p):
    """log p = -kappa d"""
    A = np.column_stack([np.ones(len(d)), d])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log(p))
    pred = np.exp(A @ c)
    ss = float(np.sum((p - p.mean()) ** 2))
    return {"kappa": float(-c[1]), "r2": float(1 - np.sum((p - pred) ** 2) / ss)}


def fit_power(d, p, eps=1.0):
    """log p = a - b log(eps + d).  eps keeps delta = 0 finite; declared, not tuned."""
    A = np.column_stack([np.ones(len(d)), np.log(eps + d)])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log(p))
    pred = np.exp(A @ c)
    ss = float(np.sum((p - p.mean()) ** 2))
    return {"exponent": float(-c[1]), "r2": float(1 - np.sum((p - pred) ** 2) / ss)}


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    n_per = A_N_TOTAL / len(A_DEPTH)
    se = np.sqrt(A_FILL * (1 - A_FILL) / n_per)

    fe = fit_exp(A_DEPTH, A_FILL)
    fp = fit_power(A_DEPTH, A_FILL)
    drop_A = float(A_FILL[0] - A_FILL[-1])
    drop_in_se = float(drop_A / np.sqrt(np.sum(se ** 2)))

    print("=== A-S45 re-derived from its published five points ===", flush=True)
    print("    depth      " + "".join("%8.1f" % d for d in A_DEPTH), flush=True)
    print("    fill       " + "".join("%8.3f" % f for f in A_FILL), flush=True)
    print("    binom SE   " + "".join("%8.3f" % s for s in se) +
          "     (n/point = %.0f)" % n_per, flush=True)
    print("    exponential fit  kappa %.5f/bp (published %.4f)   r2 %.4f"
          % (fe["kappa"], A_KAPPA_PUBLISHED, fe["r2"]), flush=True)
    print("    power-law fit    exponent %.4f                     r2 %.4f"
          % (fp["exponent"], fp["r2"]), flush=True)
    print("    end-to-end drop  %.3f  =  %.2f combined SE" % (drop_A, drop_in_se), flush=True)

    # discrimination test: can these five points, at this n, tell the forms apart?
    def gen(model):
        if model == "exp":
            p = np.exp(-fe["kappa"] * A_DEPTH)
        else:
            p = np.exp(np.log(A_FILL[0]) - fp["exponent"] * (np.log(1.0 + A_DEPTH)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        k = rng.binomial(int(round(n_per)), p)
        return np.clip(k / n_per, 1e-4, 1 - 1e-4)

    hits = {"exp": 0, "pow": 0}
    for model in ("exp", "pow"):
        for _ in range(N_SIM):
            s = gen(model)
            a, b = fit_exp(A_DEPTH, s), fit_power(A_DEPTH, s)
            pick = "exp" if a["r2"] >= b["r2"] else "pow"
            if pick == model:
                hits[model] += 1
    acc = {k: hits[k] / N_SIM for k in hits}
    overall = (hits["exp"] + hits["pow"]) / (2 * N_SIM)
    print("    DISCRIMINATION at A's own n:  generated exp -> picked exp %.3f   "
          "generated power -> picked power %.3f   overall %.3f  (chance 0.500)"
          % (acc["exp"], acc["pow"], overall), flush=True)

    ce = fit_exp(C_PHI, C_FILL)
    cp = fit_power(C_PHI, C_FILL, eps=0.0)
    drop_C = float(1.0 - C_FILL[-1] / C_FILL[0])
    print("=== C-T14 for contrast (BTCUSDT, published) ===", flush=True)
    print("    fitted range drop  %.3f of its starting value  (A: %.3f)"
          % (drop_C, 1.0 - A_FILL[-1] / A_FILL[0]), flush=True)
    print("    published r2  power %.3f  vs exponential %.3f" % (C_R2_POWER, C_R2_EXP),
          flush=True)
    print("    re-fit here   power %.3f  vs exponential %.3f" % (cp["r2"], ce["r2"]),
          flush=True)

    verdict = {
        "different_random_variables": {
            "A": "survival of the HOURLY PRICE EXCURSION (Sec 198 reachability over 1 h), "
                 "axis = DEPTH in bps -- this is Cartea Eq (8.1)'s axis",
            "C": "survival of RELATIVE ORDER SIZE per MARKET ORDER ARRIVAL, "
                 "axis = QUEUE POSITION at the touch -- not Cartea's axis",
            "conclusion": "the two tokens are not about the same quantity"},
        "A_discrimination_accuracy": overall,
        "A_drop_in_combined_SE": drop_in_se,
        "A_fits": {"exponential": fe, "power": fp},
        "C_fits_recomputed": {"exponential": ce, "power": cp},
        "C_range_drop_fraction": drop_C,
        "A_range_drop_fraction": float(1.0 - A_FILL[-1] / A_FILL[0]),
    }
    verdict["tokens"] = [
        "CT_016_DISSOLVED_DIFFERENT_RANDOM_VARIABLES",
        "ONLY_A_IS_ON_CARTEA_S_DEPTH_AXIS",
        ("A_S45_FORM_TEST_IS_UNDERPOWERED" if overall < 0.75
         else "A_S45_FORM_TEST_DISCRIMINATES"),
        "NEITHER_MEASUREMENT_IS_WITHDRAWN",
    ]
    print("=== VERDICT ===", flush=True)
    for t in verdict["tokens"]:
        print("    " + t, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT016_RESOLUTION_V1.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False, default=float)
    print("written %s/CT016_RESOLUTION_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
