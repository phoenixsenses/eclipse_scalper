# -*- coding: utf-8 -*-
"""D-E38 -- redo D-E37's EDGE_GONE claim as a CAUSE-SPECIFIC HAZARD, not a within-arm correlation.

D-E37 reported `within EDGE_GONE, n=493, rho -0.3358` and read it as "the selection runs through
EDGE_GONE".  C-T65 handed this lane the ABG sentence that makes the problem visible:

  *"In order to make statistical analyses on data, one must specify certain structured statistical
  models, and we here concentrate on Cox models and additive models."*  (verified on the shelf,
  1 hit, AALEN_BORGAN_GJESSING 8.6.1)

A rank correlation computed INSIDE the set of spells that eventually died of EDGE_GONE is not the
cause-specific hazard.  It conditions on the REALISED CAUSE, which is a post-baseline outcome, so
the subset is selected by the very thing being explained.  ABG's own object is different and it is
stated with its risk set:

  *"alpha_0h(t) dt is the probability that an individual will die of cause h in the small time
  interval [t, t+dt) given that the individual is still alive just prior to t"*, with intensity
  `alpha_0h(t) Y_0(t)` and `Y_0` the number STILL IN STATE 0.

THE ESTIMAND, FIXED BEFORE THE ESTIMATOR.  The cumulative cause-specific hazard of EDGE_GONE up to
tau, compared between HIGH and LOW pre-anchor volatility.  The unit is the spell-interval at risk,
NOT the spell that eventually failed from that cause.  Every spell enters the risk set at u = 0
and leaves it at its own transition, whatever the cause.

THE QUESTION, WRITTEN AS A QUESTION.  Does the EDGE_GONE cause-specific hazard differ by
pre-anchor volatility once the at-risk set is respected?  D-E37's number cannot answer it and this
one can be wrong in either direction.

A-S87's RULE, APPLIED.  Both strata are scored on ONE COMMON bin set, and the at-risk count of
each stratum is printed in every bin, so a difference cannot come from a bin only one side
supports.

NULL.  Sigma labels are permuted across spells; observed and null go through the SAME function.
2000 draws, so the control is an estimate rather than one draw.

SCOPE FENCE.  Outcome-blind inputs: the frozen cause labels, spell times, and pre-anchor sigma.
D-E8's estimand untouched, mu_tau not recomputed, no threshold selected.

Usage:  python tools/d_e38_cause_specific_by_sigma_v1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e11_p2_p3_v1 import collect                     # noqa: E402
from tools.d_e8_evaluator_v1 import (                         # noqa: E402
    FLOOR_PRIMARY, K_BPS, TAU_MIN, assert_spec_unchanged)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E38_CAUSE_SPECIFIC_BY_SIGMA_V1.json")
SEED = 20260828
NULL_SIMS = 2000
NBIN = 12


def cum_cause_hazard(t, cause, hi, grid, which="EDGE_GONE"):
    """Nelson-Aalen cumulative cause-specific hazard per stratum, on ONE COMMON grid.

    Returns (cumhaz_hi, cumhaz_lo, n_at_risk_hi, n_at_risk_lo) so support is visible per bin.
    """
    out = {}
    for name, m in (("hi", hi), ("lo", ~hi)):
        tt, cc = t[m], cause[m]
        ch, risk = 0.0, []
        cum = []
        for a, b in zip(grid[:-1], grid[1:]):
            y = float((tt > a).sum())                 # still in state 0 just prior to the bin
            d = float(((tt > a) & (tt <= b) & (cc == which)).sum())
            ch += (d / y) if y > 0 else 0.0
            cum.append(ch)
            risk.append(y)
        out[name] = (np.array(cum), np.array(risk))
    return out


def statistic(t, cause, hi, grid):
    """Difference in cumulative cause-specific hazard at tau.  ONE function, used for both the
    observation and every null draw -- A-S82's rule."""
    o = cum_cause_hazard(t, cause, hi, grid)
    return float(o["hi"][0][-1] - o["lo"][0][-1])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E38  EDGE_GONE cause-specific hazard by sigma   prereg %s VERIFIED" % h[:16])
    print("       D-E37 conditioned on the REALISED CAUSE; this respects the at-risk set.\n")

    rows = collect(FLOOR_PRIMARY, K_BPS)
    t = np.array([r["t_ms"] for r in rows], float)
    cause = np.array([r["cause"] for r in rows])
    sig = np.array([r["sigma_1s"] for r in rows], float)
    hi = sig > np.median(sig)
    grid = np.linspace(0.0, TAU_MIN * 60000.0, NBIN + 1)

    o = cum_cause_hazard(t, cause, hi, grid)
    print("  SUPPORT AND PATH -- A-S87: both strata on ONE bin set, at-risk printed per bin")
    print("  %8s %10s %10s %10s %10s" % ("bin(min)", "risk_hi", "risk_lo", "cum_hi", "cum_lo"))
    cells = []
    for i, (a, b) in enumerate(zip(grid[:-1], grid[1:])):
        mid = (a + b) / 2 / 60000.0
        rh, rl = o["hi"][1][i], o["lo"][1][i]
        cells.append({"mid_min": round(mid, 2), "risk_hi": int(rh), "risk_lo": int(rl),
                      "cum_hi": round(float(o["hi"][0][i]), 4),
                      "cum_lo": round(float(o["lo"][0][i]), 4)})
        print("  %8.1f %10d %10d %10.4f %10.4f" % (mid, rh, rl, o["hi"][0][i], o["lo"][0][i]))

    unsupported = [c for c in cells if c["risk_hi"] == 0 or c["risk_lo"] == 0]
    obs = statistic(t, cause, hi, grid)
    rng = np.random.default_rng(SEED)
    null = np.array([statistic(t, cause, rng.permutation(hi), grid) for _ in range(NULL_SIMS)])
    z = (obs - null.mean()) / null.std(ddof=1)
    p2 = 2 * min(float((null <= obs).mean()), float((null >= obs).mean()))

    print("\n  n hi=%d  lo=%d   EDGE_GONE events hi=%d lo=%d"
          % (hi.sum(), (~hi).sum(),
             int(((cause == "EDGE_GONE") & hi).sum()), int(((cause == "EDGE_GONE") & ~hi).sum())))
    print("  cum cause-specific hazard at tau   hi %.4f   lo %.4f   diff %+.4f"
          % (o["hi"][0][-1], o["lo"][0][-1], obs))
    print("  permutation null  %+.4f +/- %.4f   z %+.2f   two-sided p %.4f  (alpha of this"
          " threshold is the p itself)" % (null.mean(), null.std(ddof=1), z, p2))
    print("  bins with a zero-risk stratum: %d" % len(unsupported))

    # POWER LADDER, CORRECTED D-E39.  The first version started each draw from the OBSERVED
    # label vector and injected on top of it, so its `s = 0` row reproduced the observed result
    # rather than the false-positive rate.  A ladder whose zero row is not near alpha is not a
    # power curve at all.  Each draw now starts from a PERMUTED (null) vector and injects from
    # there, and the s = 0 column is printed as what it is: the false-positive rate.
    print("")
    print("  POWER LADDER -- each draw starts from a PERMUTED null, then injects")
    lad = []
    eg = np.flatnonzero(cause == "EDGE_GONE")
    for s in (0.0, 0.2, 0.4):
        hits = 0
        for _ in range(200):
            hh = rng.permutation(hi)
            hh[eg[rng.random(len(eg)) < s]] = True
            v = statistic(t, cause, hh, grid)
            hits += abs((v - null.mean()) / null.std(ddof=1)) > 1.96
        lad.append({"strength": s, "detect_rate": round(hits / 200.0, 3)})
        tag = "   <- this IS the false-positive rate" if s == 0 else ""
        print("     s=%.1f   detected %.0f%%%s" % (s, 100.0 * hits / 200, tag))

    res = {"prereg_sha256": h, "n": len(rows), "n_hi": int(hi.sum()), "n_lo": int((~hi).sum()),
           "estimand": "cumulative cause-specific hazard of EDGE_GONE to tau, hi-sigma minus lo",
           "supersedes_reading_of": "D-E37 within-arm rho -0.3358 (conditioned on realised cause)",
           "cells": cells, "unsupported_bins": len(unsupported),
           "observed_diff": round(obs, 5),
           "null_mean": round(float(null.mean()), 5), "null_sd": round(float(null.std(ddof=1)), 5),
           "z": round(float(z), 2), "two_sided_p": round(p2, 5),
           "known_positive_ladder": lad,
           "corpus": {"source": "AALEN_BORGAN_GJESSING", "locator": "8.6.1",
                      "quote_verified": True,
                      "says": "one must specify certain structured statistical models",
                      "handed_over_by": "C-T65"}}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
