# -*- coding: utf-8 -*-
"""D-E34 -- D-E33 found selection.  Is it selection on ACTIVITY?

D-E33 measured that spells still alive at u are interrupted at about a third of the rate the
population's own inter-episode gaps predict (observed 0.4956 against a calibrated null of 1.4799,
z -4.60).  Selection is present; the mechanism was left unidentified because the frailty FAMILY
is unavailable at K = 10 days.

This asks the one mechanism question that does NOT need a frailty family, because it uses a
MEASURED covariate instead of a latent one: do the spells that survive sit in QUIETER windows?

  If selection is on activity, then the mean prior activity of the still-alive set must FALL as u
  grows -- the busy ones are interrupted first and leave.

ABG's PRECONDITION, NAMED PER A-S81's RULE, AND IT CONSTRAINS THE DESIGN.  ABG on covariates:
*"Throughout the book we tacitly assume that all covariates are predictable ... the value at time
t of a time-dependent covariate should be known just before time t"*, and it directs the
external/internal distinction to Kalbfleisch and Prentice section 6.3.

  So the covariate is measured STRICTLY BEFORE t0 and never through it.  A symmetric or forward
  window would leak the very episodes that constitute the outcome -- the arrival of the next
  episode IS the INTERRUPTED event -- and would manufacture the result.

  Two PREDICTABLE covariates already exist inside the frozen construction and are used rather
  than built: `sigma_1s`, volatility on [t0-60m, t0), and `qv`, the episode's own notional.  Their
  provenance was checked IN THE CODE (`d_e11_p2_p3_v1` lines 9 and 67), not inferred from their
  names.  `sigma_1s_post` is post-anchor and is excluded by construction.

NULL.  The covariate is permuted across spells, which breaks any link to survival while preserving
both the covariate distribution and the survival pattern.  The statistic is the slope of mean
covariate against u among survivors.  Read against that null, not against zero.

SCOPE FENCE.  Outcome-blind: episode timestamps and the frozen cause labels only.  D-E8's frozen
estimand is untouched, mu_tau is not recomputed, and no threshold is selected.

Usage:  python tools/d_e34_selection_on_activity_v1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e11_p2_p3_v1 import collect                      # noqa: E402
from tools.d_e8_evaluator_v1 import (                          # noqa: E402
    FLOOR_PRIMARY, K_BPS, TAU_MIN, assert_spec_unchanged)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E34_SELECTION_ON_ACTIVITY_V1.json")
SEED = 20260827
NULL_SIMS = 2000

# THE COVARIATES ARE ALREADY IN THE FROZEN CONSTRUCTION, AND THEIR PROVENANCE WAS VERIFIED IN THE
# CODE RATHER THAN INFERRED FROM THEIR NAMES.  `d_e11_p2_p3_v1` line 9 and line 67: volatility is
# estimated on [t0 - 60m, t0), STRICTLY BEFORE the anchor.  That is exactly ABG's predictability
# requirement, so `sigma_1s` is usable as a selection covariate and needs no new construction.
#
# `sigma_1s_post` is measured AFTER the anchor and is therefore NOT predictable.  It is the harder
# null in P2 and it must never be used here; using it would read the outcome window.
#
# WHAT THIS DOES NOT ANSWER.  D-E33's `next:` said MARKET-WIDE activity.  The spell rows carry
# `stratum` at DAY resolution and no t0, so a cross-symbol look-back window cannot be built without
# `collect()` exposing t0.  That is a change to a load-bearing tool and it is named as the next
# step rather than hacked around here.  What follows is selection on the two PREDICTABLE covariates
# that already exist: pre-anchor volatility, and the episode's own size.
COVARIATES = (("sigma_1s", "pre-anchor volatility on [t0-60m, t0) -- predictable, verified in code"),
              ("qv", "the episode's own notional at t0 -- known at the anchor"))


def survivor_means(t, cov, grid):
    """Mean covariate among spells still alive at each grid point.  KEPT FOR THE RECORD ONLY.

    The first version of this study used the SLOPE of this quantity against u.  A known-positive
    ladder killed it: with survival made to depend on the covariate by construction, it fired at
    injected strengths 0.25 and 0.50 and MISSED 0.75 and 1.00.  A test that catches a weak effect
    and misses a strong one is BROKEN, not underpowered, and its null is unreadable.  The cause is
    that the statistic depends on the at-risk set size, which the injection also changes.
    """
    out = []
    for a in grid:
        m = t > a
        out.append(float(cov[m].mean()) if m.any() else np.nan)
    return np.array(out)


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def rho(a, b):
    """Spearman rank correlation -- GRID-FREE, so the at-risk set cannot deform it."""
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def ladder(t, cov, rng, reps=600):
    """The known-positive check the standing prompt requires before a zero may be read.

    Survival is made to depend on the covariate by construction at rising strength.  A usable
    statistic must move MONOTONICALLY.  Measured: rho runs 0.1532, 0.0926, 0.0195, -0.0925,
    -0.3011 across strengths 0 to 1, and the permutation p follows it down.  That is the licence
    to read the observed value; without it there is none.
    """
    order = rank(cov) / (len(cov) - 1)
    out = []
    for st in (0.0, 0.25, 0.5, 0.75, 1.0):
        tt = t * (1.0 - st * order)
        obs = rho(cov, tt)
        nulls = np.array([rho(rng.permutation(cov), tt) for _ in range(reps)])
        out.append({"strength": st, "rho": round(obs, 4),
                    "p_low": round(float((nulls <= obs).mean()), 4)})
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E34  is D-E33's selection ON ACTIVITY?   prereg sha256 %s VERIFIED" % h[:16])
    print("       covariate measured STRICTLY BEFORE t0 -- ABG requires predictable covariates,")
    print("       and a forward window would leak the episodes that ARE the outcome.\n")

    rows = collect(FLOOR_PRIMARY, K_BPS)
    t = np.array([r["t_ms"] for r in rows], float)

    grid = np.linspace(0.0, TAU_MIN * 60000.0 * 0.9, 10)
    gx = grid / 60000.0
    rng = np.random.default_rng(SEED)

    res = {"prereg_sha256": h, "n": len(rows),
           "answers": "D-E33 mechanism", "covariates": {}}
    print("  %-10s %12s %9s %9s %7s   %s"
          % ("covariate", "mean", "rho", "p_low", "z", "reading"))
    for name, kind in COVARIATES:
        cov = np.array([r[name] for r in rows], float)
        if not np.isfinite(cov).all():
            print("  %-10s  covariate unavailable" % name)
            continue
        obs = rho(cov, t)
        nulls = np.array([rho(rng.permutation(cov), t) for _ in range(NULL_SIMS)])
        z = (obs - nulls.mean()) / nulls.std(ddof=1)
        p_lo = float((nulls <= obs).mean())
        lad = ladder(t, cov, rng)
        monotone = all(lad[i]["rho"] >= lad[i + 1]["rho"] for i in range(len(lad) - 1))
        reading = ("INSTRUMENT_FAILED_ITS_KNOWN_POSITIVE_LADDER_RESULT_UNREADABLE" if not monotone
                   else ("SELECTION_SURVIVORS_ARE_QUIETER" if p_lo < 0.025 else
                         ("SELECTION_SURVIVORS_ARE_BUSIER" if p_lo > 0.975 else
                          "NOT_DISTINGUISHABLE_FROM_CHANCE")))
        res["covariates"][name] = {"kind": kind, "mean": round(float(cov.mean()), 3),
                                   "spearman_rho": round(obs, 5),
                                   "known_positive_ladder": lad, "ladder_monotone": monotone,
                                   "null_mean": round(float(nulls.mean()), 5),
                                   "null_sd": round(float(nulls.std(ddof=1)), 5),
                                   "z": round(float(z), 2), "one_sided_p_low": round(p_lo, 4),
                                   "reading": reading}
        print("  %-10s %12.3f %9.4f %9.4f %7.2f   %s"
              % (name, cov.mean(), obs, p_lo, z, reading))
        print("     ladder: %s   monotone=%s"
              % (" ".join("%.2f:%+.3f" % (x["strength"], x["rho"]) for x in lad), monotone))

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
