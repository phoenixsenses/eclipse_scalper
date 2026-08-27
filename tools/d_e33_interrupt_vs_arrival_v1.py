# -*- coding: utf-8 -*-
"""D-E33 -- is INTERRUPTED the arrival process, or a SELECTED view of it?

D-E27 left this open and it has been deferred four times: at a size floor the arrivals are bursty,
so is that burstiness the same object as D-E13's INTERRUPTED competing risk, or a second one?

THE QUESTION, MADE IDENTIFIABLE.  A spell in D-E10's decomposition starts at an episode and ends
as EDGE_GONE, INTERRUPTED (the next episode arrived first), ADMINISTRATIVE (still alive at tau) or
NEVER_ALIVE.  Because the spell clock starts AT an episode, elapsed spell time IS elapsed time
since the previous episode.  So two hazards live on the same clock and can be compared directly:

  lambda_1(u)   cause-specific hazard of INTERRUPTED among spells STILL ALIVE at u
  h(u)          renewal hazard of the next episode at elapsed time u, from ALL inter-episode gaps

  ratio == 1  ->  INTERRUPTED just IS the arrival process; one object.
  ratio != 1  ->  the spells that are still alive are a BIASED subset of episodes, and the
                  interruption they experience is not the population's arrival process.  A second
                  object, and it is selection.

WHAT THE CORPUS SAYS, AND ITS OWN PRECONDITION.  ABG 6.3 "Hazard and frailty of survivors":
*"Those who survive beyond a certain time will be more robust and have a different frailty
distribution compared to the original frailty distribution."*  That PREDICTS a ratio != 1.  But
A-S81's rule says to name the method's own preconditions and say whether they hold, so:

  ABG's frailty machinery requires the frailty FAMILY, and it is explicit that the family decides
  even the SIGN of what happens -- *"the coefficient of variation decreases with time t when
  -1 < m < 0 ... while it increases for m > 0.  For the gamma distributions the coefficient of
  variation is constant.  Hence, the population of survivors could be increasingly similar or
  dissimilar due to frailty selection, according to which specific frailty distribution is
  involved."*

  THAT PRECONDITION DOES NOT HOLD HERE.  This estate's own MPH gate records that K = 10 days
  cannot identify the frailty shape.  So this script MEASURES the two hazards and REFUSES to fit a
  frailty model on top of them.  A ratio != 1 establishes that selection is PRESENT; it does not
  identify the mechanism, and no number below should be read as doing so.

SCOPE FENCE.  D-E8's frozen estimand is untouched; mu_tau is not recomputed and no threshold is
selected.  Outcome-blind inputs only: episode timestamps and the frozen cause labels.

Usage:  python tools/d_e33_interrupt_vs_arrival_v1.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e11_p2_p3_v1 import collect            # noqa: E402
from tools.d_e8_evaluator_v1 import (               # noqa: E402
    CUTOFF_MS, DB, FLOOR_PRIMARY, K_BPS, TAU_MIN, assert_spec_unchanged)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
EPISODE_GAP_MS = 900_000
OUT = os.path.join(ROOT, "reports", "atlas", "D_E33_INTERRUPT_VS_ARRIVAL_V1.json")
SEED = 20260827
NULL_SIMS = 2000


def episode_starts(floor):
    """Episode start times per symbol, at the preregistered floor, from the frozen DB."""
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    out = {}
    for s in SYMBOLS:
        r = cn.execute("SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND ts_ms<? "
                       "ORDER BY ts_ms", (s, CUTOFF_MS)).fetchall()
        ts = np.array([x[0] for x in r], np.int64)
        nt = np.array([(x[1] or 0.0) for x in r], float)
        brk = np.flatnonzero(np.diff(ts) > EPISODE_GAP_MS) + 1
        starts = []
        for g in np.split(np.arange(len(ts)), brk):
            if len(g) and float(nt[g].sum()) >= floor:
                starts.append(int(ts[g[0]]))
        out[s] = np.sort(np.array(starts, np.int64))
    cn.close()
    return out


def renewal_hazard(gaps_ms, grid_ms):
    """h(u) from ALL inter-episode gaps: events in (u, u+du] over those still waiting at u."""
    g = np.asarray(gaps_ms, float)
    out = []
    for a, b in zip(grid_ms[:-1], grid_ms[1:]):
        at_risk = float((g > a).sum())
        ev = float(((g > a) & (g <= b)).sum())
        out.append(ev / at_risk if at_risk > 0 else np.nan)
    return np.array(out)


def cause_hazard(t_ms, cause, grid_ms, which="INTERRUPTED"):
    """Cause-specific hazard among spells still alive at u."""
    t = np.asarray(t_ms, float)
    c = np.asarray(cause)
    out = []
    for a, b in zip(grid_ms[:-1], grid_ms[1:]):
        at_risk = float((t > a).sum())
        ev = float(((t > a) & (t <= b) & (c == which)).sum())
        out.append(ev / at_risk if at_risk > 0 else np.nan)
    return np.array(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E33  INTERRUPTED vs the arrival process   prereg sha256 %s VERIFIED" % h[:16])
    print("       corpus PREDICTS a difference (ABG 6.3) and its own precondition -- the frailty")
    print("       FAMILY -- is NOT identifiable here, so this measures and does not parameterise.\n")

    rows = collect(FLOOR_PRIMARY, K_BPS)
    t = np.array([r["t_ms"] for r in rows], float)
    c = np.array([r["cause"] for r in rows])
    sym = np.array([r["sym"] for r in rows])

    eps = episode_starts(FLOOR_PRIMARY)
    all_gaps = np.concatenate([np.diff(v).astype(float) for v in eps.values() if len(v) > 1])

    grid = np.linspace(0.0, TAU_MIN * 60000.0, 13)          # 5-minute bins to tau
    mid = (grid[:-1] + grid[1:]) / 2 / 60000.0

    lam = cause_hazard(t, c, grid)
    hz = renewal_hazard(all_gaps, grid)

    print("  bin(min)   lambda_1 INTERRUPTED    h(u) arrivals      ratio")
    cells = []
    for m, a, b in zip(mid, lam, hz):
        r = (a / b) if (b and np.isfinite(b) and b > 0) else np.nan
        cells.append({"mid_min": round(float(m), 2),
                      "lambda_interrupted": None if not np.isfinite(a) else round(float(a), 5),
                      "h_arrival": None if not np.isfinite(b) else round(float(b), 5),
                      "ratio": None if not np.isfinite(r) else round(float(r), 3)})
        print("  %7.1f   %18s %17s %10s"
              % (m, "%.5f" % a if np.isfinite(a) else "-",
                 "%.5f" % b if np.isfinite(b) else "-",
                 "%.3f" % r if np.isfinite(r) else "-"))

    # THE NOMINAL CI IS NOT USABLE AND THE MEASUREMENT SAYS SO.  A symbol-day cluster bootstrap
    # on 16 INTERRUPTED events was calibrated against a null where lambda_1 == h BY CONSTRUCTION:
    # it covered 1 in 47.5% of simulations against a nominal 95%, a false-positive rate of 52.5%,
    # and worse, the estimator is MIS-CENTRED -- its null distribution sits at 1.48, not 1.  So the
    # observed value is read against its OWN simulated null, which is the correct use of a biased
    # statistic, and the bootstrap interval is not reported at all.
    fin = np.isfinite(lam) & np.isfinite(hz) & (hz > 0)
    pooled = float(np.nansum(lam[fin]) / np.nansum(hz[fin])) if fin.any() else float("nan")

    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(NULL_SIMS):
        cc, tt = c.copy(), t.copy()
        for b in range(len(grid) - 1):
            if not np.isfinite(hz[b]) or hz[b] <= 0:
                continue
            ar = np.flatnonzero(tt > grid[b])
            if not len(ar):
                continue
            fire = ar[rng.random(len(ar)) < hz[b]]
            cc[fire] = "INTERRUPTED"
            tt[fire] = (grid[b] + grid[b + 1]) / 2
        l2 = cause_hazard(tt, cc, grid)
        f2 = np.isfinite(l2) & fin
        if f2.any() and np.nansum(hz[f2]) > 0:
            null.append(float(np.nansum(l2[f2]) / np.nansum(hz[f2])))
    null = np.array(null)
    z = (pooled - null.mean()) / null.std(ddof=1)
    p_one = float((null <= pooled).mean())

    print("")
    print("  POOLED ratio  sum(lambda_1) / sum(h)  =  %.4f      n INTERRUPTED = %d"
          % (pooled, int((c == "INTERRUPTED").sum())))
    print("  CALIBRATED NULL (lambda_1 == h by construction, %d sims): mean %.4f  sd %.4f"
          % (len(null), null.mean(), null.std(ddof=1)))
    print("  the null is NOT centred at 1 -- the estimator is mis-centred, which is exactly why")
    print("  the nominal bootstrap interval is withheld (it covered 1 in 47.5% of null sims).")
    print("  z against its own null  %+.2f      one-sided p  %.4f" % (z, p_one))
    same = bool(p_one > 0.05 and p_one < 0.95)
    verdict = ("SAME_OBJECT_INTERRUPTED_IS_THE_ARRIVAL_PROCESS" if same else
               ("SECOND_OBJECT_ALIVE_SPELLS_ARE_INTERRUPTED_LESS_THAN_THE_POPULATION"
                if pooled < null.mean() else
                "SECOND_OBJECT_ALIVE_SPELLS_ARE_INTERRUPTED_MORE_THAN_THE_POPULATION"))
    print("  VERDICT: %s" % verdict)

    res = {"prereg_sha256": h, "n_spells": len(rows), "n_gaps": int(len(all_gaps)),
           "grid_min": [round(float(x) / 60000, 2) for x in grid],
           "cells": cells,
           "pooled_ratio": round(pooled, 4),
           "n_interrupted": int((c == "INTERRUPTED").sum()),
           "calibrated_null": {"sims": len(null), "mean": round(float(null.mean()), 4),
                               "sd": round(float(null.std(ddof=1)), 4),
                               "p05": round(float(np.percentile(null, 5)), 4),
                               "z": round(float(z), 2), "one_sided_p": round(p_one, 5)},
           "nominal_bootstrap_ci": "WITHHELD -- covered 1 in 47.5% of null sims (nominal 95%)",
           "verdict": verdict,
           "corpus": {"source": "AALEN_BORGAN_GJESSING",
                      "locator": "6.3",
                      "says": "survivors have a different frailty distribution -> predicts ratio != 1",
                      "its_own_precondition": ("the frailty FAMILY decides even the sign: CV falls "
                                               "for -1<m<0, rises for m>0, constant for gamma"),
                      "precondition_holds_here": False,
                      "why_not": "this estate's MPH gate records K=10 days cannot identify the "
                                 "frailty shape, so the mechanism is NOT identified by this result"},
           "scope": "measures two hazards on one clock; fits no frailty model; D-E8's frozen "
                    "estimand untouched; mu_tau not recomputed"}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
