# -*- coding: utf-8 -*-
"""D-E13 -- ABG 9.4.1 local dependence, against a standing verdict that says the naive route fails.

THE CORPUS ASKS FOR DIRECTION.  ABG 9.4.1, Schweder (1970): *"If event A occurs first, then the
intensity of event B is changed, hence A influences B.  On the other hand, if event B occurs first,
then the intensity for A is unchanged; hence B does not influence A.  We say that B is locally
dependent on A, while A is locally independent on B."*  `D-E4` measured co-firing at 6.2x chance
within +/-1 minute, but coincidence counting is SYMMETRIC BY CONSTRUCTION, so it cannot answer this.

THE ESTATE HAS ALREADY CLOSED THE NAIVE ROUTE.  Section 430, Q10, verdict
`LEAD_LAG_IS_ACTIVITY_RATE_ARTIFACT`, recorded as STANDING and measured on exactly this data class
(liquidation timestamps + symbol):
    rank corr(mean onset lag, log activity)          -0.683
    residualise on activity:  rho 0.521 -> 0.025
    rate-matched null, 200 draws:  0.612 +/- 0.019, and 100% of draws exceeded the observed
    observed z against that null                     -4.87
  "a symbol that sees many liquidations has its 'first liquidation in the window' fall early
   ARITHMETICALLY."

So the question is NOT "which symbol leads".  It is:

    DOES A PAIR-COUNT ASYMMETRY INHERIT Q10's ARTEFACT, OR IS IT IMMUNE BY CONSTRUCTION?

The statistic here is not an onset-lag ordering.  For an ordered pair and a window w,
    A(w) = (N(+w) - N(-w)) / (N(+w) + N(-w))
where N(+w) counts pairs with 0 < t_Y - t_X <= w.  Under two INDEPENDENT homogeneous Poisson
processes, E[N(+w)] = E[N(-w)] = lam_X lam_Y w T for ANY rates, so the statistic is centred at zero
regardless of the rate difference -- which is precisely the leak Q10 found in the ordering
statistic.  That is an argument, not a measurement, so it is SIMULATED below before it is used.

Three nulls, because one is not enough here:
  N1  independent Poisson at the observed rates    -- does the artefact apply to me at all?
  N2  whole-day rotation                            -- preserves each symbol's clustering AND its
                                                       intraday seasonality (D-E4's null)
  N3  free rotation                                 -- destroys seasonality too; if a result
                                                       survives N2 but not N3, seasonality is it

SCOPE FENCE.  This characterises the DEPENDENCE STRUCTURE of the competing risk `INTERRUPTED`.
It is NOT a trading signal and no lead is proposed as one.  Any such reading is a different study
with its own multiplicity budget.  Outcome-blind: liquidation timestamps and notionals only.

Usage:  python tools/d_e13_local_dependence_v1.py
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "data", "microstructure_02.db")
CUTOFF_MS = 1787270400000
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MIN_GAP_MS = 900_000
FLOORS = (50_000.0, 0.0)
WINDOWS_S = (1, 5, 30, 60, 300)
DAY_MS = 86_400_000
REPS = 300
SEED = 20260827
OUT = os.path.join(ROOT, "reports", "atlas", "D_E13_LOCAL_DEPENDENCE_V1.json")


def episodes(floor):
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    out = {}
    for s in SYMBOLS:
        r = cn.execute("SELECT ts_ms,notional FROM liquidations WHERE symbol=? AND ts_ms<? "
                       "ORDER BY ts_ms", (s, CUTOFF_MS)).fetchall()
        ts = np.array([x[0] for x in r], np.int64)
        nt = np.array([(x[1] or 0.0) for x in r], float)
        brk = np.flatnonzero(np.diff(ts) > MIN_GAP_MS) + 1
        t0s = []
        for g in np.split(np.arange(len(ts)), brk):
            if len(g) and float(nt[g].sum()) >= floor:
                t0s.append(int(ts[g[0]]))
        out[s] = np.sort(np.array(t0s, np.int64))
    cn.close()
    return out


def asym(x, y, w_ms):
    """(N(+w) - N(-w)) / (N(+w) + N(-w)).  N(+w): y follows x within w."""
    i = np.searchsorted(y, x)
    j = np.searchsorted(y, x + w_ms, side="right")
    npos = int((j - i).sum())
    i2 = np.searchsorted(x, y)
    j2 = np.searchsorted(x, y + w_ms, side="right")
    nneg = int((j2 - i2).sum())
    tot = npos + nneg
    return (float((npos - nneg) / tot) if tot else float("nan")), npos, nneg


def rot_day(t, lo, span_days, k):
    return np.sort(lo + ((t - lo + k * DAY_MS) % (span_days * DAY_MS)))


def rot_free(t, lo, span_ms, off):
    return np.sort(lo + ((t - lo + off) % span_ms))


def run_floor(floor):
    eps = episodes(floor)
    lo = min(v.min() for v in eps.values())
    hi = max(v.max() for v in eps.values())
    span_ms = int(hi - lo) + 1
    span_days = int(span_ms // DAY_MS) + 1
    rng = np.random.default_rng(SEED)
    rates = {s: len(v) / (span_ms / 1000.0) for s, v in eps.items()}
    res = {"floor_usd": floor, "n": {s: int(len(v)) for s, v in eps.items()},
           "span_days": round(span_ms / DAY_MS, 2),
           "rate_per_hour": {s: round(r * 3600, 3) for s, r in rates.items()},
           "pairs": {}}
    for a, b in itertools.permutations(sorted(eps), 2):
        if a > b:
            continue                                   # A(w) for (b,a) is just -A(w)
        key = "%s->%s" % (a, b)
        row = {}
        for w_s in WINDOWS_S:
            w = w_s * 1000
            obs, npos, nneg = asym(eps[a], eps[b], w)
            # N1 independent Poisson at the observed rates -- does Q10's artefact touch me?
            n1 = []
            for _ in range(REPS):
                xa = np.sort(rng.random(len(eps[a])) * span_ms + lo).astype(np.int64)
                xb = np.sort(rng.random(len(eps[b])) * span_ms + lo).astype(np.int64)
                v, _, _ = asym(xa, xb, w)
                if np.isfinite(v):
                    n1.append(v)
            # N2 whole-day rotation of b
            n2 = [asym(eps[a], rot_day(eps[b], lo, span_days, k), w)[0]
                  for k in range(1, span_days)]
            n2 = [v for v in n2 if np.isfinite(v)]
            # N3 free rotation of b
            n3 = []
            for _ in range(REPS):
                off = int(rng.integers(1, span_ms))
                v, _, _ = asym(eps[a], rot_free(eps[b], lo, span_ms, off), w)
                if np.isfinite(v):
                    n3.append(v)

            def z(nl):
                nl = np.asarray(nl, float)
                sd = nl.std(ddof=1)
                return (round(float(nl.mean()), 4), round(float(sd), 4),
                        round(float((obs - nl.mean()) / sd), 2) if sd > 0 else None)

            m1, s1, z1 = z(n1)
            m2, s2, z2 = z(n2)
            m3, s3, z3 = z(n3)
            row["w_%ds" % w_s] = {
                "n_pos": npos, "n_neg": nneg, "asymmetry": round(obs, 4),
                "N1_poisson_mean": m1, "N1_sd": s1, "z_vs_N1": z1,
                "N2_dayrot_mean": m2, "N2_sd": s2, "z_vs_N2": z2,
                "N3_freerot_mean": m3, "N3_sd": s3, "z_vs_N3": z3}
        res["pairs"][key] = row
    return res


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    doc = {"study": "D-E13", "lane": "D",
           "class": "accounting_integrity_outcome_blind",
           "inherits": "section 430 Q10 LEAD_LAG_IS_ACTIVITY_RATE_ARTIFACT (STANDING)",
           "scope_fence": "characterises the dependence structure of the competing risk "
                          "INTERRUPTED; NOT a trading signal; no lead proposed as one",
           "populations": []}
    for f in FLOORS:
        r = run_floor(f)
        doc["populations"].append(r)
        print("\n===== floor $%d   n=%s   rate/h=%s" % (f, r["n"], r["rate_per_hour"]))
        for k, row in r["pairs"].items():
            print("  %s" % k)
            for w, v in row.items():
                print("    %-7s A=%+.4f  N1 %+.4f+/-%.4f z%-7s  N2 %+.4f z%-7s  N3 %+.4f z%s"
                      % (w, v["asymmetry"], v["N1_poisson_mean"], v["N1_sd"], v["z_vs_N1"],
                         v["N2_dayrot_mean"], v["z_vs_N2"], v["N3_freerot_mean"], v["z_vs_N3"]))
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
