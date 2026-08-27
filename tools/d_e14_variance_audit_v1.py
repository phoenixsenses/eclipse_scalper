# -*- coding: utf-8 -*-
"""D-E14 -- is D-E10's published confidence interval too narrow?

D-E10 published mu_tau = 18.10 min, 95% CI [16.68, 19.68], from a bootstrap clustered at
SYMBOL-DAY (72 strata).  Two things on the record say that unit may be too fine:

  D-E4      the three symbols CO-FIRE at 6.2x chance within +/-1 minute against a
            seasonality-preserving null.  if BTC/ETH/SOL episodes on the same DAY are dependent,
            then symbol-day strata are not independent and the interval is too narrow.
  S123/469  ABG 8.3: for a RATE model "the variance estimates from martingale theory ... will
            typically underestimate the true variance and have to be substituted with sandwich
            type estimators".  that branch measured its own symbol-clustered SE at 2.87x to 3.02x
            its martingale SE -- so coarsening the cluster mattered a lot there.

So: recompute the same estimand under three clustering units and let the ratio answer it.

  symbol-day   72 strata   what D-E10 published
  day          24 strata   pools the three symbols within a day -- if this is WIDER, the symbols
                           are dependent within the day and D-E10's interval was too narrow
  symbol        3 strata   reported and NOT interpreted: memory records that coarse clustering
                           COLLAPSES the SE at small G, and the tell is the SE FALLING.  a smaller
                           interval at G = 3 is that collapse, not evidence of independence.

A note on scope, so this is not overclaimed.  mu_tau here is a nonparametric Aalen-Johansen
functional resampled by cluster, so the bootstrap already IS a sandwich; ABG 8.3's specific warning
about martingale-theory variance does not bite this estimator.  What transfers is the CLUSTERING
UNIT question, and only that.

Read-only, outcome-blind in the sense D-E10 already opened: no new column is touched.

Usage:  python tools/d_e14_variance_audit_v1.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e11_p2_p3_v1 import collect  # noqa: E402
from tools.d_e8_evaluator_v1 import (  # noqa: E402
    FLOOR_PRIMARY, K_BPS, TAU_MIN, aalen_johansen, assert_spec_unchanged, mu_tau)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E14_VARIANCE_AUDIT_V1.json")
BOOT = 2000
SEED = 20260827


def boot_ci(t, c, keys, reps=BOOT, seed=SEED):
    tau_ms = TAU_MIN * 60000.0
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    idx_by = {u: np.flatnonzero(keys == u) for u in uniq}
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        g, p00, _, _ = aalen_johansen(t[idx], c[idx], tau_ms)
        out.append(mu_tau(g, p00))
    out = np.array(out)
    return {"n_clusters": int(len(uniq)),
            "se": round(float(out.std(ddof=1)), 4),
            "ci95": [round(float(np.percentile(out, 2.5)), 4),
                     round(float(np.percentile(out, 97.5)), 4)],
            "width": round(float(np.percentile(out, 97.5) - np.percentile(out, 2.5)), 4)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E14  variance audit   prereg sha256 %s VERIFIED" % h[:16])

    rows = collect(FLOOR_PRIMARY, K_BPS)
    t = np.array([r["t_ms"] for r in rows], float)
    c = np.array([r["cause"] for r in rows])
    sym = np.array([r["sym"] for r in rows])
    day = np.array([s.split("|")[1] for s in (r["stratum"] for r in rows)])
    symday = np.array([r["stratum"] for r in rows])

    g, p00, _, _ = aalen_johansen(t, c, TAU_MIN * 60000.0)
    point = mu_tau(g, p00)

    units = {"symbol_day_PUBLISHED": symday, "day": day, "symbol_UNINTERPRETED": sym}
    res = {"prereg_sha256": h, "n": len(rows), "point_mu_tau_min": round(point, 4),
           "clusterings": {}}
    for name, keys in units.items():
        r = boot_ci(t, c, keys)
        res["clusterings"][name] = r
        print("  %-24s G=%-4d SE %.4f  CI %s  width %.4f"
              % (name, r["n_clusters"], r["se"], r["ci95"], r["width"]))

    sd = res["clusterings"]["symbol_day_PUBLISHED"]
    dy = res["clusterings"]["day"]
    sy = res["clusterings"]["symbol_UNINTERPRETED"]
    ratio = dy["se"] / sd["se"]
    res["day_over_symbolday_se_ratio"] = round(ratio, 3)
    res["symbol_se_is_smaller_than_day"] = bool(sy["se"] < dy["se"])

    # descriptive: per (symbol, day) mean alive-time, correlated ACROSS symbols within day
    tab = collections.defaultdict(dict)
    for r in rows:
        s, d = r["sym"], r["stratum"].split("|")[1]
        tab[d].setdefault(s, []).append(r["t_ms"] / 60000.0)
    piv = {d: {s: float(np.mean(v)) for s, v in ss.items()} for d, ss in tab.items()}
    syms = sorted({s for ss in piv.values() for s in ss})
    corrs = {}
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            xs = [(v[a], v[b]) for v in piv.values() if a in v and b in v]
            if len(xs) >= 8:
                x = np.array([p[0] for p in xs]); y = np.array([p[1] for p in xs])
                corrs["%s|%s" % (a, b)] = {"n_days": len(xs),
                                           "corr": round(float(np.corrcoef(x, y)[0, 1]), 4)}
    res["within_day_cross_symbol_corr_of_mean_alive_time"] = corrs

    if ratio > 1.10:
        v = "PUBLISHED_INTERVAL_TOO_NARROW_SYMBOLS_ARE_DEPENDENT_WITHIN_DAY"
    elif ratio < 0.90:
        v = "DAY_CLUSTER_NARROWER_WHICH_IS_THE_SMALL_G_COLLAPSE_NOT_INDEPENDENCE"
    else:
        v = "PUBLISHED_INTERVAL_STANDS_SYMBOL_DAY_IS_AN_ADEQUATE_UNIT"
    res["verdict"] = v
    print("\n  day / symbol-day SE ratio = %.3f" % ratio)
    print("  within-day cross-symbol corr of mean alive time: %s"
          % {k: v2["corr"] for k, v2 in corrs.items()})
    print("  VERDICT: %s" % v)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
