# -*- coding: utf-8 -*-
"""LANE D / D-E4 -- THE FORCED-FLOW ARRIVAL PROCESS.

Read-only, OUTCOME-BLIND.  Reads `sym`, `t0`, `q` only.  The `imp_*` and
`pre_bps` columns sit in the same pickle and are never touched.

WHY THIS, WHY NOW
-----------------
`D-E3` extracted the corpus's own questions.  Two of them are aimed straight at
this lane and have never been answered here:

  ABG 1.5.4     "Independent or dependent data?"
  STK4080 Sl.1  "Can valve life in these systems be modeled as a renewal process?"

Both are about the EPISODE ARRIVAL PROCESS -- which is also the competing risk
`D-E1` named and `D-E2` measured ("the next episode arrives").  Neither needs a
single outcome value.

WHAT IS MEASURED (declared family, Holm-corrected over the 6 primary tests)
--------------------------------------------------------------------------
  T1  DEAD-TIME-CORRECTED EXPONENTIALITY.  Episodes are separated by > 900 s BY
      CONSTRUCTION, so the process is at best a Poisson process observed through
      a 900 s dead time.  The testable statement is that (gap - 900 s) is
      exponential.  Statistic: CV of the corrected gaps (Poisson => 1).
  T2  INDEX OF DISPERSION of counts in fixed windows (Poisson => 1), with a
      day-blocked bootstrap SE because the day is this estate's unit of
      independence.
  T3  LAPLACE TREND TEST over the 24-day span (constant rate => U ~ N(0,1)).
  T4  INTRADAY SEASONALITY, counts by UTC hour vs uniform (chi-square).  This
      runs BEFORE T1/T2 are interpreted: seasonality alone manufactures
      overdispersion, and CLAUDE.md already records intraday seasonality as a
      real risk-state feature here.
  T5  LAGGED DURATION DEPENDENCE -- carried over from `D-E2` unchanged, so the
      renewal question and the Honore branch are answered on one page.
  T6  CROSS-SYMBOL DEPENDENCE.  Null = WHOLE-DAY circular rotations of the other
      symbol's timestamps, which preserve that symbol's own clustering AND its
      intraday seasonality while destroying cross-symbol alignment.  This is the
      only null that separates "the symbols co-fire" from "the symbols share a
      clock".

Every statistic is reported at the declared floor AND at `D-E2`'s $50k floor,
because `D-E2` established that a duration statement without its floor is not
interpretable.  NO THRESHOLD IS SELECTED.

Usage:  python tools/d_e4_arrival_process_v1.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = "data/pve_01_v1/_s97_extended.pkl"
OUT = "reports/atlas/D_E4_ARRIVAL_PROCESS_V1.json"

DEAD_TIME_MS = 900_000          # the episode definition's own gap rule
FLOORS = (0.0, 50_000.0)        # declared, not selected: D-E2's two populations
DAY_MS = 86_400_000
BOOT = 2000
N_PRIMARY = 6                   # T1..T6, Holm


def rows(path=SAMPLE):
    d = pickle.loads(open(path, "rb").read())
    r = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return [{"sym": x["sym"], "t0": int(x["t0"]), "q": float(x["q"])} for x in r]


def by_symbol(rs):
    out = collections.defaultdict(list)
    for r in rs:
        out[r["sym"]].append(r["t0"])
    return {s: np.sort(np.array(v, np.int64)) for s, v in sorted(out.items())}


def _norm_sf(z):
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0)) * 2.0


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (len(pvals) - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


# ---------------------------------------------------------------- T1
def t1_exponentiality(ts_by_sym):
    """CV of dead-time-corrected gaps.  Exponential => CV = 1, SE ~ 1/sqrt(n)."""
    per, allg = {}, []
    for s, ts in ts_by_sym.items():
        g = np.diff(ts).astype(float) - DEAD_TIME_MS
        g = g[g > 0]
        if len(g) < 30:
            continue
        cv = float(g.std(ddof=1) / g.mean())
        per[s] = {"n_gaps": int(len(g)), "cv": round(cv, 4),
                  "mean_min": round(float(g.mean()) / 60000.0, 2),
                  "z_vs_1": round((cv - 1.0) / (1.0 / math.sqrt(len(g))), 2)}
        allg.append(g)
    raw = np.concatenate(allg)
    cv_raw = float(raw.std(ddof=1) / raw.mean())
    # POOLING TRAP: the symbols have different mean gaps, so a raw pool is a scale
    # MIXTURE and its CV exceeds every individual symbol's.  Standardise each
    # symbol by its own mean before pooling.
    std = np.concatenate([g / g.mean() for g in allg])
    cv = float(std.std(ddof=1) / std.mean())
    z = (cv - 1.0) / (1.0 / math.sqrt(len(std)))
    return {"test": "T1_dead_time_corrected_exponentiality",
            "null": "gap - 900s is exponential => CV = 1",
            "pooled_within_symbol": {"n_gaps": int(len(std)), "cv": round(cv, 4),
                                     "z": round(z, 2)},
            "pooled_raw_MIXTURE_DO_NOT_READ": {"cv": round(cv_raw, 4),
                                               "mean_min": round(float(raw.mean()) / 60000.0, 2),
                                               "note": "scale mixture across symbols; "
                                                       "reported only to show the trap"},
            "per_symbol": per, "p": _norm_sf(z)}


# ---------------------------------------------------------------- T2
def t2_dispersion(ts_by_sym, window_min=60, null=None):
    """var/mean of counts in fixed windows; day-blocked bootstrap SE."""
    w = window_min * 60_000
    per = {}
    zs = []
    for s, ts in ts_by_sym.items():
        lo, hi = ts.min(), ts.max()
        edges = np.arange(lo, hi + w, w)
        cnt = np.histogram(ts, bins=edges)[0].astype(float)
        day = ((edges[:-1] - lo) // DAY_MS).astype(int)
        d0 = float(cnt.var(ddof=1) / cnt.mean()) if cnt.mean() > 0 else float("nan")
        days = np.unique(day)
        rng = np.random.default_rng(20260827)
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(days, size=len(days), replace=True)
            c = np.concatenate([cnt[day == d] for d in pick])
            if c.mean() > 0:
                bs.append(c.var(ddof=1) / c.mean())
        se = float(np.std(bs, ddof=1))
        z_poisson = (d0 - 1.0) / se if se > 0 else float("nan")
        row = {"n_windows": int(len(cnt)), "mean_count": round(float(cnt.mean()), 3),
               "index_of_dispersion": round(d0, 4), "boot_se_day_blocked": round(se, 4),
               "z_vs_POISSON_1_UNCALIBRATED": round(z_poisson, 2), "n_days": int(len(days))}
        if null and s in null:
            nm, nsd = null[s]["null_index_of_dispersion_mean"], null[s]["null_sd"]
            zc = (d0 - nm) / nsd
            row["null_dead_time_mean"] = nm
            row["null_dead_time_sd"] = nsd
            row["z_vs_CALIBRATED_dead_time_null"] = round(zc, 2)
            zs.append(zc)
        else:
            zs.append(z_poisson)
        per[s] = row
    zbar = float(np.mean(zs))
    return {"test": "T2_index_of_dispersion", "window_min": window_min,
            "null": "scored against the CALIBRATED dead-time null (T2b), not against "
                    "Poisson = 1; the Poisson z is kept only to show how far an "
                    "uncalibrated null was out",
            "per_symbol": per, "mean_z": round(zbar, 2), "p": _norm_sf(zbar)}


def t2b_dead_time_null(ts_by_sym, window_min=60, reps=400, seed=20260827):
    """CALIBRATE T2's NULL BEFORE READING IT.

    A 900 s dead time is built into the episode definition, and a dead time makes
    counts MORE regular than Poisson.  So `index of dispersion < 1` may be the
    detector, not the market.  This simulates the fitted dead-time model --
    gap = 900 s + Exp(mean - 900 s), i.e. exactly a Poisson process seen through
    the dead time -- at each symbol's own rate and span, and reports what index of
    dispersion that world produces.  Nothing is concluded from T2 until this is on
    the page (CLAUDE.md 380-C: never freeze a gate without its null value).
    """
    w = window_min * 60_000
    rng = np.random.default_rng(seed)
    out = {}
    for s, ts in ts_by_sym.items():
        span = float(ts.max() - ts.min())
        g = np.diff(ts).astype(float) - DEAD_TIME_MS
        g = g[g > 0]
        scale = float(g.mean())
        sims = []
        for _ in range(reps):
            t, arr = 0.0, []
            while t < span:
                t += DEAD_TIME_MS + rng.exponential(scale)
                if t < span:
                    arr.append(t)
            a = np.asarray(arr)
            if len(a) < 20:
                continue
            edges = np.arange(0.0, span + w, w)
            c = np.histogram(a, bins=edges)[0].astype(float)
            if c.mean() > 0:
                sims.append(c.var(ddof=1) / c.mean())
        out[s] = {"n_sims": len(sims),
                  "null_index_of_dispersion_mean": round(float(np.mean(sims)), 4),
                  "null_sd": round(float(np.std(sims, ddof=1)), 4),
                  "null_p05_p95": [round(float(np.percentile(sims, 5)), 4),
                                   round(float(np.percentile(sims, 95)), 4)],
                  "sim_mean_count_per_window": round(float(span / w and
                                                          (span / (DEAD_TIME_MS + scale)) / (span / w)), 3)}
    return {"test": "T2b_dead_time_null_calibration",
            "model": "gap = 900s + Exp(mean-900s) at each symbol's own rate and span",
            "why": "a dead time regularises counts; index of dispersion < 1 may be the "
                   "detector rather than the market, and T2's null value was never computed",
            "per_symbol": out}


def t4_power(ts_by_sym, reps=400, seed=20260828):
    """What peak/trough does PURE POISSON produce at this N?  T4's own null value."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, ts in ts_by_sym.items():
        n = len(ts)
        rat = []
        for _ in range(reps):
            c = rng.multinomial(n, [1 / 24.0] * 24).astype(float)
            rat.append(c.max() / max(1.0, c.min()))
        out[s] = {"n": int(n),
                  "null_peak_over_trough_p50": round(float(np.percentile(rat, 50)), 2),
                  "null_peak_over_trough_p95": round(float(np.percentile(rat, 95)), 2)}
    return {"test": "T4b_seasonality_null_value",
            "why": "an eyeballed 3.4x peak/trough is not evidence unless the uniform "
                   "null produces less than that at this N",
            "per_symbol": out}


# ---------------------------------------------------------------- T3
def t3_laplace(ts_by_sym):
    """Laplace centroid test for a monotone trend in rate.  Constant => N(0,1)."""
    per, zs = {}, []
    for s, ts in ts_by_sym.items():
        T0, T1 = float(ts.min()), float(ts.max())
        x = (ts.astype(float) - T0)
        n = len(x)
        span = T1 - T0
        if n < 10 or span <= 0:
            continue
        u = (x.sum() / n - span / 2.0) / (span * math.sqrt(1.0 / (12.0 * n)))
        per[s] = {"n": int(n), "laplace_U": round(float(u), 3)}
        zs.append(u)
    zbar = float(np.mean(zs))
    return {"test": "T3_laplace_trend", "null": "constant rate => U ~ N(0,1)",
            "per_symbol": per, "mean_U": round(zbar, 3), "p": _norm_sf(zbar)}


# ---------------------------------------------------------------- T4
def t4_seasonality(ts_by_sym):
    per, chis, dof = {}, 0.0, 0
    for s, ts in ts_by_sym.items():
        h = ((ts // 3_600_000) % 24).astype(int)
        cnt = np.bincount(h, minlength=24).astype(float)
        e = cnt.sum() / 24.0
        chi = float(((cnt - e) ** 2 / e).sum())
        per[s] = {"n": int(cnt.sum()), "chi2_23df": round(chi, 1),
                  "max_hour": int(cnt.argmax()), "min_hour": int(cnt.argmin()),
                  "peak_over_trough": round(float(cnt.max() / max(1.0, cnt.min())), 2)}
        chis += chi
        dof += 23
    # Wilson-Hilferty normal approximation for a chi-square with dof df
    z = ((chis / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return {"test": "T4_intraday_seasonality", "null": "uniform over 24 UTC hours",
            "per_symbol": per, "chi2_total": round(chis, 1), "df": dof,
            "z_wilson_hilferty": round(z, 2), "p": _norm_sf(z)}


# ---------------------------------------------------------------- T5
def t5_lagged_duration(ts_by_sym):
    xs, ys = [], []
    for s, ts in ts_by_sym.items():
        g = np.diff(ts).astype(float)
        g = g[g > 0]
        if len(g) < 5:
            continue
        lg = np.log(g)
        a, b = lg[:-1], lg[1:]
        xs.append(a - a.mean())
        ys.append(b - b.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    sxx = (x * x).sum()
    beta = float((x * y).sum() / sxx)
    res = y - beta * x
    se = float(math.sqrt((res ** 2).sum() / (len(x) - 2) / sxx))
    z = beta / se
    return {"test": "T5_lagged_duration_dependence",
            "null": "renewal / Honore Thm 1 premise => beta = 0",
            "beta": round(beta, 4), "se": round(se, 4), "z": round(z, 2),
            "n_pairs": int(len(x)), "mde_80pct": round(2.80 * se, 4),
            "p": _norm_sf(z)}


# ---------------------------------------------------------------- T6
def t6_cross_symbol(ts_by_sym, tol_min=5):
    """Co-firing vs a WHOLE-DAY-ROTATION null (preserves seasonality + clustering)."""
    tol = tol_min * 60_000
    syms = list(ts_by_sym)
    lo = min(t.min() for t in ts_by_sym.values())
    hi = max(t.max() for t in ts_by_sym.values())
    span_days = int((hi - lo) // DAY_MS) + 1

    def coincid(a, b):
        j = np.searchsorted(b, a)
        best = np.full(len(a), np.inf)
        for k in (-1, 0):
            jj = np.clip(j + k, 0, len(b) - 1)
            best = np.minimum(best, np.abs(b[jj] - a).astype(float))
        return int((best <= tol).sum())

    pairs = {}
    zs = []
    for i in range(len(syms)):
        for k in range(i + 1, len(syms)):
            a, b = ts_by_sym[syms[i]], ts_by_sym[syms[k]]
            obs = coincid(a, b)
            null = []
            for sh in range(1, span_days):
                bb = np.sort(lo + ((b - lo + sh * DAY_MS) % (span_days * DAY_MS)))
                null.append(coincid(a, bb))
            m, sd = float(np.mean(null)), float(np.std(null, ddof=1))
            z = (obs - m) / sd if sd > 0 else float("nan")
            pairs["%s|%s" % (syms[i], syms[k])] = {
                "n_a": int(len(a)), "observed_coincidences": obs,
                "share_of_a": round(obs / len(a), 4),
                "null_mean": round(m, 2), "null_sd": round(sd, 2),
                "n_rotations": len(null), "z": round(z, 2),
                "excess_ratio": round(obs / m, 2) if m > 0 else None}
            zs.append(z)
    zbar = float(np.mean(zs))
    return {"test": "T6_cross_symbol_dependence", "tolerance_min": tol_min,
            "null": "whole-day rotation: preserves each symbol's own clustering and "
                    "its intraday seasonality, destroys cross-symbol alignment",
            "pairs": pairs, "mean_z": round(zbar, 2), "p": _norm_sf(zbar)}


def analyse(rs, floor):
    sub = [r for r in rs if r["q"] >= floor]
    ts = by_symbol(sub)
    cal_t2 = t2b_dead_time_null(ts)
    calibrations = [cal_t2, t4_power(ts)]
    tests = [t1_exponentiality(ts), t2_dispersion(ts, null=cal_t2["per_symbol"]),
             t3_laplace(ts), t4_seasonality(ts), t5_lagged_duration(ts),
             t6_cross_symbol(ts)]
    # tolerance is a choice, so report the family (D-E2's rule) -- NOT in the
    # Holm family, these are the same test at other tolerances
    tol_family = [t6_cross_symbol(ts, tol_min=m) for m in (1, 15, 30, 60)]
    ps = [t["p"] for t in tests]
    for t, ph in zip(tests, holm(ps)):
        t["p"] = float("%.3g" % t["p"])
        t["p_holm6"] = float("%.3g" % ph)
        t["reject_at_0_05"] = bool(ph < 0.05)
    lo = min(t.min() for t in ts.values())
    hi = max(t.max() for t in ts.values())
    return {"floor_usd": floor, "n_episodes": len(sub),
            "per_symbol_n": {s: int(len(v)) for s, v in ts.items()},
            "span_days": round((hi - lo) / DAY_MS, 2),
            "family": "T1..T6, Holm over %d" % N_PRIMARY,
            "tests": tests, "null_calibrations": calibrations,
            "T6_tolerance_family_not_in_holm": [
                {"tolerance_min": t["tolerance_min"], "mean_z": t["mean_z"],
                 "pairs": {k: {"share_of_a": v["share_of_a"],
                               "excess_ratio": v["excess_ratio"], "z": v["z"]}
                           for k, v in t["pairs"].items()}}
                for t in tol_family]}


def main():
    rs = rows()
    doc = {"study": "D-E4", "lane": "D",
           "class": "accounting_integrity_outcome_blind",
           "reads": ["sym", "t0", "q"],
           "never_reads": ["imp_1", "imp_5", "imp_15", "imp_30", "imp_60", "imp_360",
                           "pre_bps"],
           "source": SAMPLE, "dead_time_ms": DEAD_TIME_MS,
           "no_threshold_selected": True,
           "corpus_questions_answered": [
               "ABG 1.5.4 -- Independent or dependent data?",
               "STK4080 Slides 1 -- Can it be modeled as a renewal process?"],
           "populations": [analyse(rs, f) for f in FLOORS]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1))
    for pop in doc["populations"]:
        print("\n===== floor $%d  n=%d  %s" % (pop["floor_usd"], pop["n_episodes"],
                                               pop["per_symbol_n"]))
        for t in pop["tests"]:
            print("  %-38s p=%-9s holm=%-9s %s"
                  % (t["test"], t["p"], t["p_holm6"],
                     "REJECT" if t["reject_at_0_05"] else "-"))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
