# -*- coding: utf-8 -*-
"""LANE D / D-E7 -- IS THE D-E4 vs S126 DISAGREEMENT A CLOCK OR AN ESTIMATOR?

Read-only, OUTCOME-BLIND.  Reads liquidation TIMESTAMPS, sides and notionals only.
No price, no return, no outcome column is opened.

THE DISAGREEMENT
----------------
Two lanes measured the SAME episode recurrence process and reported opposite shapes.

  D-E4  (lane D, section 512)   dead-time-corrected gaps are EXPONENTIAL
                                CV 1.040 / 1.057 / 1.011, pooled within symbol 1.036, z 1.27
                                -> constant hazard, not rejected
  S126  (corpus branch, 472)    CONSTANT_HAZARD_REJECTED
                                Nelson-Aalen late/early ratio 0.628 pooled, p 0.0000
                                -> the recurrence hazard FALLS with elapsed time

Three things differ and any of them could produce it:

  CLOCK       D-E4 used START-TO-START gaps minus the 900 s dead time.
              S125/S126 re-based to END-TO-START waits, because an episode has DURATION and a
              unit "cannot fail before its own span has elapsed" -- S125 measured 38.0% of the
              risk set structurally incapable of failing, and the error is CONFOUNDED WITH THE
              REGRESSOR by construction, corr(log Q/ADV, log span) = +0.5212.
  WINDOW      S126 reads the hazard on [0.25, 1.66] h.  D-E4's CV reads the whole distribution.
  ESTIMATOR   a CV is a global shape summary; a late/early hazard ratio is a local contrast.

So the test is a 2x2: BOTH estimators on BOTH clocks, same spells, same day.  If the answer
tracks the clock, it is a clock artefact (C-T40's finding, one level down).  If it tracks the
estimator, one of the two estimators is answering a question it was not asked.  If neither, the
disagreement is real and goes to CONTRADICTION_REGISTER.md.

EVERY NULL IS CALIBRATED BEFORE ITS TEST IS READ -- D-E4's own lesson, where 2 of 6 needed it and
both changed the answer.  Here the null is: constant individual hazard, dead time as built, no
frailty.  What do BOTH statistics return in that world?

The notional floor is DECLARED, not selected: both floors are reported (D-E2's rule).

Usage:  python tools/d_e7_clock_vs_estimator_v1.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = "data/microstructure_02.db"
PICKLE = "data/pve_01_v1/_s97_extended.pkl"   # the population S125/S126/S127 and D-E4 both used
CUTOFF_MS = 1787270400000          # 2026-08-21 lawful cutoff, as every study on this sample uses
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MIN_GAP_MS = 900_000               # the episode rule's own dead time
FLOORS = (0.0, 50_000.0)           # declared, not selected
H = 3_600_000.0
CEILING_H = 1.66                   # S126's own window ceiling, reused unchanged
OUT = "reports/atlas/D_E7_CLOCK_VS_ESTIMATOR_V1.json"


def con():
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    c.execute("PRAGMA query_only=ON")
    return c


def reference_window():
    """The published population's own span.  A rebuild without it is a DIFFERENT population --
    my first cut had no lower bound and produced 2,130 spells against the published 1,268."""
    import pickle
    d = pickle.loads(open(PICKLE, "rb").read())
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    t0 = [int(r["t0"]) for r in rows]
    by = collections.defaultdict(set)
    for r in rows:
        by[r["sym"]].add(int(r["t0"]))
    return min(t0), max(t0), {k: v for k, v in by.items()}


LO_MS, HI_MS, REF_T0 = reference_window()


def episodes(floor):
    """Episodes with BOTH endpoints.  Timestamps and notionals only -- no outcome."""
    cn = con()
    out = collections.defaultdict(list)
    for s in SYMBOLS:
        r = cn.execute("SELECT ts_ms,notional FROM liquidations WHERE symbol=? "
                       "AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
                       (s, LO_MS - MIN_GAP_MS, min(HI_MS + MIN_GAP_MS, CUTOFF_MS))).fetchall()
        ts = np.array([x[0] for x in r], np.int64)
        nt = np.array([(x[1] or 0.0) for x in r], float)
        brk = np.flatnonzero(np.diff(ts) > MIN_GAP_MS) + 1
        for g in np.split(np.arange(len(ts)), brk):
            if not len(g):
                continue
            q = float(nt[g].sum())
            if q < floor:
                continue
            t0 = int(ts[g[0]])
            if t0 < LO_MS or t0 > HI_MS:
                continue
            out[s].append((t0, int(ts[g[-1]]), q))
    cn.close()
    return {s: sorted(v) for s, v in out.items()}


def verify_against_reference(eps):
    """A rebuild that does not reproduce the published t0 set is a different population.
    This is reported, never silently accepted."""
    rep = {}
    for s in sorted(REF_T0):
        mine = {e[0] for e in eps.get(s, [])}
        ref = REF_T0[s]
        rep[s] = {"rebuilt": len(mine), "published": len(ref),
                  "matched": len(mine & ref),
                  "in_published_not_rebuilt": len(ref - mine),
                  "in_rebuilt_not_published": len(mine - ref)}
    ok = all(v["in_published_not_rebuilt"] == 0 and v["in_rebuilt_not_published"] == 0
             for v in rep.values())
    return {"exact_reproduction": ok, "per_symbol": rep}


def waits(eps):
    """Both clocks, per symbol, in hours.  start-to-start and end-to-start."""
    ss, es, span = {}, {}, {}
    for s, v in eps.items():
        st = np.array([x[0] for x in v], np.int64)
        en = np.array([x[1] for x in v], np.int64)
        ss[s] = np.diff(st) / H
        es[s] = (st[1:] - en[:-1]) / H
        span[s] = (en - st) / H
    return ss, es, span


# ---------------------------------------------------------------- estimator 1
def cv_stat(w_by_sym, dead_h):
    """D-E4's estimator: CV of dead-time-corrected waits, pooled WITHIN symbol."""
    parts, per = [], {}
    for s, w in sorted(w_by_sym.items()):
        g = w - dead_h
        g = g[g > 0]
        if len(g) < 30:
            continue
        cv = float(g.std(ddof=1) / g.mean())
        per[s] = {"n": int(len(g)), "cv": round(cv, 4),
                  "z_vs_1": round((cv - 1.0) / (1.0 / math.sqrt(len(g))), 2)}
        parts.append(g / g.mean())
    p = np.concatenate(parts)
    cv = float(p.std(ddof=1) / p.mean())
    return {"estimator": "CV_of_dead_time_corrected_waits",
            "pooled_within_symbol_cv": round(cv, 4),
            "z_vs_1": round((cv - 1.0) / (1.0 / math.sqrt(len(p))), 2),
            "n": int(len(p)), "per_symbol": per}


# ---------------------------------------------------------------- estimator 2
def na_ratio(w_by_sym, entry_by_sym, floor_h, ceiling_h=CEILING_H):
    """S126's estimator: Nelson-Aalen with DELAYED ENTRY, late/early hazard ratio.

    Y_i(t) = 0 for t < entry_i  -- S125's repair, applied here unchanged.
    Ratio = A(hi)-A(mid) over the late half divided by A(mid)-A(floor) over the early half,
    each divided by its own width, so it is a hazard-per-hour ratio.
    """
    per, num_e, num_l, den_e, den_l = {}, 0.0, 0.0, 0.0, 0.0
    mid = 0.5 * (floor_h + ceiling_h)
    for s in sorted(w_by_sym):
        w, ent = w_by_sym[s], entry_by_sym[s]
        n = min(len(w), len(ent))
        w, ent = np.asarray(w[:n], float), np.asarray(ent[:n], float)
        ev = np.sort(w[(w >= floor_h) & (w <= ceiling_h)])
        if len(ev) < 30:
            continue
        a_early = a_late = 0.0
        for t in ev:
            risk = int(((w >= t) & (ent <= t)).sum())
            if risk <= 0:
                continue
            if t <= mid:
                a_early += 1.0 / risk
            else:
                a_late += 1.0 / risk
        e_rate = a_early / (mid - floor_h)
        l_rate = a_late / (ceiling_h - mid)
        per[s] = {"n_events": int(len(ev)), "early_per_h": round(e_rate, 4),
                  "late_per_h": round(l_rate, 4),
                  "late_over_early": round(l_rate / e_rate, 4) if e_rate > 0 else None}
        num_e += a_early
        num_l += a_late
        den_e += 1.0
        den_l += 1.0
    pe = num_e / (mid - floor_h)
    pl = num_l / (ceiling_h - mid)
    return {"estimator": "nelson_aalen_late_over_early_with_delayed_entry",
            "window_h": [floor_h, ceiling_h], "midpoint_h": mid,
            "pooled_late_over_early": round(pl / pe, 4) if pe > 0 else None,
            "per_symbol": per}


# ---------------------------------------------------------------- the null
def calibrate(eps, dead_h, floor_h, reps=300, seed=20260827):
    """CONSTANT individual hazard, the dead time as built, NO frailty.
    What do BOTH statistics return in that world?"""
    rng = np.random.default_rng(seed)
    cvs, ratios = [], []
    rates, spans, ns = {}, {}, {}
    for s, v in eps.items():
        st = np.array([x[0] for x in v], np.int64)
        en = np.array([x[1] for x in v], np.int64)
        w = (st[1:] - en[:-1]) / H
        rates[s] = float((w[w > dead_h] - dead_h).mean())
        spans[s] = (en - st) / H
        ns[s] = len(v)
    for _ in range(reps):
        sim_es, sim_ss, sim_entry = {}, {}, {}
        for s in eps:
            n = ns[s] - 1
            e = dead_h + rng.exponential(rates[s], n)
            sp = rng.choice(spans[s], size=n, replace=True)   # spans resampled, not modelled
            sim_es[s] = e
            sim_ss[s] = e + sp
            sim_entry[s] = np.zeros(n)
        cvs.append(cv_stat(sim_es, dead_h)["pooled_within_symbol_cv"])
        r = na_ratio(sim_es, sim_entry, floor_h)["pooled_late_over_early"]
        if r:
            ratios.append(r)
    return {"world": "constant individual hazard + built-in dead time + NO frailty; "
                     "episode spans resampled from the data",
            "reps": reps,
            "null_cv_mean": round(float(np.mean(cvs)), 4),
            "null_cv_sd": round(float(np.std(cvs, ddof=1)), 4),
            "null_late_over_early_mean": round(float(np.mean(ratios)), 4),
            "null_late_over_early_sd": round(float(np.std(ratios, ddof=1)), 4),
            "null_late_over_early_p05_p95": [round(float(np.percentile(ratios, 5)), 4),
                                             round(float(np.percentile(ratios, 95)), 4)]}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    res = {"study": "D-E7", "lane": "D", "class": "accounting_integrity_outcome_blind",
           "reads": ["liquidations.ts_ms", "liquidations.notional"],
           "never_reads": ["any price, any return, any outcome column"],
           "reconciles": ["D-E4 / section 512 (lane D)", "S126 / section 472 (corpus branch)"],
           "populations": []}

    for floor in FLOORS:
        eps = episodes(floor)
        ss, es, span = waits(eps)
        dead_ss = MIN_GAP_MS / H                    # start-to-start floor is the dead time
        dead_es = MIN_GAP_MS / H                    # end-to-start floor is the SAME by the rule
        zero = {s: np.zeros(len(v) - 1) for s, v in eps.items()}
        # start-to-start: a unit cannot fail before its own span -- S125's delayed entry
        entry_ss = {s: span[s][:-1] for s in eps}

        cell = {
            "floor_usd": floor,
            "reference_window_utc_ms": [LO_MS, HI_MS],
            "rebuild_vs_published_population": verify_against_reference(eps),
            "n_episodes": {s: len(v) for s, v in sorted(eps.items())},
            "n_spells": {s: len(v) - 1 for s, v in sorted(eps.items())},
            "span_hours_p50": {s: round(float(np.median(span[s])), 4) for s in sorted(eps)},
            "cell_A_CV_on_start_to_start": cv_stat(ss, dead_ss),
            "cell_B_CV_on_end_to_start": cv_stat(es, dead_es),
            "cell_C_NA_on_start_to_start_delayed_entry":
                na_ratio(ss, entry_ss, dead_ss),
            "cell_D_NA_on_end_to_start": na_ratio(es, zero, dead_es),
            "null_calibration": calibrate(eps, dead_es, dead_es),
        }
        res["populations"].append(cell)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))

    for p in res["populations"]:
        print("\n===== floor $%d   spells %s" % (p["floor_usd"], p["n_spells"]))
        print("  span hours p50            %s" % p["span_hours_p50"])
        print("  A  CV   start-to-start    %.4f   z %+.2f"
              % (p["cell_A_CV_on_start_to_start"]["pooled_within_symbol_cv"],
                 p["cell_A_CV_on_start_to_start"]["z_vs_1"]))
        print("  B  CV   end-to-start      %.4f   z %+.2f"
              % (p["cell_B_CV_on_end_to_start"]["pooled_within_symbol_cv"],
                 p["cell_B_CV_on_end_to_start"]["z_vs_1"]))
        print("  C  NA   start-to-start    late/early %s" %
              p["cell_C_NA_on_start_to_start_delayed_entry"]["pooled_late_over_early"])
        print("  D  NA   end-to-start      late/early %s" %
              p["cell_D_NA_on_end_to_start"]["pooled_late_over_early"])
        n = p["null_calibration"]
        print("  NULL (constant hazard, no frailty):  CV %.4f +/- %.4f   late/early %.4f +/- %.4f  p05-p95 %s"
              % (n["null_cv_mean"], n["null_cv_sd"],
                 n["null_late_over_early_mean"], n["null_late_over_early_sd"],
                 n["null_late_over_early_p05_p95"]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
