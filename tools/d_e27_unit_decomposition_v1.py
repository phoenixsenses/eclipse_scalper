# -*- coding: utf-8 -*-
"""D-E27 -- A-S65 says our contamination numbers differ by UNIT.  Is that one difference or two?

A-S65 (delivered only after the D-E25 reader repair) reports contamination of 99.9% at no floor
and 76.2% at a $500k floor, against D-E4's 47.5% and 12.3%, and attributes the whole gap to the
counting unit: *"your lambda is per EPISODE and mine is per LIQUIDATION"*.

That explanation is plausible and it is not verified.  There are TWO differences between the two
constructions, not one, and they are easy to confuse:

  UNIT      what counts as an arrival -- a raw liquidation, or an EPISODE (a cluster of
            liquidations separated by a >= 15 min gap).
  FLOOR     where the size floor is applied -- to an INDIVIDUAL liquidation's notional, or to the
            SUM of notional over an episode.  A $500k floor on a sum admits clusters of small
            prints that a $500k floor per print rejects, so the two select different populations
            even at an identical counting unit.

A third thing is folded in and worth separating: D-E4's number came from a CLOSED FORM,
1 - exp(-lambda w), which assumes a Poisson process.  D-E4 itself measured that the dead time
makes the Poisson null wrong for this data (the null was 0.68-0.72 where Poisson said 1).  So the
closed form may be wrong for the liquidation-level process even at the right unit.  Every cell
below reports the EMPIRICAL probability beside the closed form and the ratio between them.

  2 units  x  2 floor placements  x  2 floors  x  3 symbols, plus a pooled row.

SCOPE FENCE.  This is a UNIT-SENSITIVITY measurement for a number two lanes disagree about.  It is
outcome-blind: liquidation timestamps and notionals only, no price, no return, no P&L.  It does
NOT change D-E8's frozen estimand; mu_tau stays as preregistered and any effect on it is reported
as a declared sensitivity, never as a replacement.

Usage:  python tools/d_e27_unit_decomposition_v1.py
"""
from __future__ import annotations

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
EPISODE_GAP_MS = 900_000          # the >= 15 min gap that defines an episode, as in D-E4/D-E13
WINDOW_MIN = 60                   # A-S65's window
FLOORS = (0.0, 500_000.0)
OUT = os.path.join(ROOT, "reports", "atlas", "D_E27_UNIT_DECOMPOSITION_V1.json")


def raw(symbol):
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    r = cn.execute("SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND ts_ms<? "
                   "ORDER BY ts_ms", (symbol, CUTOFF_MS)).fetchall()
    cn.close()
    ts = np.array([x[0] for x in r], np.int64)
    nt = np.array([(x[1] or 0.0) for x in r], float)
    return ts, nt


def arrivals(ts, nt, unit, floor_where, floor):
    """Arrival times under a declared (unit, floor placement, floor)."""
    if unit == "liquidation":
        if floor_where == "individual":
            m = nt >= floor
            return ts[m]
        # a floor on an episode SUM, but still counting individual liquidations inside the
        # episodes that qualify -- this is the cell that separates UNIT from FLOOR PLACEMENT
        keep = []
        brk = np.flatnonzero(np.diff(ts) > EPISODE_GAP_MS) + 1
        for g in np.split(np.arange(len(ts)), brk):
            if len(g) and float(nt[g].sum()) >= floor:
                keep.append(ts[g])
        return np.sort(np.concatenate(keep)) if keep else np.array([], np.int64)

    # unit == "episode"
    if floor_where == "individual":
        m = nt >= floor
        ts2, nt2 = ts[m], nt[m]
    else:
        ts2, nt2 = ts, nt
    if not len(ts2):
        return np.array([], np.int64)
    brk = np.flatnonzero(np.diff(ts2) > EPISODE_GAP_MS) + 1
    out = []
    for g in np.split(np.arange(len(ts2)), brk):
        if not len(g):
            continue
        if floor_where == "episode_sum" and float(nt2[g].sum()) < floor:
            continue
        out.append(int(ts2[g[0]]))
    return np.sort(np.array(out, np.int64))


def contamination(t, w_ms, dead_ms):
    """P(another arrival within w), EMPIRICAL beside D-E4's DEAD-TIME closed form.

    D-E4 published `1 - exp(-lambda (w - 900s))`, not plain Poisson: episodes are separated by at
    least the 15-minute gap BY CONSTRUCTION, so the process has a dead time and both the rate
    estimator and the exponent must carry it.  A first version of this script compared against
    plain Poisson and would have "refuted" an instrument it had not actually implemented -- the
    exact failure this lane has been cataloguing all day.  For the LIQUIDATION unit the dead time
    is zero, so the same expression collapses to ordinary Poisson, which is correct there.

    Rate estimator matched to the process: for a dead-time renewal process the mean gap is
    dead + 1/lambda, so lambda_hat = 1 / (mean_gap - dead).
    """
    if len(t) < 2:
        return None
    nxt = np.diff(t)
    emp = float((nxt <= w_ms).mean())
    mean_gap = float(nxt.mean())
    eff = mean_gap - dead_ms
    lam = (1.0 / eff) if eff > 0 else float("nan")
    w_eff = max(0.0, w_ms - dead_ms)
    closed = (1.0 - math.exp(-lam * w_eff)) if np.isfinite(lam) else float("nan")
    return {"n": int(len(t)), "empirical": round(emp, 4), "closed_form": round(closed, 4),
            "closed_over_empirical": round(closed / emp, 3) if emp > 0 else None,
            "dead_ms": int(dead_ms),
            "lambda_per_min": round(lam * 60000.0, 5) if np.isfinite(lam) else None,
            "mean_gap_min": round(mean_gap / 60000.0, 2),
            "median_gap_min": round(float(np.median(nxt)) / 60000.0, 2)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    w = WINDOW_MIN * 60000
    data = {s: raw(s) for s in SYMBOLS}
    res = {"study": "D-E27", "lane": "D", "window_min": WINDOW_MIN,
           "episode_gap_min": EPISODE_GAP_MS / 60000,
           "answers": "A-S65 unit mismatch", "class": "unit_sensitivity_outcome_blind",
           "cells": {}}

    print("D-E27  is the gap ONE difference (unit) or TWO (unit + floor placement)?")
    print("       window = %d min   episode gap = %d min\n" % (WINDOW_MIN, EPISODE_GAP_MS / 60000))
    hdr = "%-9s %-12s %-12s %8s %7s %9s %9s %8s"
    print(hdr % ("symbol", "unit", "floor@", "floor$k", "n", "EMPIRICAL", "closed", "cf/emp"))
    for floor in FLOORS:
        for unit in ("liquidation", "episode"):
            for where in ("individual", "episode_sum"):
                if floor == 0.0 and where == "episode_sum":
                    continue                       # at a zero floor the placement cannot matter
                for s in SYMBOLS:
                    ts, nt = data[s]
                    t = arrivals(ts, nt, unit, where, floor)
                    dead = EPISODE_GAP_MS if unit == "episode" else 0
                    c = contamination(t, w, dead)
                    if not c:
                        continue
                    key = "%s|%s|%s|%d" % (s, unit, where, int(floor))
                    res["cells"][key] = c
                    print(hdr % (s, unit, where, int(floor / 1000), c["n"],
                                 "%.4f" % c["empirical"], "%.4f" % c["closed_form"],
                                 ("%.2f" % c["closed_over_empirical"])
                                 if c["closed_over_empirical"] else "-"))
                print()

    # ---- the decomposition A-S65's claim needs
    def cell(s, unit, where, floor):
        return res["cells"].get("%s|%s|%s|%d" % (s, unit, where, int(floor)))

    print("DECOMPOSITION at the $500k floor, per symbol (empirical probabilities)")
    print("  liq@individual -> liq@episode_sum  isolates FLOOR PLACEMENT at a fixed unit")
    print("  liq@episode_sum -> epi@episode_sum isolates UNIT at a fixed floor placement")
    dec = {}
    for s in SYMBOLS:
        a = cell(s, "liquidation", "individual", 500_000)
        b = cell(s, "liquidation", "episode_sum", 500_000)
        c = cell(s, "episode", "episode_sum", 500_000)
        if not (a and b and c):
            continue
        dec[s] = {"liq_individual": a["empirical"], "liq_episode_sum": b["empirical"],
                  "epi_episode_sum": c["empirical"],
                  "floor_placement_effect": round(b["empirical"] - a["empirical"], 4),
                  "unit_effect": round(c["empirical"] - b["empirical"], 4),
                  "total": round(c["empirical"] - a["empirical"], 4)}
        print("  %-9s %.4f -> %.4f -> %.4f    placement %+0.4f   unit %+0.4f   total %+0.4f"
              % (s, a["empirical"], b["empirical"], c["empirical"],
                 dec[s]["floor_placement_effect"], dec[s]["unit_effect"], dec[s]["total"]))
    res["decomposition_500k"] = dec

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
