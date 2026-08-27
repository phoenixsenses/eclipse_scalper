# -*- coding: utf-8 -*-
"""LANE D / D-E2 -- IS A SPELL A MARKET OBJECT OR A DETECTOR ARTEFACT?

Read-only, OUTCOME-BLIND.  Reads only `sym`, `t0` and `q` from two already-built
episode tables.  It never touches `imp_*` or `pre_bps` -- the outcome columns are
present in both pickles and are deliberately not read.

WHY
---
`S101` (SYSTEM_STATE 437) discharged two Honore (1993) MPH conditions and measured
lagged duration dependence on "the forced-flow episode sample": 1,271 episodes.
`D-E1` reproduced "the forced-flow episode sample" behind the published H2 response
curve: 629 episodes.  Same 3 symbols, same 24 UTC days.  Two populations, one name.

The only difference is the notional floor.  This quantifies what that floor does to
the SPELL PROCESS -- the object lane D's whole question is about:

  1. spell counts and per-symbol spells per unit (Honore's premise)
  2. the inter-episode gap distribution (D-E1's CIF)
  3. LAGGED DURATION DEPENDENCE, log gap_k ~ log gap_{k-1} within symbol, with the
     80%-power MDE -- S101's `CONDITIONAL` verdict rests on this number
  4. a floor sweep, so the threshold-dependence is a curve and not two points

NO THRESHOLD IS SELECTED.  The sweep is a sensitivity analysis of a measurement;
nothing here proposes an entry rule, a filter or a family.

Usage:  python tools/d_e2_spell_process_audit_v1.py
"""
from __future__ import annotations

import collections
import json
import math
import os
import pickle

import numpy as np

SAMPLES = {
    "S101_s97_extended": "data/pve_01_v1/_s97_extended.pkl",
    "H2_h1_deep": "data/pve_01_v1/_h1_deep.pkl",
}
OUT = "reports/atlas/D_E2_SPELL_PROCESS_V1.json"
POWER_Z = 2.80          # z(0.975) + z(0.80), S101's constant, reused unchanged
WINDOWS_MIN = (1, 5, 15, 30, 60, 120, 240, 360)


def rows_of(path):
    d = pickle.loads(open(path, "rb").read())
    r = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return [{"sym": x["sym"], "t0": int(x["t0"]), "q": float(x["q"])} for x in r]


def ols1(x, y):
    """Slope, its SE, and n.  S101's estimator, reimplemented identically."""
    n = len(x)
    if n < 5:
        return None
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum()
    if sxx <= 0:
        return None
    b = ((x - mx) * (y - my)).sum() / sxx
    res = y - (my + b * (x - mx))
    s2 = (res ** 2).sum() / (n - 2)
    se = math.sqrt(s2 / sxx)
    return {"beta": float(b), "se": float(se), "z": float(b / se) if se else None,
            "n_pairs": int(n), "mde_80pct": float(POWER_Z * se)}


def gaps_by_symbol(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sym"]].append(r["t0"])
    out = {}
    for s, ts in by.items():
        ts = np.sort(np.array(ts, np.int64))
        out[s] = np.diff(ts) / 60000.0 if len(ts) > 1 else np.array([])
    return out


def q(a, ps=(5, 10, 25, 50, 75, 90, 95)):
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {("p%g" % p): round(float(np.percentile(a, p)), 3) for p in ps}


def ldd(gaps):
    """log spell_k ~ log spell_{k-1}, pooled within symbol (symbol-demeaned)."""
    xs, ys = [], []
    for s, g in gaps.items():
        if len(g) < 3:
            continue
        lg = np.log(g[g > 0])
        if len(lg) < 3:
            continue
        a, b = lg[:-1], lg[1:]
        xs.append(a - a.mean())
        ys.append(b - b.mean())
    if not xs:
        return None
    return ols1(np.concatenate(xs), np.concatenate(ys))


def ldd_per_symbol(gaps):
    out = {}
    for s, g in sorted(gaps.items()):
        lg = np.log(g[g > 0]) if len(g) else np.array([])
        out[s] = ols1(lg[:-1], lg[1:]) if len(lg) >= 5 else None
    return out


def competing_cif(rows):
    gaps = gaps_by_symbol(rows)
    n = len(rows)
    out = {}
    for w in WINDOWS_MIN:
        hit = sum(int((g <= w).sum()) for g in gaps.values())
        out["%dm" % w] = round(hit / max(1, n), 4)
    return out


def describe(name, rows):
    per = collections.Counter(r["sym"] for r in rows)
    gaps = gaps_by_symbol(rows)
    allg = np.concatenate([g for g in gaps.values() if len(g)])
    qs = np.array([r["q"] for r in rows])
    days = len({r["t0"] // 86400000 for r in rows})
    return {
        "sample": name, "n_episodes": len(rows), "utc_days": days,
        "per_symbol": dict(sorted(per.items())),
        "spells_per_unit": {s: len(g) for s, g in sorted(gaps.items())},
        "q_usd": {"min": round(float(qs.min()), 0), "p50": round(float(np.median(qs)), 0),
                  "p90": round(float(np.percentile(qs, 90)), 0)},
        "inter_episode_gap_minutes": q(allg),
        "competing_event_cif": competing_cif(rows),
        "lagged_duration_dependence_pooled_within_symbol": ldd(gaps),
        "lagged_duration_dependence_per_symbol": ldd_per_symbol(gaps),
    }


def main():
    res = {"study": "D-E2", "lane": "D",
           "class": "accounting_integrity_outcome_blind",
           "reads": ["sym", "t0", "q"],
           "never_reads": ["imp_1", "imp_5", "imp_15", "imp_30", "imp_60", "imp_360",
                           "pre_bps"],
           "no_threshold_selected": True,
           "samples": {}}

    base = rows_of(SAMPLES["S101_s97_extended"])
    for name, path in SAMPLES.items():
        res["samples"][name] = describe(name, rows_of(path))
        res["samples"][name]["path"] = path

    # ---- floor sweep on the SUPERSET, so every point is the same underlying feed
    sweep = []
    for floor in (0, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000):
        sub = [r for r in base if r["q"] >= floor]
        if len(sub) < 30:
            continue
        gaps = gaps_by_symbol(sub)
        allg = np.concatenate([g for g in gaps.values() if len(g)])
        L = ldd(gaps)
        sweep.append({
            "floor_usd": floor, "n_episodes": len(sub),
            "median_gap_min": round(float(np.median(allg)), 2),
            "p25_gap_min": round(float(np.percentile(allg, 25)), 2),
            "cif_60m": competing_cif(sub)["60m"],
            "cif_240m": competing_cif(sub)["240m"],
            "ldd_beta": None if not L else round(L["beta"], 4),
            "ldd_z": None if not L else round(L["z"], 2),
            "ldd_mde80": None if not L else round(L["mde_80pct"], 4),
        })
    res["notional_floor_sweep"] = sweep

    a = res["samples"]["S101_s97_extended"]
    b = res["samples"]["H2_h1_deep"]
    res["headline"] = {
        "two_populations_one_name": True,
        "n": [a["n_episodes"], b["n_episodes"]],
        "median_q_ratio": round(b["q_usd"]["p50"] / a["q_usd"]["p50"], 2),
        "median_gap_min": [a["inter_episode_gap_minutes"].get("p50"),
                           b["inter_episode_gap_minutes"].get("p50")],
        "ldd_mde80": [None if not a["lagged_duration_dependence_pooled_within_symbol"]
                      else round(a["lagged_duration_dependence_pooled_within_symbol"]["mde_80pct"], 4),
                      None if not b["lagged_duration_dependence_pooled_within_symbol"]
                      else round(b["lagged_duration_dependence_pooled_within_symbol"]["mde_80pct"], 4)],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
