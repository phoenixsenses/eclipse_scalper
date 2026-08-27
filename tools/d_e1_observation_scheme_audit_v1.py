# -*- coding: utf-8 -*-
"""LANE D / D-E1 -- WHAT ENDED THE OBSERVATIONS.

Read-only, OUTCOME-BLIND audit of the observation scheme behind the forced-flow
line (h2).  It reads liquidation timestamps and mark-price TIMESTAMPS only; it
never forms a return, a mean, or any outcome quantity.  Accounting / data
integrity class (CLAUDE.md: N-non-consuming work item (b)).

It answers, for the sample H2 actually used:

  1. LEFT TRUNCATION   -- who was excluded by delayed entry, and when.
  2. RIGHT CENSORING   -- who was excluded by the data edge / lawful cutoff.
  3. HORIZON SLIP      -- the coverage rule checks only the price series
                          ENDPOINTS, and the horizon reader takes the first mark
                          at-or-after t0+h with no tolerance.  So an internal
                          feed gap does not censor an observation: it silently
                          moves the clock.  This measures how far.
  4. COMPETING EVENT   -- how often the next episode of the same symbol starts
                          inside the response window (the third competing risk
                          named in LANE_CHARTERS_V1.md), as a raw incidence.
  5. RISK SET vs CLUSTER -- `components()` in the H2 driver is a clustering for
                          standard errors, not a risk set.  This prints both so
                          the two counts can never again be confused.

Usage:  python tools/d_e1_observation_scheme_audit_v1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.h2_response_shape_driver import (  # noqa: E402
    ALL_H_MIN, ADV_DAYS, CUTOFF_MS, MIN_ADV_DAYS, MIN_COMPONENTS,
    MIN_FUTURE_MARKS, MIN_PRIOR_MARKS, PRE_VOL_MIN, SYMBOLS,
    _con, _load_series, anchors, components)
from tools.coverage_invariant import event_is_measurable  # noqa: E402

OUT = "reports/atlas/D_E1_OBSERVATION_SCHEME_V1.json"


def q(a, ps=(0, 50, 75, 90, 95, 99, 100)):
    if not len(a):
        return {}
    a = np.asarray(a, float)
    return {("p%g" % p): float(np.percentile(a, p)) for p in ps}


def main():
    con = _con()
    px, adv, liqts = _load_series(con)
    a_list = anchors(con)
    con.close()

    hmax_ms = max(ALL_H_MIN) * 60_000
    lb_ms = PRE_VOL_MIN * 60_000

    attr = {"raw_episodes": len(a_list), "coverage": 0, "adv_history": 0,
            "complete_case": 0}
    cov_reason = {}
    trunc_days, cens_days = [], []
    kept = []

    for e in a_list:
        s, t0 = e["sym"], e["t0"]
        a, _c = px[s]
        i0 = np.searchsorted(a, t0, side="right") - 1
        ok, why = event_is_measurable(
            t0, [lb_ms], [hmax_ms],
            int(a[0]) if len(a) else None, int(a[-1]) if len(a) else None,
            CUTOFF_MS)
        if i0 < MIN_PRIOR_MARKS or i0 + MIN_FUTURE_MARKS >= len(a) or not ok:
            attr["coverage"] += 1
            r = why if not ok else ("row_count_prior" if i0 < MIN_PRIOR_MARKS
                                    else "row_count_future")
            cov_reason[r] = cov_reason.get(r, 0) + 1
            cens_days.append((s, t0 // 86400000))
            continue
        dd, vv = adv[s]
        if (dd < (t0 // 86400000)).sum() < MIN_ADV_DAYS:
            attr["adv_history"] += 1
            trunc_days.append((s, t0 // 86400000))
            continue
        kept.append(e)
    attr["complete_case"] = len(kept)

    # ---- 3. horizon slip, measured with the driver's own reader semantics
    slip = {("%dm" % h): [] for h in ALL_H_MIN}
    unresolved = {("%dm" % h): 0 for h in ALL_H_MIN}
    for e in kept:
        a, _c = px[e["sym"]]
        for h in ALL_H_MIN:
            tgt = e["t0"] + h * 60_000
            j = np.searchsorted(a, tgt, side="left")
            if j >= len(a):
                unresolved["%dm" % h] += 1
                continue
            slip["%dm" % h].append(int(a[j] - tgt))

    # ---- internal gap structure of the mark series (the thing that makes slip)
    gaps = {}
    for s in SYMBOLS:
        d = np.diff(px[s][0])
        gaps[s] = {"n_marks": int(len(px[s][0])),
                   "span_days": round((px[s][0][-1] - px[s][0][0]) / 86400000.0, 2),
                   "gap_ms": q(d), "n_gaps_over_60s": int((d > 60_000).sum()),
                   "n_gaps_over_600s": int((d > 600_000).sum()),
                   "max_gap_minutes": round(float(d.max()) / 60000.0, 1)}

    # ---- 4. competing event: next same-symbol episode inside the window
    comp_risk = {}
    by_sym = {}
    for e in kept:
        by_sym.setdefault(e["sym"], []).append(e["t0"])
    t0_all = np.array([e["t0"] for e in kept], np.int64)
    _gaps_min = []
    for s, ts in by_sym.items():
        ts = np.sort(np.array(ts, np.int64))
        if len(ts) > 1:
            _gaps_min.extend((np.diff(ts) / 60000.0).tolist())
    inter_episode_minutes = q(_gaps_min, (5, 10, 25, 50, 75, 90, 95))
    inter_episode_minutes["n_gaps"] = len(_gaps_min)
    for h in ALL_H_MIN:
        hit = 0
        for s, ts in by_sym.items():
            ts = np.sort(np.array(ts, np.int64))
            nxt = np.append(np.diff(ts), 10 ** 18)
            hit += int((nxt <= h * 60_000).sum())
        comp_risk["%dm" % h] = {"n_with_next_episode_inside_window": hit,
                                "share": round(hit / max(1, len(kept)), 4)}

    # ---- 5. risk set vs cluster count
    rs = {}
    for h in ALL_H_MIN:
        ncl = int(len(np.unique(components(t0_all, h * 60_000))))
        rs["%dm" % h] = {
            "CLUSTERS_for_standard_errors": ncl,
            "RISK_SET_episodes_still_observed": len(kept) - unresolved["%dm" % h],
            "cluster_verdict": ("SUPPORTED" if ncl >= MIN_COMPONENTS
                                else "SUPPORT_INADEQUATE")}

    def first_days(pairs):
        out = {}
        for s, d in pairs:
            out.setdefault(s, []).append(int(d))
        return {s: {"n": len(v), "utc_days": sorted(set(v))} for s, v in out.items()}

    res = {
        "study": "D-E1", "lane": "D", "class": "accounting_integrity_outcome_blind",
        "reads": ["liquidations.ts_ms", "mark_prices.ts_ms", "agg_trades day index"],
        "reads_no_outcome": True,
        "attrition": attr,
        "coverage_refusal_reasons": cov_reason,
        "left_truncation_delayed_entry": {
            "rule": "at least %d prior UTC days of daily notional strictly before "
                    "the episode's day (ADV window %d)" % (MIN_ADV_DAYS, ADV_DAYS),
            "n_excluded": attr["adv_history"],
            "share_of_raw": round(attr["adv_history"] / max(1, len(a_list)), 4),
            "by_symbol": first_days(trunc_days)},
        "right_censoring_data_edge": {
            "rule": "required window [t0-%dm, t0+%dm] must lie inside the symbol's "
                    "price span AND end before the lawful cutoff; plus row-count "
                    "margins %d prior / %d future"
                    % (PRE_VOL_MIN, max(ALL_H_MIN), MIN_PRIOR_MARKS,
                       MIN_FUTURE_MARKS),
            "n_excluded": attr["coverage"],
            "share_of_raw": round(attr["coverage"] / max(1, len(a_list)), 4),
            "by_symbol": first_days(cens_days)},
        "horizon_slip_ms": {k: q(v) for k, v in slip.items()},
        "horizon_slip_over_60s": {
            k: int((np.asarray(v) > 60_000).sum()) for k, v in slip.items()},
        "horizon_unresolved": unresolved,
        "mark_series_gaps": gaps,
        "competing_event_next_episode_in_window": comp_risk,
        "inter_episode_gap_minutes_same_symbol": inter_episode_minutes,
        "risk_set_vs_cluster": rs,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, sort_keys=False))
    print(json.dumps(res, indent=1)[:6000])
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
