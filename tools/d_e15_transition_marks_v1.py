# -*- coding: utf-8 -*-
"""D-E15 -- can the competing-risks framework carry a MARK?  Lane A asked; this answers.

A-S72: *"if your competing-risk framework can carry a MARK (the P&L at each transition) rather than
only the transition times, that mark is this number"* -- the number being the increment from t+18
to t+60, whose median A measured as zero.

A marked point process is the standard object: each transition carries a time AND a value.  This
attaches the signed return at the transition to every row D-E10 already produced, and asks three
things in order, the first of which is a check on MY OWN estimator rather than a finding.

  M1  STRUCTURAL SELF-CHECK.  EDGE_GONE is DEFINED as the signed return crossing back below
      k = 10 bps.  So its mark MUST be pinned at approximately k.  If it is not, the estimator is
      wrong, not the market.  This runs first and the rest is not read unless it passes.
  M2  WHICH MARKS CARRY INFORMATION.  A mark fixed by construction carries none.  The free marks
      are at ADMINISTRATIVE (still alive at tau) and INTERRUPTED (a new episode arrived first).
  M3  WAS THE TIME WORTH ANYTHING.  Maximum favourable excursion DURING the alive spell against the
      mark at exit.  This is descriptive and it is the honest form of A's question: mu_tau measures
      TIME, and time x edge is not P&L because the path inside the spell is not monotone.

SCOPE FENCE, stated because M3 is close to something forbidden.  NO exit rule is proposed, NO
threshold is selected, NO horizon is chosen.  The estate's graveyard already contains partial exit
and tight stop, and nothing here reopens them.  This is a DESCRIPTIVE marking of an existing,
frozen decomposition -- it is not a preregistered test and it spends no alpha.

Usage:  python tools/d_e15_transition_marks_v1.py
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e8_evaluator_v1 import (  # noqa: E402
    CUTOFF_MS, DB, FLOOR_PRIMARY, K_BPS, SLIP_TOLERANCE_MS, TAU_MIN,
    alive_spell, assert_spec_unchanged, episodes, marks)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E15_TRANSITION_MARKS_V1.json")


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {("p%g" % p): round(float(np.percentile(a, p)), 3) for p in ps}


def build_marked(floor, k_bps):
    """Every row D-E10 produced, plus the signed return AT the transition and the MFE inside."""
    tau_ms = int(TAU_MIN * 60000)
    eps = episodes(floor)
    cn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=300)
    cn.execute("PRAGMA query_only=ON")
    rows = []
    for s, v in eps.items():
        ms, px = marks(cn, s)
        t0s = np.array([x[0] for x in v], np.int64)
        nxt = np.append(t0s[1:], np.int64(1 << 62))
        for (t0, d, qv), nt in zip(v, nxt):
            i0 = np.searchsorted(ms, t0, side="right") - 1
            if i0 < 1:
                continue
            j = np.searchsorted(ms, t0 + tau_ms, side="left")
            if j >= len(ms) or (ms[j] - (t0 + tau_ms)) > SLIP_TOLERANCE_MS:
                continue
            p_ref = float(px[i0 - 1])
            cause, tt = alive_spell(ms, px, t0, p_ref, d, k_bps, tau_ms)
            if nt - t0 < tau_ms and (cause == "ADMINISTRATIVE" or tt > nt - t0):
                cause, tt = "INTERRUPTED", int(nt - t0)
            # the MARK: signed return at the transition time
            jt = np.searchsorted(ms, t0 + tt, side="left")
            jt = min(jt, len(ms) - 1)
            mark = float(d * (px[jt] / p_ref - 1.0) * 1e4)
            # MFE inside [t0, transition]
            seg = px[i0:max(i0 + 1, jt + 1)]
            r_seg = d * (seg / p_ref - 1.0) * 1e4
            mfe = float(r_seg.max()) if len(r_seg) else float("nan")
            rows.append({"sym": s, "cause": cause, "t_min": tt / 60000.0,
                         "mark_bps": mark, "mfe_bps": mfe, "qv": float(qv),
                         "day": t0 // 86400000})
    cn.close()
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E15  transition marks   prereg sha256 %s VERIFIED" % h[:16])
    rows = build_marked(FLOOR_PRIMARY, K_BPS)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cause"]].append(r)

    # ---- M1 structural self-check
    eg = np.array([r["mark_bps"] for r in by.get("EDGE_GONE", [])], float)
    pinned = bool(len(eg) and abs(float(np.median(eg)) - K_BPS) < 2.0
                  and float(np.percentile(np.abs(eg - K_BPS), 90)) < 6.0)
    print("\nM1  STRUCTURAL SELF-CHECK -- the EDGE_GONE mark must be pinned near k = %.1f bps"
          % K_BPS)
    print("    n=%d  median %.3f  p05-p95 %s  |mark-k| p90 %.3f   -> %s"
          % (len(eg), float(np.median(eg)), [round(float(np.percentile(eg, 5)), 2),
                                             round(float(np.percentile(eg, 95)), 2)],
             float(np.percentile(np.abs(eg - K_BPS), 90)),
             "PASS" if pinned else "FAIL -- estimator is wrong, not the market"))
    if not pinned:
        raise SystemExit("REFUSED: M1 failed; nothing below is read.")

    # ---- M2 which marks are free
    print("\nM2  MARKS BY TRANSITION TYPE  (bps, signed by the episode's own direction)")
    m2 = {}
    for cause in ("EDGE_GONE", "INTERRUPTED", "ADMINISTRATIVE", "NEVER_ALIVE"):
        v = by.get(cause, [])
        if not v:
            continue
        mk = np.array([x["mark_bps"] for x in v], float)
        m2[cause] = {"n": len(v), "share": round(len(v) / len(rows), 4),
                     "mark_mean": round(float(mk.mean()), 3),
                     "mark_median": round(float(np.median(mk)), 3),
                     "mark_q": q(mk),
                     "fixed_by_construction": cause in ("EDGE_GONE", "NEVER_ALIVE")}
        print("    %-15s n=%-4d share %.3f  mean %+8.3f  median %+8.3f  %s"
              % (cause, len(v), len(v) / len(rows), mk.mean(), np.median(mk),
                 "FIXED BY CONSTRUCTION" if m2[cause]["fixed_by_construction"] else "FREE"))

    # ---- M3 was the time worth anything
    print("\nM3  MFE INSIDE THE ALIVE SPELL vs THE MARK AT EXIT   (descriptive)")
    m3 = {}
    for cause in ("EDGE_GONE", "INTERRUPTED", "ADMINISTRATIVE"):
        v = by.get(cause, [])
        if not v:
            continue
        mfe = np.array([x["mfe_bps"] for x in v], float)
        mk = np.array([x["mark_bps"] for x in v], float)
        tm = np.array([x["t_min"] for x in v], float)
        give = mfe - mk
        m3[cause] = {"n": len(v), "mfe_median": round(float(np.median(mfe)), 3),
                     "mark_median": round(float(np.median(mk)), 3),
                     "giveback_median": round(float(np.median(give)), 3),
                     "giveback_q": q(give),
                     "t_min_median": round(float(np.median(tm)), 3),
                     "mfe_per_minute_median": round(float(np.median(mfe / np.maximum(tm, 1e-6))), 3)}
        print("    %-15s MFE med %+8.3f   mark med %+8.3f   GIVEBACK med %+8.3f   t med %6.2f min"
              % (cause, np.median(mfe), np.median(mk), np.median(give), np.median(tm)))

    res = {"prereg_sha256": h, "n": len(rows), "answers": "A-S72",
           "M1_structural_self_check": {"pinned": pinned, "n": int(len(eg)),
                                        "median": round(float(np.median(eg)), 3),
                                        "abs_dev_from_k_p90": round(
                                            float(np.percentile(np.abs(eg - K_BPS), 90)), 3)},
           "M2_marks_by_transition": m2,
           "M3_mfe_vs_exit_mark": m3,
           "scope": "DESCRIPTIVE marking of a frozen decomposition; no exit rule, no threshold, "
                    "no horizon selected; spends no alpha"}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
