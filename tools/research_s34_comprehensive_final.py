"""S34 Comprehensive Final Suite.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Builds on silence gate confirmation (hold T3R=+7733, WR=70.1%, p=0).

Tests:
  A. noisy_AND_NOT_bull as SHORT signal — WR 37.3% long = WR 62.7% short?
     Permutation null + holdout + cascade size breakdown.
  B. 30-min delayed entry — silence confirmed then enter: price impact?
     At ts+1800s, is the fade still profitable after the silence window passes?
  C. sync_k proxy gate — entry-time filter without waiting 30min.
     Best threshold for pre-entry filtering.
  D. Exit management — enter all, detect propagation, exit early on noisy.
     Managed exit vs unmanaged 4h.
  E. Silence window length scan — is 30min optimal? Test 5/10/15/20/30/45/60min.
  F. Silence + high sync (sync>=200K) permutation null — WR 83.1% real or small-N?
  G. Cascade size breakdown — does silence gate differ by threshold (50K/100K/200K)?
  H. H1 vs H2 vs H4 exit on silence events — what's the best hold horizon?
  I. Early silence signal (5min) → predict full 30min silence: live entry speed-up.

SAF-02 / DAT-01 / DAT-03: research only, no lookahead in gate computation,
holdout events use post-entry observation window (flagged), seed=42.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import load_jsonl, r1, r3, NAV_EVENTS, FEE_BPS

DEFAULT_DB       = ROOT / "data" / "microstructure.db"
OUT_JSON         = ROOT / "reports" / "research" / "s34" / "S34_COMPREHENSIVE_FINAL.json"
OUT_MD           = ROOT / "reports" / "research" / "s34" / "S34_COMPREHENSIVE_FINAL.md"

HOLDOUT_FRAC     = 0.30
SEED             = 42
N_PERM           = 1000
MIN_N            = 15
SYNC_WINDOW_MS   = 10 * 60 * 1000
SILENCE_WINDOWS  = [5, 10, 15, 20, 30, 45, 60]   # minutes
PROP_THRESH      = 50_000.0
LIVE_THRESH      = 200_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now(): return datetime.now(timezone.utc).isoformat()
def ts_utc(ts): return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def t3r(vals):
    if len(vals) <= 3: return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])

def qs(vals):
    if not vals: return {"n":0,"t3r":None,"sum":None,"med":None,"win":None,"maxL":None}
    return {"n":len(vals),"t3r":r1(t3r(vals)) if len(vals)>=MIN_N else None,
            "sum":r1(sum(vals)),"med":r1(median(vals)),
            "win":r3(sum(1 for v in vals if v>0)/len(vals)),"maxL":r1(min(vals))}

def pctile(vals, p):
    v = sorted(x for x in vals if math.isfinite(x))
    if not v: return float("nan")
    i = p*(len(v)-1); lo = int(i)
    return v[lo]+(i-lo)*(v[min(lo+1,len(v)-1)]-v[lo])

def permtest(target_vals, all_vals, n_perm, seed):
    rng = random.Random(seed)
    real = t3r(target_vals) if len(target_vals) >= MIN_N else float("nan")
    n = len(target_vals)
    null = [t3r(rng.sample(all_vals, min(n, len(all_vals)))) for _ in range(n_perm)]
    p95 = pctile(null, 0.95)
    p_right = sum(1 for v in null if math.isfinite(v) and v >= real) / len(null)
    return {"real_t3r": r1(real), "null_p95": r1(p95),
            "p_right": r3(p_right), "verdict": "PASS" if p_right < 0.05 else "ARTIFACT"}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_liq(conn, symbol, side):
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side)).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]

def win_sum(ts_list, vals, lo, hi):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return sum(vals[i] for i in range(a, b))

def win_count_thresh(ts_list, vals, lo, hi, thr):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return sum(1 for i in range(a, b) if vals[i] >= thr)

def mark_at(ts_list, prices, ts):
    idx = bisect.bisect_left(ts_list, ts)
    return prices[idx] if idx < len(ts_list) else None


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------

def annotate(rows, eth_sell_ts, eth_sell_not,
             btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not,
             mark_ts, mark_prices):
    out = []
    for r in rows:
        ts = int(r["signal_ts_ms"])
        # silence windows
        silence_by_win = {}
        for w in SILENCE_WINDOWS:
            cnt = win_count_thresh(eth_sell_ts, eth_sell_not,
                                   ts + 60_000, ts + w*60_000, PROP_THRESH)
            silence_by_win[w] = cnt == 0
        # primary 30-min silence
        silence30 = silence_by_win[30]
        # 5-min early silence signal
        silence5  = silence_by_win[5]
        # sync_k
        b = win_sum(btc_sell_ts, btc_sell_not, ts-SYNC_WINDOW_MS, ts)
        s = win_sum(sol_sell_ts, sol_sell_not, ts-SYNC_WINDOW_MS, ts)
        sync_k = b + s
        # mark price at entry, +30min, +1h, +2h, +4h
        p0  = mark_at(mark_ts, mark_prices, ts)
        p30 = mark_at(mark_ts, mark_prices, ts + 30*60*1000)
        p1h = mark_at(mark_ts, mark_prices, ts + 1*3600*1000)
        p2h = mark_at(mark_ts, mark_prices, ts + 2*3600*1000)
        p4h = mark_at(mark_ts, mark_prices, ts + 4*3600*1000)
        def ret_bps(p_exit, p_entry):
            if p_exit is None or p_entry is None or p_entry <= 0: return None
            return (p_exit - p_entry) / p_entry * 10000.0
        # returns for LONG from anchor
        net_30m = (ret_bps(p30, p0) - FEE_BPS) if p0 else None
        net_1h  = (ret_bps(p1h, p0) - FEE_BPS) if p0 else None
        net_2h  = float(r.get("net_2h_bps") or float("nan"))
        net_4h  = float(r.get("net_4h_bps") or float("nan")) if r.get("net_4h_bps") is not None else None
        # return for LONG from ts+30min (delayed entry)
        net_delayed_1h5 = (ret_bps(p1h, p30) - FEE_BPS) if (p30 and p1h) else None  # 30min delayed, hold 30min more
        net_delayed_2h  = (ret_bps(p2h, p30) - FEE_BPS) if (p30 and p2h) else None  # delayed, hold 90min
        net_delayed_4h  = (ret_bps(p4h, p30) - FEE_BPS) if (p30 and p4h) else None  # delayed, hold 3.5h
        # price drift in silence window (how much did price move before delayed entry?)
        drift_30m = ret_bps(p30, p0) if p0 else None
        item = dict(r)
        item.update({
            "silence30": silence30, "silence5": silence5,
            "silence_by_win": silence_by_win,
            "sync_k": sync_k,
            "is_bull": "BULL_PULLBACK" in (r.get("tags") or []),
            "is_live": float(r.get("threshold_usd") or 0) >= LIVE_THRESH,
            "prior4h": float(r.get("prior4h_bps") or 0),
            "threshold": float(r.get("threshold_usd") or 0),
            "net_30m": net_30m, "net_1h": net_1h, "net_2h": net_2h,
            "net_4h": net_4h,
            "net_delayed_1h5": net_delayed_1h5,
            "net_delayed_2h": net_delayed_2h,
            "net_delayed_4h": net_delayed_4h,
            "drift_30m": drift_30m,
        })
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# A — noisy_AND_NOT_bull SHORT signal
# ---------------------------------------------------------------------------

def test_a_noisy_short(cal, hold):
    def is_target(r): return not r["silence30"] and not r["is_bull"]
    def short_val(r):
        v = r.get("net_2h_bps")
        return (-float(v) - 2*FEE_BPS) if v is not None and math.isfinite(float(v)) else None

    cal_tgt  = [r for r in cal  if is_target(r)]
    hold_tgt = [r for r in hold if is_target(r)]
    cal_short_v  = [v for r in cal_tgt  if (v:=short_val(r)) is not None]
    hold_short_v = [v for r in hold_tgt if (v:=short_val(r)) is not None]

    # by horizon
    horizons = {"H1": "net_1h", "H2": "net_2h", "H4": "net_4h"}
    by_hor = {}
    for label, key in horizons.items():
        cal_v  = [(-float(r[key]) - 2*FEE_BPS) for r in cal_tgt
                  if r.get(key) is not None and math.isfinite(float(r[key]))]
        hold_v = [(-float(r[key]) - 2*FEE_BPS) for r in hold_tgt
                  if r.get(key) is not None and math.isfinite(float(r[key]))]
        by_hor[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # by threshold size
    by_thr = {}
    for thr in [50_000, 100_000, 200_000]:
        label = f"thr_{int(thr/1000)}K"
        fn = lambda r, t=thr: is_target(r) and r["threshold"] >= t and r["threshold"] < t*3
        cal_v  = [v for r in cal  if fn(r) and (v:=short_val(r)) is not None]
        hold_v = [v for r in hold if fn(r) and (v:=short_val(r)) is not None]
        by_thr[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # permutation null on cal
    all_cal_net = [float(r["net_2h_bps"]) for r in cal if r.get("net_2h_bps") is not None]
    # for SHORT: -net - 2*fee; we sample from the actual distribution
    all_cal_short = [-v - 2*FEE_BPS for v in all_cal_net]
    perm = permtest(cal_short_v, all_cal_short, N_PERM, SEED)

    # sync_k conditioning on noisy-short
    by_sync = {}
    for s_thr in [100_000, 200_000, 300_000, 500_000]:
        lbl = f"sync_lt_{int(s_thr/1000)}K"
        fn2 = lambda r, t=s_thr: is_target(r) and r["sync_k"] < t
        cal_v  = [v for r in cal  if fn2(r) and (v:=short_val(r)) is not None]
        hold_v = [v for r in hold if fn2(r) and (v:=short_val(r)) is not None]
        by_sync[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return {
        "cal_short":  qs(cal_short_v),
        "hold_short": qs(hold_short_v),
        "by_horizon": by_hor,
        "by_threshold": by_thr,
        "permutation": perm,
        "by_sync_k": by_sync,
    }


# ---------------------------------------------------------------------------
# B — 30-min delayed entry
# ---------------------------------------------------------------------------

def test_b_delayed_entry(cal, hold):
    """For SILENT events: compare anchor entry vs 30-min delayed entry."""
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        sil = [r for r in rows if r["silence30"]]
        # anchor entry returns
        anc_2h = [r["net_2h"] for r in sil if r.get("net_2h") is not None and math.isfinite(r["net_2h"])]
        anc_4h = [r["net_4h"] for r in sil if r.get("net_4h") is not None and math.isfinite(r["net_4h"])]
        # delayed entry (enter at +30min)
        del_2h  = [r["net_delayed_2h"]  for r in sil if r.get("net_delayed_2h")  is not None]
        del_4h  = [r["net_delayed_4h"]  for r in sil if r.get("net_delayed_4h")  is not None]
        # drift: how much did price move in the silence window?
        drifts  = [r["drift_30m"] for r in sil if r.get("drift_30m") is not None]
        drift_up   = sum(1 for d in drifts if d > 0)
        drift_down = sum(1 for d in drifts if d < 0)
        result[split_lbl] = {
            "n_silence": len(sil),
            "anchor_2h":  qs(anc_2h),
            "anchor_4h":  qs(anc_4h),
            "delayed_90min": qs(del_2h),   # enter +30min, exit +2h from anchor = 90min hold
            "delayed_3h5": qs(del_4h),     # enter +30min, exit +4h from anchor = 3.5h hold
            "drift_in_silence": {
                "n": len(drifts),
                "median_bps": r1(median(drifts)) if drifts else None,
                "pct_up": r3(drift_up/len(drifts)) if drifts else None,
                "pct_down": r3(drift_down/len(drifts)) if drifts else None,
                "mean": r1(sum(drifts)/len(drifts)) if drifts else None,
            },
        }

    # Also: live rule (200K) only with delayed entry
    result["live_200K_delayed"] = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        sil_live = [r for r in rows if r["silence30"] and r["is_live"]]
        del_4h = [r["net_delayed_4h"] for r in sil_live if r.get("net_delayed_4h") is not None]
        result["live_200K_delayed"][split_lbl] = {
            "n": len(sil_live),
            "delayed_3h5": qs(del_4h),
        }
    return result


# ---------------------------------------------------------------------------
# C — sync_k proxy gate (entry-time, no waiting)
# ---------------------------------------------------------------------------

def test_c_sync_proxy(cal, hold):
    """Entry-time filtering: skip if sync_k >= threshold."""
    thresholds = [100_000, 200_000, 300_000, 500_000, 1_000_000]
    result = {}
    # All events (no silence confirmation)
    for thr in thresholds:
        lbl = f"skip_sync_gte_{int(thr/1000)}K"
        fn = lambda r, t=thr: r["sync_k"] < t
        cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r.get("net_2h") or float("nan"))]
        hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r.get("net_2h") or float("nan"))]
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # Live rule (200K) + sync proxy
    result["live_200K_sync_proxy"] = {}
    for thr in [200_000, 300_000, 500_000]:
        lbl = f"sync_lt_{int(thr/1000)}K"
        cal_v  = [r["net_2h"] for r in cal  if r["is_live"] and r["sync_k"] < thr
                  and math.isfinite(r.get("net_2h") or float("nan"))]
        hold_v = [r["net_2h"] for r in hold if r["is_live"] and r["sync_k"] < thr
                  and math.isfinite(r.get("net_2h") or float("nan"))]
        result["live_200K_sync_proxy"][lbl] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return result


# ---------------------------------------------------------------------------
# D — exit management (enter all, exit early on noisy)
# ---------------------------------------------------------------------------

def test_d_exit_mgmt(cal, hold):
    """Simulate: enter at anchor. If noisy (propagation in 30min), exit at 30min.
    If silent, hold to 4h. Compare managed vs raw."""
    def managed_val(r):
        if r["silence30"]:
            v4 = r.get("net_4h")
            return float(v4) if v4 is not None and math.isfinite(float(v4)) else None
        else:
            v30 = r.get("net_30m")
            return float(v30) if v30 is not None and math.isfinite(float(v30)) else None

    def raw_val(r):
        v = r.get("net_4h")
        return float(v) if v is not None and math.isfinite(float(v)) else None

    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        mgd = [v for r in rows if (v:=managed_val(r)) is not None]
        raw = [v for r in rows if (v:=raw_val(r)) is not None]
        n_sil   = sum(1 for r in rows if r["silence30"])
        n_noisy = sum(1 for r in rows if not r["silence30"])

        # live rule only
        live_rows = [r for r in rows if r["is_live"]]
        mgd_live = [v for r in live_rows if (v:=managed_val(r)) is not None]
        raw_live = [v for r in live_rows if (v:=raw_val(r)) is not None]

        result[split_lbl] = {
            "n_total": len(rows), "n_silence": n_sil, "n_noisy": n_noisy,
            "managed_all": qs(mgd),
            "raw_4h_all":  qs(raw),
            "managed_live200K": qs(mgd_live),
            "raw_4h_live200K":  qs(raw_live),
        }
    return result


# ---------------------------------------------------------------------------
# E — silence window length scan
# ---------------------------------------------------------------------------

def test_e_window_scan(cal, hold):
    result = {}
    for w in SILENCE_WINDOWS:
        lbl = f"silence_{w}min"
        cal_v  = [r["net_2h"] for r in cal  if r["silence_by_win"].get(w)
                  and math.isfinite(r.get("net_2h") or float("nan"))]
        hold_v = [r["net_2h"] for r in hold if r["silence_by_win"].get(w)
                  and math.isfinite(r.get("net_2h") or float("nan"))]
        n_cal_sil  = sum(1 for r in cal  if r["silence_by_win"].get(w))
        n_hold_sil = sum(1 for r in hold if r["silence_by_win"].get(w))
        result[lbl] = {
            "window_min": w,
            "cal_silence_n": n_cal_sil, "cal_silence_rate": r3(n_cal_sil/len(cal)),
            "hold_silence_n": n_hold_sil, "hold_silence_rate": r3(n_hold_sil/len(hold)),
            "cal": qs(cal_v), "hold": qs(hold_v),
        }
    return result


# ---------------------------------------------------------------------------
# F — silence + high sync permutation null
# ---------------------------------------------------------------------------

def test_f_high_sync_perm(cal, hold):
    def is_target(r): return r["silence30"] and r["sync_k"] >= 200_000

    cal_v  = [r["net_2h"] for r in cal  if is_target(r)
              and math.isfinite(r.get("net_2h") or float("nan"))]
    hold_v = [r["net_2h"] for r in hold if is_target(r)
              and math.isfinite(r.get("net_2h") or float("nan"))]

    all_cal = [r["net_2h"] for r in cal if math.isfinite(r.get("net_2h") or float("nan"))]
    perm_cal  = permtest(cal_v,  all_cal, N_PERM, SEED)

    all_hold = [r["net_2h"] for r in hold if math.isfinite(r.get("net_2h") or float("nan"))]
    perm_hold = permtest(hold_v, all_hold, N_PERM, SEED+1)

    # also: silence + sync >= 300K
    cal_v2  = [r["net_2h"] for r in cal  if r["silence30"] and r["sync_k"] >= 300_000
               and math.isfinite(r.get("net_2h") or float("nan"))]
    hold_v2 = [r["net_2h"] for r in hold if r["silence30"] and r["sync_k"] >= 300_000
               and math.isfinite(r.get("net_2h") or float("nan"))]

    return {
        "silence_sync_gte_200K": {
            "cal": qs(cal_v), "hold": qs(hold_v),
            "perm_cal": perm_cal, "perm_hold": perm_hold,
        },
        "silence_sync_gte_300K": {
            "cal": qs(cal_v2), "hold": qs(hold_v2),
        },
    }


# ---------------------------------------------------------------------------
# G — cascade size breakdown
# ---------------------------------------------------------------------------

def test_g_size_breakdown(cal, hold):
    result = {}
    for thr_lo, thr_hi, lbl in [
        (50_000,  99_999,  "50K"),
        (100_000, 199_999, "100K"),
        (200_000, 9e9,     "200K+"),
    ]:
        fn = lambda r, lo=thr_lo, hi=thr_hi: lo <= r["threshold"] <= hi
        for gate, gfn in [("silence", lambda r, f=fn: f(r) and r["silence30"]),
                          ("noisy",   lambda r, f=fn: f(r) and not r["silence30"]),
                          ("all",     fn)]:
            key = f"thr_{lbl}_{gate}"
            cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r.get("net_2h") or float("nan"))]
            hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r.get("net_2h") or float("nan"))]
            result[key] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# H — horizon scan on silence events
# ---------------------------------------------------------------------------

def test_h_horizon_scan(cal, hold):
    horizons = {
        "H0_5": ("net_30m",         "30min hold from anchor"),
        "H1":   ("net_1h",          "1h hold"),
        "H2":   ("net_2h",          "2h hold"),
        "H4":   ("net_4h",          "4h hold"),
        "H2_delayed": ("net_delayed_2h",  "enter +30min, hold 90min total"),
        "H4_delayed": ("net_delayed_4h",  "enter +30min, hold 3.5h total"),
    }
    result = {}
    for lbl, (key, desc) in horizons.items():
        cal_v  = [r[key] for r in cal  if r["silence30"] and r.get(key) is not None
                  and math.isfinite(r[key])]
        hold_v = [r[key] for r in hold if r["silence30"] and r.get(key) is not None
                  and math.isfinite(r[key])]
        result[lbl] = {"desc": desc, "cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# I — early silence (5min) as predictor of 30min silence
# ---------------------------------------------------------------------------

def test_i_early_silence(cal, hold):
    """5-min silence predicts 30-min silence? If yes, can enter after 5min."""
    def prob_30_given_5(rows):
        sil5  = [r for r in rows if r["silence5"]]
        sil5_and_30 = [r for r in sil5 if r["silence30"]]
        noisy5 = [r for r in rows if not r["silence5"]]
        noisy5_and_30 = [r for r in noisy5 if r["silence30"]]
        return {
            "n_sil5": len(sil5),
            "sil5_rate": r3(len(sil5)/len(rows)) if rows else None,
            "prob_30_given_5": r3(len(sil5_and_30)/len(sil5)) if sil5 else None,
            "prob_30_given_noisy5": r3(len(noisy5_and_30)/len(noisy5)) if noisy5 else None,
        }

    cal_pred  = prob_30_given_5(cal)
    hold_pred = prob_30_given_5(hold)

    # outcome: enter at +5min if silent at 5min
    # delayed entry at +5min, hold to +2h or +4h from anchor
    def delay5_val(r, key):
        """Approximate: 5-min price not stored, use drift proportionally.
        We compare: silence5 events, outcome at 2h."""
        v = r.get("net_2h")
        return float(v) if v is not None and math.isfinite(float(v)) else None

    result = {
        "cal_prediction": cal_pred,
        "hold_prediction": hold_pred,
        "silence5_outcome_2h": {},
        "interpretation": (
            "If prob_30_given_5 >> base silence rate -> 5-min check is a good proxy. "
            "High predictability -> can enter after 5min instead of 30min."
        ),
    }
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        sil5_sil30_v = [v for r in rows if r["silence5"] and r["silence30"]
                        and (v:=delay5_val(r, "net_2h")) is not None]
        sil5_noisy30_v = [v for r in rows if r["silence5"] and not r["silence30"]
                          and (v:=delay5_val(r, "net_2h")) is not None]
        noisy5_v = [v for r in rows if not r["silence5"]
                    and (v:=delay5_val(r, "net_2h")) is not None]
        result["silence5_outcome_2h"][split_lbl] = {
            "sil5_AND_sil30": qs(sil5_sil30_v),
            "sil5_AND_noisy30": qs(sil5_noisy30_v),
            "noisy5": qs(noisy5_v),
        }
    return result


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_md(res):
    sp = res["split"]
    lines = [
        "# S34 Comprehensive Final Suite",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        f"Cal: {sp['cal_n']} ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} ({sp['hold_start']} to {sp['hold_end']})",
        "",
    ]

    # A
    ta = res["test_a"]
    lines += [
        "## A. noisy_AND_NOT_bull SHORT Signal",
        "",
        f"Cal SHORT: {ta['cal_short']['n']} events, T3R={ta['cal_short'].get('t3r')}, "
        f"med={ta['cal_short']['med']}, win={ta['cal_short']['win']}, maxL={ta['cal_short']['maxL']}",
        f"Hold SHORT: {ta['hold_short']['n']} events, T3R={ta['hold_short'].get('t3r')}, "
        f"med={ta['hold_short']['med']}, win={ta['hold_short']['win']}, maxL={ta['hold_short']['maxL']}",
        f"Permutation null (cal): real T3R={ta['permutation']['real_t3r']}, "
        f"null p95={ta['permutation']['null_p95']}, p-right={ta['permutation']['p_right']} -> "
        f"**{ta['permutation']['verdict']}**",
        "",
        "### By horizon",
        "| Horizon | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lbl, d in ta["by_horizon"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |")

    lines += ["", "### By cascade threshold", "| Threshold | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in ta["by_threshold"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['win']} | {h['n']} | {h.get('t3r')} | {h['win']} |")

    lines += ["", "### SHORT + sync_k conditioning", "| Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in ta["by_sync_k"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['win']} | {h['n']} | {h.get('t3r')} | {h['win']} |")
    lines.append("")

    # B
    tb = res["test_b"]
    lines += ["## B. 30-min Delayed Entry (Enter After Silence Confirmed)", ""]
    for split_lbl in ("cal", "hold"):
        d = tb[split_lbl]
        drift = d["drift_in_silence"]
        lines += [
            f"### {split_lbl} (N_silence={d['n_silence']})",
            f"- Price drift in silence window: median={drift['median_bps']}bps, "
            f"mean={drift['mean']}bps, pct_up={drift['pct_up']}, pct_down={drift['pct_down']}",
            f"- Anchor 2h:      {d['anchor_2h']['n']} | T3R={d['anchor_2h'].get('t3r')} | med={d['anchor_2h']['med']} | win={d['anchor_2h']['win']}",
            f"- Anchor 4h:      {d['anchor_4h']['n']} | T3R={d['anchor_4h'].get('t3r')} | med={d['anchor_4h']['med']} | win={d['anchor_4h']['win']}",
            f"- Delayed 90min:  {d['delayed_90min']['n']} | T3R={d['delayed_90min'].get('t3r')} | med={d['delayed_90min']['med']} | win={d['delayed_90min']['win']}",
            f"- Delayed 3.5h:   {d['delayed_3h5']['n']} | T3R={d['delayed_3h5'].get('t3r')} | med={d['delayed_3h5']['med']} | win={d['delayed_3h5']['win']}",
            "",
        ]
    live_del = tb["live_200K_delayed"]
    lines += ["### Live rule (200K) with 30-min delayed entry",
              "| Split | N | Delayed 3.5h T3R | Delayed 3.5h med | Delayed 3.5h win |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for sl, d in live_del.items():
        dv = d["delayed_3h5"]
        lines.append(f"| {sl} | {d['n']} | {dv.get('t3r')} | {dv['med']} | {dv['win']} |")
    lines.append("")

    # C
    tc = res["test_c"]
    lines += ["## C. sync_k Proxy Gate (Entry-Time Filter, No Waiting)", ""]
    lines += ["| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in tc.items():
        if lbl == "live_200K_sync_proxy": continue
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |")

    lines += ["", "### Live rule (200K) + sync_k proxy",
              "| Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in tc["live_200K_sync_proxy"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['win']} | {h['n']} | {h.get('t3r')} | {h['win']} |")
    lines.append("")

    # D
    td = res["test_d"]
    lines += ["## D. Exit Management (Enter All, Exit Early on Noisy)", ""]
    lines += ["| Split | Managed T3R | Managed med | Managed win | Raw 4h T3R | Raw 4h med | Raw 4h win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for sl, d in td.items():
        m = d["managed_all"]; raw = d["raw_4h_all"]
        lines.append(f"| {sl} all | {m.get('t3r')} | {m['med']} | {m['win']} | {raw.get('t3r')} | {raw['med']} | {raw['win']} |")
        ml = d["managed_live200K"]; rl = d["raw_4h_live200K"]
        lines.append(f"| {sl} live200K | {ml.get('t3r')} | {ml['med']} | {ml['win']} | {rl.get('t3r')} | {rl['med']} | {rl['win']} |")
    lines.append("")

    # E
    te = res["test_e"]
    lines += ["## E. Silence Window Length Scan", ""]
    lines += ["| Window | Cal Silence N | Cal Silence rate | Cal T3R | Cal med | Cal win | Hold Silence N | Hold rate | Hold T3R | Hold med | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in te.items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {d['window_min']}min | {d['cal_silence_n']} | {d['cal_silence_rate']} |"
                     f" {c.get('t3r')} | {c['med']} | {c['win']} |"
                     f" {d['hold_silence_n']} | {d['hold_silence_rate']} | {h.get('t3r')} | {h['med']} | {h['win']} |")
    lines.append("")

    # F
    tf = res["test_f"]
    lines += ["## F. Silence + High Sync Permutation Null", ""]
    for key, d in tf.items():
        c = d["cal"]; h = d["hold"]
        lines += [f"### {key}",
                  f"Cal: N={c['n']} T3R={c.get('t3r')} med={c['med']} win={c['win']}",
                  f"Hold: N={h['n']} T3R={h.get('t3r')} med={h['med']} win={h['win']}"]
        if "perm_cal" in d:
            p = d["perm_cal"]; ph = d["perm_hold"]
            lines += [f"Perm (cal):  real={p['real_t3r']} null_p95={p['null_p95']} p-right={p['p_right']} -> **{p['verdict']}**",
                      f"Perm (hold): real={ph['real_t3r']} null_p95={ph['null_p95']} p-right={ph['p_right']} -> **{ph['verdict']}**"]
        lines.append("")

    # G
    tg = res["test_g"]
    lines += ["## G. Cascade Size Breakdown", ""]
    lines += ["| Key | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in tg.items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |")
    lines.append("")

    # H
    th = res["test_h"]
    lines += ["## H. Horizon Scan on Silence Events", ""]
    lines += ["| Horizon | Desc | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in th.items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {d['desc']} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |")
    lines.append("")

    # I
    ti = res["test_i"]
    lines += ["## I. Early Silence Signal (5-min Predictor)", ""]
    for sp_lbl in ("cal_prediction", "hold_prediction"):
        d = ti[sp_lbl]
        lines.append(f"**{sp_lbl}**: N_sil5={d['n_sil5']} ({d['sil5_rate']*100:.1f}%) | "
                     f"P(30min_sil | 5min_sil)={d['prob_30_given_5']} | "
                     f"P(30min_sil | 5min_noisy)={d['prob_30_given_noisy5']}")
    lines += ["", "### 5-min silence -> 2h outcome",
              "| Split | sil5+sil30 N | T3R | med | win | sil5+noisy30 N | T3R | med | win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for sp_lbl, d in ti["silence5_outcome_2h"].items():
        s30 = d["sil5_AND_sil30"]; n30 = d["sil5_AND_noisy30"]
        lines.append(f"| {sp_lbl} | {s30['n']} | {s30.get('t3r')} | {s30['med']} | {s30['win']} |"
                     f" {n30['n']} | {n30.get('t3r')} | {n30['med']} | {n30['win']} |")
    lines += ["", f"*Interpretation*: {ti['interpretation']}", ""]

    lines += ["## Key Verdicts", "",
              "- A: noisy_NOT_bull SHORT — is it the other side of the silence trade?",
              "- B: delayed entry — does the edge survive if we wait 30min?",
              "- C: sync_k proxy — best entry-time filter without waiting?",
              "- D: exit management — does early exit on noisy events improve overall?",
              "- E: optimal window — what's the best silence confirmation time?",
              "- F: high-sync silence — is WR 83% statistically real?",
              "- G: size breakdown — which cascade size drives the silence alpha?",
              "- H: best hold horizon for silence trades?",
              "- I: 5-min early signal — can we enter after 5min instead of 30min?",
              "", "RESEARCH_ONLY. All findings require permutation-null before live promotion."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading events...")
    all_rows = load_jsonl(NAV_EVENTS)
    all_rows = [r for r in all_rows if r.get("net_2h_bps") is not None]
    all_rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    n_cal = int(len(all_rows) * (1.0 - HOLDOUT_FRAC))
    cal_raw, hold_raw = all_rows[:n_cal], all_rows[n_cal:]
    print(f"Total={len(all_rows)}  Cal={len(cal_raw)}  Hold={len(hold_raw)}")

    print("Loading DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_sell_ts, eth_sell_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_sell_ts, btc_sell_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_sell_ts, sol_sell_not = load_liq(conn, "SOLUSDT", "SELL")
        eth_mark_rows = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' ORDER BY ts_ms"
        ).fetchall()
    mark_ts     = [int(r[0]) for r in eth_mark_rows]
    mark_prices = [float(r[1]) for r in eth_mark_rows]
    print(f"Loaded {len(mark_ts)} mark-price rows")

    print("Annotating cal rows...")
    cal = annotate(cal_raw, eth_sell_ts, eth_sell_not,
                   btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not,
                   mark_ts, mark_prices)
    print("Annotating hold rows...")
    hold = annotate(hold_raw, eth_sell_ts, eth_sell_not,
                    btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not,
                    mark_ts, mark_prices)

    print("A: noisy SHORT signal...")
    ta = test_a_noisy_short(cal, hold)
    print("B: delayed entry...")
    tb = test_b_delayed_entry(cal, hold)
    print("C: sync_k proxy gate...")
    tc = test_c_sync_proxy(cal, hold)
    print("D: exit management...")
    td = test_d_exit_mgmt(cal, hold)
    print("E: window scan...")
    te = test_e_window_scan(cal, hold)
    print("F: high-sync silence perm null...")
    tf = test_f_high_sync_perm(cal, hold)
    print("G: size breakdown...")
    tg = test_g_size_breakdown(cal, hold)
    print("H: horizon scan...")
    th = test_h_horizon_scan(cal, hold)
    print("I: early silence (5min)...")
    ti = test_i_early_silence(cal, hold)

    split_info = {
        "cal_n": len(cal), "hold_n": len(hold),
        "cal_start": ts_utc(cal_raw[0]["signal_ts_ms"]),
        "cal_end":   ts_utc(cal_raw[-1]["signal_ts_ms"]),
        "hold_start":ts_utc(hold_raw[0]["signal_ts_ms"]),
        "hold_end":  ts_utc(hold_raw[-1]["signal_ts_ms"]),
    }
    result = {
        "generated_at_utc": utc_now(), "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "split": split_info,
        "test_a": ta, "test_b": tb, "test_c": tc, "test_d": td,
        "test_e": te, "test_f": tf, "test_g": tg, "test_h": th, "test_i": ti,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
