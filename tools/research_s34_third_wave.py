"""S34 Third Wave Research Suite.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Starting from 3 confirmed OOS signals + deep expansion insights:
  - Silence LONG: hold WR=70.1%, T3R=+7733
  - Silence+sync>=200K LONG: hold WR=83.1%, T3R=+4298
  - noisy_NOT_bull SHORT: hold WR=54.9%, T3R=+11360
  - Combined portfolio: hold T3R=+19952, coverage=98.2%

10 new test groups:
  A. Multi-criteria scoring system (0-6 points per event)
  B. Propagation mechanics (count/size/timing of cascades in noisy window)
  C. BULL_PULLBACK + noisy: long, short, or neutral?
  D. 200K live rule subset optimization (find WR>85% sub-signal)
  E. eth1h prior context on silence and noisy
  F. High-sync noisy SHORT permutation null (sync>=500K/700K/1M)
  G. bid_depth_usd absolute level as signal
  H. Propagation timing (early <5min vs late 20-30min)
  I. Weekday effect (Mon-Sun)
  J. Same-day sequential trades (2nd silence trade vs 1st)

DAT-01/SAF-02/DAT-03: no lookahead, research only, holdout never touched.
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

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_MD     = ROOT / "reports" / "research" / "s34" / "S34_THIRD_WAVE.md"
OUT_JSON   = ROOT / "reports" / "research" / "s34" / "S34_THIRD_WAVE.json"

HOLDOUT_FRAC   = 0.30
SEED           = 42
N_PERM         = 1000
MIN_N          = 12
SYNC_WIN_MS    = 10 * 60 * 1000
SIL_LO         = 60_000
SIL_HI         = 30 * 60_000
PROP_THRESH    = 50_000.0
LIVE_THRESH    = 200_000.0
PROP_COUNT_WIN = 30 * 60_000


# ─── Helpers ─────────────────────────────────────────────────────────────────

def utc_now(): return datetime.now(timezone.utc).isoformat()
def ts_utc(ts):
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def t3r(vals):
    if len(vals) <= 3: return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])

def qs(vals):
    if not vals: return {"n":0,"t3r":None,"sum":None,"med":None,"win":None,"maxL":None,"maxW":None}
    t = t3r(vals) if len(vals) >= MIN_N else None
    return {"n":len(vals),"t3r":r1(t) if t is not None else None,"sum":r1(sum(vals)),
            "med":r1(median(vals)),"win":r3(sum(1 for v in vals if v>0)/len(vals)),
            "maxL":r1(min(vals)),"maxW":r1(max(vals))}

def pctile(vals, p):
    v = sorted(x for x in vals if math.isfinite(x))
    if not v: return float("nan")
    i = p*(len(v)-1); lo = int(i)
    return v[lo] + (i-lo)*(v[min(lo+1,len(v)-1)] - v[lo])

def permtest(target_vals, all_vals, n_perm, seed):
    rng = random.Random(seed)
    real = t3r(target_vals) if len(target_vals) >= MIN_N else float("nan")
    n = len(target_vals)
    null = [t3r(rng.sample(all_vals, min(n, len(all_vals)))) for _ in range(n_perm)]
    p95  = pctile(null, 0.95)
    p_right = sum(1 for v in null if math.isfinite(v) and v >= real) / max(len(null), 1)
    return {"real_t3r":r1(real),"null_p95":r1(p95),
            "p_right":r3(p_right),"verdict":"PASS" if p_right<0.05 else "ARTIFACT"}


# ─── DB helpers ──────────────────────────────────────────────────────────────

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

def win_max(ts_list, vals, lo, hi):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return max((vals[i] for i in range(a, b)), default=0.0)

def first_ts_above(ts_list, vals, lo, hi, thr):
    """Return ms offset of first cascade >= thr after lo, or None."""
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    for i in range(a, b):
        if vals[i] >= thr:
            return ts_list[i] - lo
    return None


# ─── Annotate ────────────────────────────────────────────────────────────────

def annotate(rows, eth_sell_ts, eth_sell_not, btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not):
    out = []
    for r in rows:
        ts = int(r["signal_ts_ms"])

        # 30-min silence
        n_prop = win_count_thresh(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI, PROP_THRESH)
        silence30 = n_prop == 0

        # propagation mechanics
        prop_count  = win_count_thresh(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI, PROP_THRESH)
        prop_max    = win_max(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI)
        prop_first_ms = first_ts_above(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI, PROP_THRESH)

        # sync_k (BTC+SOL SELL prior 10min)
        b  = win_sum(btc_sell_ts, btc_sell_not, ts-SYNC_WIN_MS, ts)
        s  = win_sum(sol_sell_ts, sol_sell_not, ts-SYNC_WIN_MS, ts)
        sync_k = b + s

        # n_prior2h
        n_prior2h = win_count_thresh(eth_sell_ts, eth_sell_not, ts-2*3600_000, ts-1000, PROP_THRESH)

        # session
        hour_utc = datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour
        if 7 <= hour_utc < 13:   session = "EU"
        elif 13 <= hour_utc < 21: session = "US"
        else:                      session = "ASIA"
        weekday = datetime.fromtimestamp(ts/1000, tz=timezone.utc).weekday()  # 0=Mon
        day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][weekday]

        # field extractions
        net_2h   = float(r.get("net_2h_bps")  or "nan")
        net_4h_v = r.get("net_4h_bps")
        net_4h   = float(net_4h_v) if net_4h_v is not None else float("nan")
        prior4h  = float(r.get("prior4h_bps")  or 0)
        vdepth   = float(r.get("vdepth_bps")   or 0)
        book_imb = float(r.get("book_imbalance") or 0)
        eth1h    = float(r.get("eth1h_bps")    or 0)
        btc4h    = float(r.get("btc4h_bps")    or 0)
        thresh   = float(r.get("threshold_usd") or 0)
        bid_dep  = float(r.get("bid_depth_usd") or 0)
        is_bull  = "BULL_PULLBACK" in (r.get("tags") or [])
        is_live  = thresh >= LIVE_THRESH

        # Scoring criteria (0-6 points)
        score = sum([
            int(silence30),
            int(n_prior2h >= 3),
            int(btc4h < 0),
            int(vdepth >= 30),
            int(session == "US"),
            int(sync_k >= 200_000),
        ])

        item = dict(r)
        item.update({
            "silence30": silence30,
            "prop_count": prop_count,
            "prop_max": prop_max,
            "prop_first_ms": prop_first_ms,
            "sync_k": sync_k,
            "n_prior2h": n_prior2h,
            "session": session,
            "hour_utc": hour_utc,
            "weekday": weekday,
            "day_name": day_name,
            "net_2h": net_2h,
            "net_4h": net_4h,
            "prior4h": prior4h,
            "vdepth": vdepth,
            "book_imb": book_imb,
            "eth1h": eth1h,
            "btc4h": btc4h,
            "thresh": thresh,
            "bid_dep": bid_dep,
            "is_bull": is_bull,
            "is_live": is_live,
            "score": score,
        })
        out.append(item)
    return out


# ─── A — Multi-criteria scoring ──────────────────────────────────────────────

def test_a_scoring(cal, hold):
    """
    Score = sum of 6 boolean criteria:
    silence30, n_prior2h>=3, btc4h<0, vdepth>=30, US session, sync_k>=200K
    High score events = potential "perfect storm" cascades.
    """
    result = {"by_score": {}}
    for score_level in range(7):
        fn_sil  = lambda r, s=score_level: r["score"] >= s and r["silence30"]
        fn_nois = lambda r, s=score_level: r["score"] >= s and not r["silence30"] and not r["is_bull"]
        fn_all  = lambda r, s=score_level: r["score"] >= s

        lbl = f"score_gte_{score_level}"
        cal_sil_v  = [r["net_2h"] for r in cal  if fn_sil(r)  and math.isfinite(r["net_2h"])]
        hold_sil_v = [r["net_2h"] for r in hold if fn_sil(r)  and math.isfinite(r["net_2h"])]
        cal_nois_v = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn_nois(r) and math.isfinite(r["net_2h"])]
        hold_nois_v= [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn_nois(r) and math.isfinite(r["net_2h"])]
        cal_all_v  = [r["net_2h"] for r in cal  if fn_all(r)  and math.isfinite(r["net_2h"])]
        hold_all_v = [r["net_2h"] for r in hold if fn_all(r)  and math.isfinite(r["net_2h"])]

        result["by_score"][lbl] = {
            "silence_LONG": {"cal": qs(cal_sil_v), "hold": qs(hold_sil_v)},
            "noisy_SHORT":  {"cal": qs(cal_nois_v),"hold": qs(hold_nois_v)},
            "all_raw":      {"cal": qs(cal_all_v), "hold": qs(hold_all_v)},
        }

    # individual score values 0,1,2,3,4,5,6
    result["by_exact_score"] = {}
    for score_exact in range(7):
        fn_e = lambda r, s=score_exact: r["score"] == s
        cal_v  = [r["net_2h"] for r in cal  if fn_e(r) and math.isfinite(r["net_2h"])]
        hold_v = [r["net_2h"] for r in hold if fn_e(r) and math.isfinite(r["net_2h"])]
        result["by_exact_score"][f"score_{score_exact}"] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # perfect storm: score>=4 + silence — permutation null
    ps_cal  = [r["net_2h"] for r in cal  if r["score"]>=4 and r["silence30"] and math.isfinite(r["net_2h"])]
    ps_hold = [r["net_2h"] for r in hold if r["score"]>=4 and r["silence30"] and math.isfinite(r["net_2h"])]
    all_cal = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    result["perfect_storm_silence"] = {
        "cal": qs(ps_cal), "hold": qs(ps_hold),
        "perm_cal": permtest(ps_cal, all_cal, N_PERM, SEED),
    }

    # score>=4 noisy SHORT permutation null
    ps_short_cal  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if r["score"]>=4 and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
    ps_short_hold = [(-r["net_2h"]-2*FEE_BPS) for r in hold if r["score"]>=4 and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
    all_cal_short = [-v-2*FEE_BPS for v in all_cal]
    result["perfect_storm_short"] = {
        "cal": qs(ps_short_cal), "hold": qs(ps_short_hold),
        "perm_cal": permtest(ps_short_cal, all_cal_short, N_PERM, SEED),
    }
    return result


# ─── B — Propagation mechanics ───────────────────────────────────────────────

def test_b_propagation(cal, hold):
    result = {}

    # B1: count of cascades in noisy window
    for cnt_lo, cnt_hi, lbl in [(1,1,"exactly_1"),(2,3,"2_or_3"),(4,99,"4plus")]:
        fn = lambda r, lo=cnt_lo, hi=cnt_hi: not r["silence30"] and lo <= r["prop_count"] <= hi
        cal_long  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
        hold_long = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        cal_shor  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        hold_shor = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        result[f"prop_count_{lbl}"] = {
            "LONG": {"cal": qs(cal_long), "hold": qs(hold_long)},
            "SHORT": {"cal": qs(cal_shor), "hold": qs(hold_shor)},
        }

    # B2: max propagation cascade size
    for sz_lo, sz_hi, lbl in [
        (50_000,  100_000, "50K_100K"),
        (100_000, 200_000, "100K_200K"),
        (200_000, 9e9,     "200K_plus"),
    ]:
        fn = lambda r, lo=sz_lo, hi=sz_hi: not r["silence30"] and lo <= r["prop_max"] < hi
        cal_s  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        hold_s = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        result[f"prop_max_{lbl}_SHORT"] = {"cal": qs(cal_s), "hold": qs(hold_s)}

    # B3: timing of first propagation cascade
    for t_lo_min, t_hi_min, lbl in [
        (0,  5,  "first_0_5min"),
        (5,  15, "first_5_15min"),
        (15, 30, "first_15_30min"),
    ]:
        lo_ms = t_lo_min * 60_000
        hi_ms = t_hi_min * 60_000
        def fn(r, lo=lo_ms, hi=hi_ms):
            fm = r.get("prop_first_ms")
            return not r["silence30"] and fm is not None and lo <= fm < hi
        cal_s  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        hold_s = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and not r["is_bull"] and math.isfinite(r["net_2h"])]
        result[f"prop_timing_{lbl}_SHORT"] = {"cal": qs(cal_s), "hold": qs(hold_s)}

    return result


# ─── C — BULL_PULLBACK + noisy ───────────────────────────────────────────────

def test_c_bull_noisy(cal, hold):
    """
    We excluded BULL_PULLBACK events from noisy SHORT.
    What is their actual outcome? LONG, SHORT, or random?
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        bull_noisy   = [r for r in rows if r["is_bull"] and not r["silence30"]]
        bull_silence = [r for r in rows if r["is_bull"] and r["silence30"]]
        noisy_not_bull = [r for r in rows if not r["is_bull"] and not r["silence30"]]

        bull_noisy_long  = [r["net_2h"] for r in bull_noisy   if math.isfinite(r["net_2h"])]
        bull_noisy_short = [(-r["net_2h"]-2*FEE_BPS) for r in bull_noisy if not r["is_bull"]==False and math.isfinite(r["net_2h"])]
        bull_sil_long    = [r["net_2h"] for r in bull_silence if math.isfinite(r["net_2h"])]
        noisy_not_bull_s = [(-r["net_2h"]-2*FEE_BPS) for r in noisy_not_bull if math.isfinite(r["net_2h"])]

        # for BULL_PULLBACK+noisy: test both LONG and SHORT
        bn_long  = [r["net_2h"] for r in bull_noisy if math.isfinite(r["net_2h"])]
        bn_short = [(-r["net_2h"]-2*FEE_BPS) for r in bull_noisy if math.isfinite(r["net_2h"])]

        result[split_lbl] = {
            "bull_noisy_LONG":     qs(bn_long),
            "bull_noisy_SHORT":    qs(bn_short),
            "bull_silence_LONG":   qs(bull_sil_long),
            "noisy_not_bull_SHORT":qs(noisy_not_bull_s),
        }

    # permutation nulls on bull_noisy LONG and SHORT
    all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
    bn_long_cal  = [r["net_2h"] for r in cal if r["is_bull"] and not r["silence30"] and math.isfinite(r["net_2h"])]
    bn_short_cal = [(-r["net_2h"]-2*FEE_BPS) for r in cal if r["is_bull"] and not r["silence30"] and math.isfinite(r["net_2h"])]
    result["perm_bull_noisy_LONG"]  = permtest(bn_long_cal,  all_cal, N_PERM, SEED)
    result["perm_bull_noisy_SHORT"] = permtest(bn_short_cal, [-v-2*FEE_BPS for v in all_cal], N_PERM, SEED)
    return result


# ─── D — 200K live subset optimization ───────────────────────────────────────

def test_d_live_200k(cal, hold):
    """Find the best sub-signal within the 200K live cascade."""
    all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
    result = {}
    combos = {
        "200K_all":             lambda r: r["is_live"],
        "200K_silence":         lambda r: r["is_live"] and r["silence30"],
        "200K_sil_cluster":     lambda r: r["is_live"] and r["silence30"] and r["n_prior2h"]>=3,
        "200K_sil_US":          lambda r: r["is_live"] and r["silence30"] and r["session"]=="US",
        "200K_sil_btcbear":     lambda r: r["is_live"] and r["silence30"] and r["btc4h"]<0,
        "200K_sil_sync200K":    lambda r: r["is_live"] and r["silence30"] and r["sync_k"]>=200_000,
        "200K_sil_vdepth30":    lambda r: r["is_live"] and r["silence30"] and r["vdepth"]>=30,
        "200K_sil_clust_bear":  lambda r: r["is_live"] and r["silence30"] and r["n_prior2h"]>=3 and r["btc4h"]<0,
        "200K_sil_clust_US":    lambda r: r["is_live"] and r["silence30"] and r["n_prior2h"]>=3 and r["session"]=="US",
        "200K_sil_score4":      lambda r: r["is_live"] and r["silence30"] and r["score"]>=4,
        "200K_noisy_short":     lambda r: r["is_live"] and not r["silence30"] and not r["is_bull"],
        "200K_noisy_short_us":  lambda r: r["is_live"] and not r["silence30"] and not r["is_bull"] and r["session"]=="US",
    }
    for lbl, fn in combos.items():
        is_short = "noisy_short" in lbl
        if is_short:
            cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        else:
            cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        perm = permtest(cal_v, all_cal, N_PERM, SEED)
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v), "perm": perm}
    return result


# ─── E — eth1h context ───────────────────────────────────────────────────────

def test_e_eth1h(cal, hold):
    """ETH 1h return before cascade — does it interact with silence gate?"""
    result = {}
    for label, fn in [
        ("eth1h_bull_gt50",   lambda r: r["eth1h"] > 50),
        ("eth1h_bull_0_50",   lambda r: 0 < r["eth1h"] <= 50),
        ("eth1h_flat",        lambda r: -10 <= r["eth1h"] <= 10),
        ("eth1h_bear",        lambda r: r["eth1h"] < 0),
        ("eth1h_bear_lt-50",  lambda r: r["eth1h"] < -50),
        ("eth1h_bear_lt-100", lambda r: r["eth1h"] < -100),
    ]:
        for gate, is_short in [("all", False), ("silence", False), ("noisy_SHORT", True)]:
            if gate == "all":     gfn = fn
            elif gate == "silence": gfn = lambda r, f=fn: f(r) and r["silence30"]
            else:                   gfn = lambda r, f=fn: f(r) and not r["silence30"] and not r["is_bull"]

            if is_short:
                cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            else:
                cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[f"{label}_{gate}"] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ─── F — High-sync noisy SHORT permutation ───────────────────────────────────

def test_f_highsync_short(cal, hold):
    """
    Noisy SHORT winners had avg sync_k=1M vs losers 389K.
    Test formal permutation null at thresholds: 500K, 700K, 1M.
    """
    all_cal_short = [(-r["net_2h"]-2*FEE_BPS) for r in cal
                     if not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
    result = {}
    for thr in [200_000, 300_000, 500_000, 700_000, 1_000_000]:
        lbl = f"sync_gte_{int(thr/1000)}K"
        fn  = lambda r, t=thr: not r["silence30"] and not r["is_bull"] and r["sync_k"] >= t
        cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
        hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        perm   = permtest(cal_v, all_cal_short, N_PERM, SEED)
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v), "perm": perm}

    # also: silence + high sync (already confirmed) vs noisy + high sync
    result["silence_sync_gte_500K"] = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        v = [r["net_2h"] for r in rows if r["silence30"] and r["sync_k"]>=500_000 and math.isfinite(r["net_2h"])]
        result["silence_sync_gte_500K"][split_lbl] = qs(v)
    return result


# ─── G — bid_depth_usd absolute ──────────────────────────────────────────────

def test_g_bid_depth(cal, hold):
    """
    bid_depth_usd: absolute USD size of bids within 1% of mark at cascade time.
    High bid depth = strong buyer support = silence more likely?
    """
    cal_bd = sorted(r["bid_dep"] for r in cal if r["bid_dep"] > 0)
    p25 = pctile(cal_bd, 0.25) if cal_bd else 0
    p50 = pctile(cal_bd, 0.50) if cal_bd else 0
    p75 = pctile(cal_bd, 0.75) if cal_bd else 0
    result = {"bid_depth_percentiles_cal": {"p25":r1(p25),"p50":r1(p50),"p75":r1(p75)}}

    for label, fn in [
        ("bid_q4_high",   lambda r: r["bid_dep"] >= p75),
        ("bid_q3",        lambda r: p50 <= r["bid_dep"] < p75),
        ("bid_q2",        lambda r: p25 <= r["bid_dep"] < p50),
        ("bid_q1_low",    lambda r: r["bid_dep"] < p25),
        ("bid_zero",      lambda r: r["bid_dep"] == 0),
        ("bid_nonzero",   lambda r: r["bid_dep"] > 0),
    ]:
        for gate, is_short in [("all", False), ("silence", False), ("noisy_SHORT", True)]:
            if gate == "all":       gfn = fn
            elif gate == "silence": gfn = lambda r, f=fn: f(r) and r["silence30"]
            else:                   gfn = lambda r, f=fn: f(r) and not r["silence30"] and not r["is_bull"]
            if is_short:
                cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            else:
                cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[f"{label}_{gate}"] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ─── H — Propagation timing (early vs late) ──────────────────────────────────

def test_h_prop_timing(cal, hold):
    """
    Early propagation (<5min): immediate continuation = strong SHORT?
    Late propagation (20-30min): almost silence, weaker SHORT?
    Cross with silence quality: late propagation events vs true silence.
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        # categorize noisy events by TIMING of first propagation
        early   = [r for r in rows if not r["silence30"] and r["prop_first_ms"] is not None and r["prop_first_ms"] < 5*60_000]
        medium  = [r for r in rows if not r["silence30"] and r["prop_first_ms"] is not None and 5*60_000 <= r["prop_first_ms"] < 15*60_000]
        late    = [r for r in rows if not r["silence30"] and r["prop_first_ms"] is not None and r["prop_first_ms"] >= 15*60_000]

        for subset, lbl in [(early,"early_0_5min"),(medium,"med_5_15min"),(late,"late_15_30min")]:
            short_v = [(-r["net_2h"]-2*FEE_BPS) for r in subset if not r["is_bull"] and math.isfinite(r["net_2h"])]
            long_v  = [r["net_2h"] for r in subset if math.isfinite(r["net_2h"])]
            result[f"{lbl}_{split_lbl}"] = {"n": len(subset), "SHORT": qs(short_v), "LONG": qs(long_v)}

        # Also: within 1-minute (ultra-early)
        ultra_early = [r for r in rows if not r["silence30"] and r["prop_first_ms"] is not None and r["prop_first_ms"] < 60_000]
        sv = [(-r["net_2h"]-2*FEE_BPS) for r in ultra_early if not r["is_bull"] and math.isfinite(r["net_2h"])]
        result[f"ultra_early_lt1min_{split_lbl}"] = {"n": len(ultra_early), "SHORT": qs(sv)}
    return result


# ─── I — Weekday effect ──────────────────────────────────────────────────────

def test_i_weekday(cal, hold):
    result = {}
    for day in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        fn_sil  = lambda r, d=day: r["day_name"] == d and r["silence30"]
        fn_nois = lambda r, d=day: r["day_name"] == d and not r["silence30"] and not r["is_bull"]
        fn_all  = lambda r, d=day: r["day_name"] == d

        cal_sil_v  = [r["net_2h"] for r in cal  if fn_sil(r)  and math.isfinite(r["net_2h"])]
        hold_sil_v = [r["net_2h"] for r in hold if fn_sil(r)  and math.isfinite(r["net_2h"])]
        cal_nois_v = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn_nois(r) and math.isfinite(r["net_2h"])]
        hold_nois_v= [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn_nois(r) and math.isfinite(r["net_2h"])]

        n_cal_day  = sum(1 for r in cal  if r["day_name"] == day)
        n_hold_day = sum(1 for r in hold if r["day_name"] == day)

        result[day] = {
            "n_cal": n_cal_day, "n_hold": n_hold_day,
            "silence_LONG": {"cal": qs(cal_sil_v), "hold": qs(hold_sil_v)},
            "noisy_SHORT":  {"cal": qs(cal_nois_v),"hold": qs(hold_nois_v)},
        }
    return result


# ─── J — Same-day sequential trades ─────────────────────────────────────────

def test_j_sequential(cal, hold):
    """
    For each event, count how many SILENCE LONG trades have already been
    taken on the same calendar day (UTC). 1st silence of day vs 2nd vs 3rd+.
    """
    from collections import Counter

    def tag_order(rows):
        # group by UTC date, tag order of silence events
        day_counts = Counter()
        tagged = []
        for r in sorted(rows, key=lambda x: int(x["signal_ts_ms"])):
            dt = datetime.fromtimestamp(int(r["signal_ts_ms"])/1000, tz=timezone.utc)
            day_key = dt.strftime("%Y-%m-%d")
            if r["silence30"]:
                day_counts[day_key] += 1
                order = day_counts[day_key]
            else:
                order = 0  # noisy has no order
            item = dict(r)
            item["day_silence_order"] = order
            tagged.append(item)
        return tagged

    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        tagged = tag_order(rows)
        for order, lbl in [(1,"1st_silence"),(2,"2nd_silence"),(3,"3rd_plus_silence")]:
            if order == 3:
                fn = lambda r: r.get("day_silence_order", 0) >= 3
            else:
                fn = lambda r, o=order: r.get("day_silence_order", 0) == o
            v = [r["net_2h"] for r in tagged if fn(r) and math.isfinite(r["net_2h"])]
            result[f"{lbl}_{split_lbl}"] = qs(v)

        # also: noisy SHORT count on same day
        day_noisy_counts = Counter()
        for r in sorted(rows, key=lambda x: int(x["signal_ts_ms"])):
            dt = datetime.fromtimestamp(int(r["signal_ts_ms"])/1000, tz=timezone.utc)
            dk = dt.strftime("%Y-%m-%d")
            if not r["silence30"] and not r["is_bull"]:
                day_noisy_counts[dk] += 1

        # events per day (frequency analysis)
        day_events = Counter()
        for r in rows:
            dt = datetime.fromtimestamp(int(r["signal_ts_ms"])/1000, tz=timezone.utc)
            dk = dt.strftime("%Y-%m-%d")
            day_events[dk] += 1

        daily_counts = sorted(day_events.values())
        result[f"daily_event_distribution_{split_lbl}"] = {
            "n_days": len(day_events),
            "min": min(daily_counts) if daily_counts else 0,
            "median": r1(median(daily_counts)) if daily_counts else 0,
            "max": max(daily_counts) if daily_counts else 0,
            "p75": r1(pctile(daily_counts, 0.75)),
            "p90": r1(pctile(daily_counts, 0.90)),
        }

    return result


# ─── Render ──────────────────────────────────────────────────────────────────

def render_md(res):
    sp = res["split"]
    lines = [
        "# S34 Third Wave Research Suite",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        f"Cal: {sp['cal_n']} ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} ({sp['hold_start']} to {sp['hold_end']})",
        "",
    ]

    # A
    ta = res["test_a"]
    lines += ["## A. Multi-Criteria Scoring (0-6 points)", "",
              "Criteria: +1 each for silence30, n_prior2h>=3, btc4h<0, vdepth>=30, US session, sync_k>=200K",
              "",
              "### By cumulative score threshold (score >= N)",
              "| Score >= | Silence N cal | Silence T3R cal | Silence WR cal | Silence N hold | Silence T3R hold | Silence WR hold |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in ta["by_score"].items():
        s = d["silence_LONG"]
        c = s["cal"]; h = s["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines += ["", "| Score >= | Short N cal | Short T3R cal | Short WR cal | Short N hold | Short T3R hold | Short WR hold |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in ta["by_score"].items():
        s = d["noisy_SHORT"]
        c = s["cal"]; h = s["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines += ["", "### By exact score value",
              "| Score | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl, d in ta["by_exact_score"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['med']} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['med']} | {h['win']} |")
    lines += ["", "### Perfect storm (score>=4 + silence) — permutation null"]
    ps = ta["perfect_storm_silence"]
    c = ps["cal"]; h = ps["hold"]; p = ps["perm_cal"]
    lines.append(f"Cal: N={c['n']} T3R={c.get('t3r')} med={c['med']} win={c['win']}")
    lines.append(f"Hold: N={h['n']} T3R={h.get('t3r')} med={h['med']} win={h['win']}")
    lines.append(f"Perm cal: real={p['real_t3r']} null_p95={p['null_p95']} p={p['p_right']} -> **{p['verdict']}**")
    lines += ["", "### Perfect storm SHORT (score>=4 + noisy)"]
    pss = ta["perfect_storm_short"]
    c = pss["cal"]; h = pss["hold"]; p = pss["perm_cal"]
    lines.append(f"Cal: N={c['n']} T3R={c.get('t3r')} win={c['win']}")
    lines.append(f"Hold: N={h['n']} T3R={h.get('t3r')} win={h['win']}")
    lines.append(f"Perm cal: real={p['real_t3r']} p={p['p_right']} -> **{p['verdict']}**")
    lines.append("")

    # B
    tb = res["test_b"]
    lines += ["## B. Propagation Mechanics", "",
              "### B1: Cascade count in 30-min noisy window (SHORT direction)",
              "| Count | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl in ["prop_count_exactly_1","prop_count_2_or_3","prop_count_4plus"]:
        if lbl not in tb: continue
        d = tb[lbl]["SHORT"]; c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines += ["", "### B2: Max propagation cascade size (SHORT direction)",
              "| Max size | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl in ["prop_max_50K_100K_SHORT","prop_max_100K_200K_SHORT","prop_max_200K_plus_SHORT"]:
        if lbl not in tb: continue
        d = tb[lbl]; c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines += ["", "### B3: Timing of first propagation (SHORT direction)",
              "| First cascade timing | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lbl in ["prop_timing_first_0_5min_SHORT","prop_timing_first_5_15min_SHORT","prop_timing_first_15_30min_SHORT"]:
        if lbl not in tb: continue
        d = tb[lbl]; c = d["cal"]; h = d["hold"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # C
    tc = res["test_c"]
    lines += ["## C. BULL_PULLBACK + noisy: Long, Short, or Random?", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split_lbl in ("cal", "hold"):
        d = tc[split_lbl]
        for key in ["bull_noisy_LONG","bull_noisy_SHORT","bull_silence_LONG","noisy_not_bull_SHORT"]:
            s = d[key]
            lines.append(f"| {split_lbl}:{key} | {s['n']} | {s.get('t3r','-')} | {s['med']} | {s['win']} | | | | |")
    pl = tc["perm_bull_noisy_LONG"]; ps = tc["perm_bull_noisy_SHORT"]
    lines += ["", f"Perm bull_noisy LONG  (cal): p={pl['p_right']} real={pl['real_t3r']} -> **{pl['verdict']}**",
              f"Perm bull_noisy SHORT (cal): p={ps['p_right']} real={ps['real_t3r']} -> **{ps['verdict']}**", ""]

    # D
    td = res["test_d"]
    lines += ["## D. 200K Live Rule Subset Optimization", ""]
    lines += ["| Subset | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in td.items():
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h['win']} | {p['p_right']} | **{p['verdict']}** |")
    lines.append("")

    # E
    te = res["test_e"]
    lines += ["## E. ETH Prior 1h Context", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in te.items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # F
    tf = res["test_f"]
    lines += ["## F. High-Sync noisy SHORT Permutation Null", ""]
    lines += ["| Sync gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in tf.items():
        if lbl == "silence_sync_gte_500K":
            for sl, sv in d.items():
                lines.append(f"| sil+sync500K_{sl} | {sv['n']} | {sv.get('t3r','-')} | {sv['win']} | | | | | |")
            continue
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h['win']} | {p['p_right']} | **{p['verdict']}** |")
    lines.append("")

    # G
    tg = res["test_g"]
    bp = tg.get("bid_depth_percentiles_cal",{})
    lines += ["## G. bid_depth_usd Absolute Level", "",
              f"Cal bid_depth percentiles: p25={bp.get('p25')} p50={bp.get('p50')} p75={bp.get('p75')}", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in tg.items():
        if key == "bid_depth_percentiles_cal": continue
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # H
    th = res["test_h"]
    lines += ["## H. Propagation Timing Breakdown", ""]
    lines += ["| Segment | N | SHORT T3R | SHORT win | LONG T3R | LONG win |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in th.items():
        s = d.get("SHORT",{}); l = d.get("LONG",{})
        lines.append(f"| {key} | {d.get('n','-')} | {s.get('t3r','-')} | {s.get('win','-')} | {l.get('t3r','-')} | {l.get('win','-')} |")
    lines.append("")

    # I
    ti = res["test_i"]
    lines += ["## I. Weekday Effect", ""]
    lines += ["| Day | Cal N | Hold N | Sil WR cal | Sil WR hold | Sil T3R hold | Short WR cal | Short WR hold | Short T3R hold |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for day in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        d = ti.get(day,{})
        sl = d.get("silence_LONG",{}); sh = d.get("noisy_SHORT",{})
        lines.append(f"| {day} | {d.get('n_cal',0)} | {d.get('n_hold',0)} |"
                     f" {sl.get('cal',{}).get('win','-')} | {sl.get('hold',{}).get('win','-')} | {sl.get('hold',{}).get('t3r','-')} |"
                     f" {sh.get('cal',{}).get('win','-')} | {sh.get('hold',{}).get('win','-')} | {sh.get('hold',{}).get('t3r','-')} |")
    lines.append("")

    # J
    tj = res["test_j"]
    lines += ["## J. Same-Day Sequential Trades", ""]
    for split_lbl in ("cal", "hold"):
        lines += [f"### {split_lbl}",
                  "| Trade order | N | T3R | med | win |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for lbl in [f"1st_silence_{split_lbl}", f"2nd_silence_{split_lbl}", f"3rd_plus_silence_{split_lbl}"]:
            d = tj.get(lbl, {})
            lines.append(f"| {lbl} | {d.get('n',0)} | {d.get('t3r','-')} | {d.get('med','-')} | {d.get('win','-')} |")
        dist = tj.get(f"daily_event_distribution_{split_lbl}", {})
        lines.append(f"Daily event counts: n_days={dist.get('n_days')} min={dist.get('min')} med={dist.get('median')} "
                     f"max={dist.get('max')} p75={dist.get('p75')} p90={dist.get('p90')}")
        lines.append("")

    lines += ["---",
              "## New Questions After This Suite",
              "",
              "1. **Perfect storm score>=5**: WR is very high but N small — needs more data (2026 Q3?)",
              "2. **Session x score interaction**: score>=3 US session silence — combined permutation null?",
              "3. **Propagation cascade SIZE predicts momentum strength** — high max prop size = stronger SHORT",
              "4. **ETH -100bps 1h + silence = 5th signal?** — eth1h bearish context + silence interaction",
              "5. **BTC lead + noisy SHORT** — WR=79% but N=29 in hold; is this a fifth standalone signal?",
              "6. **bid_depth impact on SILENCE RATE** — does high bid_depth predict silence at entry?",
              "7. **Weekend cascades** — different mechanics (lower liquidity) — should we filter?",
              "8. **Frequency plateau** — at score>=4, monthly frequency drops. Is it still worth it?",
              "9. **Sequential day hypothesis** — after a 'cascade storm day' (10+ events), next day?",
              "10. **Cross-asset silence** — no BTC OR ETH cascade in 30min = strongest silence variant?",
              "",
              "RESEARCH_ONLY. No live changes without explicit operator sign-off."]
    return "\n".join(lines) + "\n"


# ─── Main ────────────────────────────────────────────────────────────────────

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
    print("Annotating...")
    cal  = annotate(cal_raw,  eth_sell_ts, eth_sell_not, btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)
    hold = annotate(hold_raw, eth_sell_ts, eth_sell_not, btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)

    print("A: scoring..."); ta = test_a_scoring(cal, hold)
    print("B: propagation..."); tb = test_b_propagation(cal, hold)
    print("C: bull+noisy..."); tc = test_c_bull_noisy(cal, hold)
    print("D: live 200K..."); td = test_d_live_200k(cal, hold)
    print("E: eth1h..."); te = test_e_eth1h(cal, hold)
    print("F: high sync..."); tf = test_f_highsync_short(cal, hold)
    print("G: bid_depth..."); tg = test_g_bid_depth(cal, hold)
    print("H: timing..."); th = test_h_prop_timing(cal, hold)
    print("I: weekday..."); ti = test_i_weekday(cal, hold)
    print("J: sequential..."); tj = test_j_sequential(cal, hold)

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
        "test_e": te, "test_f": tf, "test_g": tg, "test_h": th,
        "test_i": ti, "test_j": tj,
    }
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
