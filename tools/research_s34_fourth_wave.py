"""S34 Fourth Wave Research Suite.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

10 new tests from third-wave open questions:
  A. bid_depth=0 filter: silence rate + outcome when bid_depth=0 vs >0
  B. 4+ cascade + silence: ultra-event — is this a standalone validated signal?
  C. Sunday silence LONG: WR=86% with small N — permutation null
  D. 200K + cluster + bear + bid_depth: 5th validated signal candidate (perm null hold)
  E. Ultra-early (<1min) SHORT mechanics: what IS different about these events?
  F. Score>=3 + bid_nonzero combined system: refined portfolio T3R
  G. Wed+Thu US session + score>=3 silence: best quality sub-signal?
  H. ETH 1h bear + bid_nonzero + silence: combined quality gate
  I. BULL_PULLBACK + noisy LONG: full analysis + perm null on hold events
  J. Cross-asset silence: ETH AND BTC both quiet = strongest silence variant?

DAT-01/SAF-02: no lookahead, research only.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import load_jsonl, r1, r3, NAV_EVENTS, FEE_BPS

DEFAULT_DB  = ROOT / "data" / "microstructure.db"
OUT_MD      = ROOT / "reports" / "research" / "s34" / "S34_FOURTH_WAVE.md"
OUT_JSON    = ROOT / "reports" / "research" / "s34" / "S34_FOURTH_WAVE.json"

HOLDOUT_FRAC  = 0.30
SEED          = 42
N_PERM        = 2000          # more perms for small-N signals
MIN_N         = 10
SYNC_WIN_MS   = 10 * 60_000
SIL_LO        = 60_000
SIL_HI_ETH    = 30 * 60_000
SIL_HI_BTC    = 30 * 60_000
PROP_THRESH   = 50_000.0
BTC_SIL_THRESH= 500_000.0    # BTC silence threshold (larger = less sensitive)
LIVE_THRESH   = 200_000.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def utc_now(): return datetime.now(timezone.utc).isoformat()
def ts_utc(ts):
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def t3r(vals):
    if len(vals) <= 3: return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])

def qs(vals):
    if not vals:
        return {"n":0,"t3r":None,"sum":None,"med":None,"win":None,"maxL":None,"maxW":None}
    t = t3r(vals) if len(vals) >= MIN_N else None
    return {
        "n":    len(vals),
        "t3r":  r1(t) if t is not None else None,
        "sum":  r1(sum(vals)),
        "med":  r1(median(vals)),
        "win":  r3(sum(1 for v in vals if v > 0) / len(vals)),
        "maxL": r1(min(vals)),
        "maxW": r1(max(vals)),
    }

def pctile(vals, p):
    v = sorted(x for x in vals if math.isfinite(x))
    if not v: return float("nan")
    i = p*(len(v)-1); lo = int(i)
    return v[lo] + (i - lo)*(v[min(lo+1, len(v)-1)] - v[lo])

def permtest(target_vals, pool, n_perm, seed, label=""):
    rng = random.Random(seed)
    real = t3r(target_vals) if len(target_vals) >= MIN_N else float("nan")
    n = len(target_vals)
    null = [t3r(rng.sample(pool, min(n, len(pool)))) for _ in range(n_perm)]
    p95      = pctile(null, 0.95)
    p_right  = sum(1 for v in null if math.isfinite(v) and v >= real) / max(len(null), 1)
    return {
        "label": label, "n": n,
        "real_t3r": r1(real), "null_p95": r1(p95),
        "p_right": r3(p_right),
        "verdict": "PASS" if p_right < 0.05 else "ARTIFACT",
    }


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
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    for i in range(a, b):
        if vals[i] >= thr:
            return ts_list[i] - lo
    return None


# ─── Annotate ────────────────────────────────────────────────────────────────

def annotate(rows,
             eth_sell_ts, eth_sell_not,
             btc_sell_ts, btc_sell_not,
             sol_sell_ts, sol_sell_not):
    out = []
    for r in rows:
        ts = int(r["signal_ts_ms"])

        # ETH silence (30-min window)
        n_eth_prop = win_count_thresh(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI_ETH, PROP_THRESH)
        silence_eth = n_eth_prop == 0

        # BTC silence (no large BTC cascade in same 30-min window)
        max_btc = win_max(btc_sell_ts, btc_sell_not, ts+SIL_LO, ts+SIL_HI_BTC)
        silence_btc = max_btc < BTC_SIL_THRESH

        # Cross-asset silence = both ETH and BTC quiet
        silence_both = silence_eth and silence_btc

        # Propagation metrics
        prop_count = win_count_thresh(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI_ETH, PROP_THRESH)
        prop_max   = win_max(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI_ETH)
        prop_first_ms = first_ts_above(eth_sell_ts, eth_sell_not, ts+SIL_LO, ts+SIL_HI_ETH, PROP_THRESH)

        # sync_k
        b  = win_sum(btc_sell_ts, btc_sell_not, ts - SYNC_WIN_MS, ts)
        s  = win_sum(sol_sell_ts, sol_sell_not, ts - SYNC_WIN_MS, ts)
        sync_k = b + s

        # n_prior2h
        n_prior2h = win_count_thresh(eth_sell_ts, eth_sell_not, ts - 2*3600_000, ts - 1000, PROP_THRESH)

        # time
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        hour_utc = dt.hour
        weekday  = dt.weekday()          # 0=Mon
        day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][weekday]
        if 7 <= hour_utc < 13:    session = "EU"
        elif 13 <= hour_utc < 21: session = "US"
        else:                      session = "ASIA"

        # fields
        net_2h   = float(r.get("net_2h_bps")  or "nan")
        net_4h_v = r.get("net_4h_bps")
        net_4h   = float(net_4h_v) if net_4h_v is not None else float("nan")
        prior4h  = float(r.get("prior4h_bps")  or 0)
        vdepth   = float(r.get("vdepth_bps")   or 0)
        eth1h    = float(r.get("eth1h_bps")    or 0)
        btc4h    = float(r.get("btc4h_bps")    or 0)
        thresh   = float(r.get("threshold_usd") or 0)
        bid_dep  = float(r.get("bid_depth_usd") or 0)
        is_bull  = "BULL_PULLBACK" in (r.get("tags") or [])
        is_live  = thresh >= LIVE_THRESH

        # 6-point score
        score = sum([
            int(silence_eth),
            int(n_prior2h >= 3),
            int(btc4h < 0),
            int(vdepth >= 30),
            int(session == "US"),
            int(sync_k >= 200_000),
        ])

        item = dict(r)
        item.update({
            "silence_eth":   silence_eth,
            "silence_btc":   silence_btc,
            "silence_both":  silence_both,
            "prop_count":    prop_count,
            "prop_max":      prop_max,
            "prop_first_ms": prop_first_ms,
            "sync_k":        sync_k,
            "n_prior2h":     n_prior2h,
            "session":       session,
            "hour_utc":      hour_utc,
            "weekday":       weekday,
            "day_name":      day_name,
            "net_2h":        net_2h,
            "net_4h":        net_4h,
            "prior4h":       prior4h,
            "vdepth":        vdepth,
            "eth1h":         eth1h,
            "btc4h":         btc4h,
            "thresh":        thresh,
            "bid_dep":       bid_dep,
            "is_bull":       is_bull,
            "is_live":       is_live,
            "score":         score,
        })
        out.append(item)
    return out


# ─── A — bid_depth=0 filter analysis ─────────────────────────────────────────

def test_a_bid_depth_zero(cal, hold):
    """
    bid_depth_usd = 0 means no bid depth data at cascade time.
    Test: silence gate WR when bid_depth=0 vs >0.
    Also: does bid_depth=0 correlate with specific conditions?
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        zero    = [r for r in rows if r["bid_dep"] == 0]
        nonzero = [r for r in rows if r["bid_dep"] >  0]

        def silence_long(subset):
            return [r["net_2h"] for r in subset if r["silence_eth"] and math.isfinite(r["net_2h"])]
        def noisy_short(subset):
            return [(-r["net_2h"]-2*FEE_BPS) for r in subset
                    if not r["silence_eth"] and not r["is_bull"] and math.isfinite(r["net_2h"])]

        n_total   = len(rows)
        n_zero    = len(zero)
        n_nonzero = len(nonzero)

        # silence rate within each group
        sil_rate_zero    = sum(1 for r in zero    if r["silence_eth"]) / max(n_zero,    1)
        sil_rate_nonzero = sum(1 for r in nonzero if r["silence_eth"]) / max(n_nonzero, 1)

        # average features in zero vs nonzero
        def avg_feature(subset, key):
            v = [r[key] for r in subset if math.isfinite(r.get(key, float("nan")))]
            return r1(sum(v)/len(v)) if v else None

        result[split_lbl] = {
            "n_total": n_total,
            "n_zero":  n_zero,
            "n_nonzero": n_nonzero,
            "zero_rate": r3(n_zero / n_total),
            "sil_rate_zero":    r3(sil_rate_zero),
            "sil_rate_nonzero": r3(sil_rate_nonzero),
            "bid_zero_silence_LONG":    qs(silence_long(zero)),
            "bid_zero_noisy_SHORT":     qs(noisy_short(zero)),
            "bid_nonzero_silence_LONG": qs(silence_long(nonzero)),
            "bid_nonzero_noisy_SHORT":  qs(noisy_short(nonzero)),
            "avg_sync_k_zero":    avg_feature(zero,    "sync_k"),
            "avg_sync_k_nonzero": avg_feature(nonzero, "sync_k"),
            "avg_vdepth_zero":    avg_feature(zero,    "vdepth"),
            "avg_vdepth_nonzero": avg_feature(nonzero, "vdepth"),
            "avg_thresh_zero":    avg_feature(zero,    "thresh"),
            "avg_thresh_nonzero": avg_feature(nonzero, "thresh"),
        }
    # permutation null: bid_nonzero silence LONG (cal)
    all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
    bnz_cal = [r["net_2h"] for r in cal if r["bid_dep"]>0 and r["silence_eth"] and math.isfinite(r["net_2h"])]
    result["perm_bid_nonzero_silence"] = permtest(bnz_cal, all_cal, N_PERM, SEED, "bid_nonzero_silence")
    return result


# ─── B — 4+ cascade + silence ────────────────────────────────────────────────

def test_b_ultra_event(cal, hold):
    """
    4+ cascades in prior 2h AND then silence = ultra exhaustion event.
    Is this a standalone validated signal?
    """
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    all_hold = [r["net_2h"] for r in hold if math.isfinite(r["net_2h"])]
    result   = {}

    for count_thr, lbl in [(3, "n_prior2h_gte3"), (4, "n_prior2h_gte4"),
                            (5, "n_prior2h_gte5"), (2, "n_prior2h_gte2")]:
        fn_sil  = lambda r, c=count_thr: r["n_prior2h"] >= c and r["silence_eth"]
        fn_nois = lambda r, c=count_thr: r["n_prior2h"] >= c and not r["silence_eth"] and not r["is_bull"]
        cal_sil  = [r["net_2h"] for r in cal  if fn_sil(r)  and math.isfinite(r["net_2h"])]
        hold_sil = [r["net_2h"] for r in hold if fn_sil(r)  and math.isfinite(r["net_2h"])]
        cal_nois = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn_nois(r) and math.isfinite(r["net_2h"])]
        hold_nois= [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn_nois(r) and math.isfinite(r["net_2h"])]
        perm_sil  = permtest(cal_sil,  all_cal, N_PERM, SEED, f"{lbl}_silence")
        all_short = [-v-2*FEE_BPS for v in all_cal]
        perm_nois = permtest(cal_nois, all_short, N_PERM, SEED+1, f"{lbl}_noisy_short")
        result[lbl] = {
            "silence_LONG": {"cal": qs(cal_sil), "hold": qs(hold_sil), "perm": perm_sil},
            "noisy_SHORT":  {"cal": qs(cal_nois),"hold": qs(hold_nois),"perm": perm_nois},
        }

    # Also: 4+ prop_count in noisy window (concurrent cascades, not prior 2h)
    fn_4plus_now = lambda r: r["prop_count"] >= 4 and not r["silence_eth"] and not r["is_bull"]
    c4p_cal  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn_4plus_now(r) and math.isfinite(r["net_2h"])]
    c4p_hold = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn_4plus_now(r) and math.isfinite(r["net_2h"])]
    all_short = [-v-2*FEE_BPS for v in all_cal]
    result["prop_count_gte4_noisy_SHORT"] = {
        "cal": qs(c4p_cal), "hold": qs(c4p_hold),
        "perm": permtest(c4p_cal, all_short, N_PERM, SEED+2, "prop_4plus_short"),
    }
    return result


# ─── C — Sunday silence LONG perm null ───────────────────────────────────────

def test_c_sunday(cal, hold):
    """
    Sunday silence hold WR=86.4% (small N).
    Perm null on cal. Also test all day-specific silence signals.
    """
    all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
    result  = {}
    for day in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        fn_sil  = lambda r, d=day: r["day_name"] == d and r["silence_eth"]
        fn_nois = lambda r, d=day: r["day_name"] == d and not r["silence_eth"] and not r["is_bull"]
        cal_sil  = [r["net_2h"] for r in cal  if fn_sil(r)  and math.isfinite(r["net_2h"])]
        hold_sil = [r["net_2h"] for r in hold if fn_sil(r)  and math.isfinite(r["net_2h"])]
        cal_nois = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn_nois(r) and math.isfinite(r["net_2h"])]
        hold_nois= [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn_nois(r) and math.isfinite(r["net_2h"])]
        perm_sil  = permtest(cal_sil,  all_cal, N_PERM, SEED, f"{day}_silence")
        all_short = [-v-2*FEE_BPS for v in all_cal]
        perm_nois = permtest(cal_nois, all_short, N_PERM, SEED+1, f"{day}_short")
        result[day] = {
            "silence_LONG": {
                "cal": qs(cal_sil), "hold": qs(hold_sil), "perm": perm_sil,
            },
            "noisy_SHORT": {
                "cal": qs(cal_nois), "hold": qs(hold_nois), "perm": perm_nois,
            },
        }
    return result


# ─── D — 200K + cluster + bear + bid_depth (5th signal candidate) ────────────

def test_d_fifth_signal(cal, hold):
    """
    Candidate for 5th validated signal:
    200K cascade + silence + deep cluster (n_prior2h>=3) + BTC bear (btc4h<0) + bid_depth>0
    Run perm null on BOTH cal and hold.
    """
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    all_hold = [r["net_2h"] for r in hold if math.isfinite(r["net_2h"])]
    result   = {}

    combos = {
        "200K_sil_cluster_bear_biddep": lambda r: (r["is_live"] and r["silence_eth"] and
                                                    r["n_prior2h"]>=3 and r["btc4h"]<0 and r["bid_dep"]>0),
        "200K_sil_cluster_bear":        lambda r: r["is_live"] and r["silence_eth"] and r["n_prior2h"]>=3 and r["btc4h"]<0,
        "200K_sil_biddep":              lambda r: r["is_live"] and r["silence_eth"] and r["bid_dep"]>0,
        "200K_sil_score3":              lambda r: r["is_live"] and r["silence_eth"] and r["score"]>=3,
        "200K_sil_score4":              lambda r: r["is_live"] and r["silence_eth"] and r["score"]>=4,
        "any_sil_score4_biddep":        lambda r: r["silence_eth"] and r["score"]>=4 and r["bid_dep"]>0,
        "any_sil_score3_biddep":        lambda r: r["silence_eth"] and r["score"]>=3 and r["bid_dep"]>0,
        "any_sil_cluster_bear_biddep":  lambda r: r["silence_eth"] and r["n_prior2h"]>=3 and r["btc4h"]<0 and r["bid_dep"]>0,
    }
    for lbl, fn in combos.items():
        cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
        hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        perm_cal  = permtest(cal_v,  all_cal,  N_PERM, SEED,   f"{lbl}_cal")
        perm_hold = permtest(hold_v, all_hold, N_PERM, SEED+1, f"{lbl}_hold")
        result[lbl] = {
            "cal":  qs(cal_v),  "perm_cal":  perm_cal,
            "hold": qs(hold_v), "perm_hold": perm_hold,
        }
    return result


# ─── E — Ultra-early (<1min) SHORT mechanics ─────────────────────────────────

def test_e_ultra_early(cal, hold):
    """
    When first propagation arrives < 1min after signal: WR=45.8% (SHORT trap).
    What's different? Feature profile vs normal early events.
    Also: what happens at H1 vs H2 vs H4 for ultra-early?
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        ultra  = [r for r in rows if not r["silence_eth"] and r["prop_first_ms"] is not None
                  and r["prop_first_ms"] < 60_000]
        normal = [r for r in rows if not r["silence_eth"] and r["prop_first_ms"] is not None
                  and r["prop_first_ms"] >= 60_000]
        sil    = [r for r in rows if r["silence_eth"]]

        def avg(subset, key):
            v = [r[key] for r in subset if math.isfinite(r.get(key, float("nan")))]
            return r1(sum(v)/len(v)) if v else None

        def short_vals(subset):
            return [(-r["net_2h"]-2*FEE_BPS) for r in subset
                    if not r["is_bull"] and math.isfinite(r["net_2h"])]

        result[split_lbl] = {
            # outcome by horizon
            "ultra_early_short_H1": qs([(-float(r.get("net_1h_bps",r["net_2h"]))-2*FEE_BPS)
                                         for r in ultra if not r["is_bull"] and math.isfinite(r["net_2h"])]),
            "ultra_early_short_H2": qs(short_vals(ultra)),
            "normal_early_short_H2":qs(short_vals(normal)),
            # feature profile
            "ultra_profile": {
                "n": len(ultra),
                "avg_sync_k": avg(ultra, "sync_k"),
                "avg_vdepth": avg(ultra, "vdepth"),
                "avg_btc4h":  avg(ultra, "btc4h"),
                "avg_prop_count": avg(ultra, "prop_count"),
                "avg_thresh": avg(ultra, "thresh"),
                "pct_US": r3(sum(1 for r in ultra if r["session"]=="US") / max(len(ultra),1)),
                "pct_highsync": r3(sum(1 for r in ultra if r["sync_k"]>=300_000) / max(len(ultra),1)),
            },
            "normal_profile": {
                "n": len(normal),
                "avg_sync_k": avg(normal, "sync_k"),
                "avg_vdepth": avg(normal, "vdepth"),
                "avg_btc4h":  avg(normal, "btc4h"),
                "avg_prop_count": avg(normal, "prop_count"),
                "pct_US": r3(sum(1 for r in normal if r["session"]=="US") / max(len(normal),1)),
                "pct_highsync": r3(sum(1 for r in normal if r["sync_k"]>=300_000) / max(len(normal),1)),
            },
        }
    return result


# ─── F — Score>=3 + bid_nonzero refined portfolio ────────────────────────────

def test_f_refined_portfolio(cal, hold):
    """
    Apply refined filters:
      LONG:  score>=3 + bid_dep>0 + silence_eth
      SHORT: score>=3 + bid_dep>0 + not silence + not bull + prop_count>1 + prop_max>=100K
    What is the combined T3R, WR, and coverage?
    """
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    all_hold = [r["net_2h"] for r in hold if math.isfinite(r["net_2h"])]
    result   = {}

    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        n_total = len(rows)

        # refined LONG gate
        long_fn  = lambda r: r["score"]>=3 and r["bid_dep"]>0 and r["silence_eth"]
        # refined SHORT gate
        short_fn = lambda r: (r["score"]>=3 and r["bid_dep"]>0 and not r["silence_eth"]
                              and not r["is_bull"] and r["prop_count"]>1 and r["prop_max"]>=100_000)
        # untraded
        untrade_fn = lambda r: not long_fn(r) and not short_fn(r)

        long_trades  = [r for r in rows if long_fn(r)  and math.isfinite(r["net_2h"])]
        short_trades = [r for r in rows if short_fn(r) and math.isfinite(r["net_2h"])]

        lvals = [r["net_2h"]               for r in long_trades]
        svals = [(-r["net_2h"]-2*FEE_BPS) for r in short_trades]
        cvals = lvals + svals

        n_long    = len(long_trades)
        n_short   = len(short_trades)
        n_untrade = sum(1 for r in rows if untrade_fn(r))

        result[split_lbl] = {
            "n_total": n_total, "n_long": n_long, "n_short": n_short,
            "n_untraded": n_untrade,
            "coverage": r3((n_long+n_short)/n_total),
            "LONG_only":  qs(lvals),
            "SHORT_only": qs(svals),
            "combined":   qs(cvals),
        }

    # permutation null on combined cal
    cal_long_v  = [r["net_2h"]               for r in cal if (r["score"]>=3 and r["bid_dep"]>0 and r["silence_eth"]) and math.isfinite(r["net_2h"])]
    cal_short_v = [(-r["net_2h"]-2*FEE_BPS) for r in cal if (r["score"]>=3 and r["bid_dep"]>0 and not r["silence_eth"] and not r["is_bull"] and r["prop_count"]>1 and r["prop_max"]>=100_000) and math.isfinite(r["net_2h"])]
    combined_cal = cal_long_v + cal_short_v
    pool_cal = all_cal + [-v-2*FEE_BPS for v in all_cal]
    result["perm_combined_cal"]  = permtest(combined_cal, pool_cal, N_PERM, SEED, "refined_portfolio_cal")

    hold_long_v  = [r["net_2h"]               for r in hold if (r["score"]>=3 and r["bid_dep"]>0 and r["silence_eth"]) and math.isfinite(r["net_2h"])]
    hold_short_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if (r["score"]>=3 and r["bid_dep"]>0 and not r["silence_eth"] and not r["is_bull"] and r["prop_count"]>1 and r["prop_max"]>=100_000) and math.isfinite(r["net_2h"])]
    combined_hold = hold_long_v + hold_short_v
    result["perm_combined_hold"] = permtest(combined_hold, all_hold, N_PERM, SEED+1, "refined_portfolio_hold")

    return result


# ─── G — Wed+Thu US session + score>=3 silence ───────────────────────────────

def test_g_wedthu_us(cal, hold):
    """
    Best time + best quality: Wed/Thu + US session + score>=3 silence.
    Also test all combinations of day subset + session + score for silence.
    """
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    result   = {}
    combos   = {
        "baseline_silence":        lambda r: r["silence_eth"],
        "US_session_silence":      lambda r: r["session"]=="US" and r["silence_eth"],
        "WedThu_silence":          lambda r: r["day_name"] in ("Wed","Thu") and r["silence_eth"],
        "WedThu_US_silence":       lambda r: r["day_name"] in ("Wed","Thu") and r["session"]=="US" and r["silence_eth"],
        "WedThu_US_score3":        lambda r: r["day_name"] in ("Wed","Thu") and r["session"]=="US" and r["score"]>=3 and r["silence_eth"],
        "MonFri_silence":          lambda r: r["day_name"] in ("Mon","Fri") and r["silence_eth"],
        "weekday_silence":         lambda r: r["weekday"] < 5 and r["silence_eth"],
        "weekend_silence":         lambda r: r["weekday"] >= 5 and r["silence_eth"],
        "score3_silence":          lambda r: r["score"]>=3 and r["silence_eth"],
        "score3_US_silence":       lambda r: r["score"]>=3 and r["session"]=="US" and r["silence_eth"],
        "score3_biddep_silence":   lambda r: r["score"]>=3 and r["bid_dep"]>0 and r["silence_eth"],
        "score3_US_biddep_silence":lambda r: r["score"]>=3 and r["session"]=="US" and r["bid_dep"]>0 and r["silence_eth"],
        "score4_US_silence":       lambda r: r["score"]>=4 and r["session"]=="US" and r["silence_eth"],
        # SHORT combos
        "US_score3_noisy_short":   lambda r: r["session"]=="US" and r["score"]>=3 and not r["silence_eth"] and not r["is_bull"],
        "WedThu_US_score3_short":  lambda r: r["day_name"] in ("Wed","Thu") and r["session"]=="US" and r["score"]>=3 and not r["silence_eth"] and not r["is_bull"],
    }
    for lbl, fn in combos.items():
        is_short = "short" in lbl.lower()
        if is_short:
            cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = [-v-2*FEE_BPS for v in all_cal]
        else:
            cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = all_cal
        perm = permtest(cal_v, pool, N_PERM, SEED, lbl)
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v), "perm": perm}
    return result


# ─── H — ETH 1h bear + bid_nonzero + silence ─────────────────────────────────

def test_h_eth1h_bear(cal, hold):
    """
    eth1h < -50 (prior 1h ETH falling) + bid_depth>0 + silence gate.
    Also: eth1h_bear_lt-50 cross with other gates.
    """
    all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
    result  = {}
    combos  = {
        "eth1h_lt-50_silence":           lambda r: r["eth1h"]<-50 and r["silence_eth"],
        "eth1h_lt-50_biddep_silence":    lambda r: r["eth1h"]<-50 and r["bid_dep"]>0 and r["silence_eth"],
        "eth1h_lt-100_biddep_silence":   lambda r: r["eth1h"]<-100 and r["bid_dep"]>0 and r["silence_eth"],
        "eth1h_lt-50_score3_silence":    lambda r: r["eth1h"]<-50 and r["score"]>=3 and r["silence_eth"],
        "eth1h_lt-50_cluster_silence":   lambda r: r["eth1h"]<-50 and r["n_prior2h"]>=3 and r["silence_eth"],
        "eth1h_bear_noisy_short":        lambda r: r["eth1h"]<-50 and not r["silence_eth"] and not r["is_bull"],
        "eth1h_bear_sync300_noisy_short":lambda r: r["eth1h"]<-50 and r["sync_k"]>=300_000 and not r["silence_eth"] and not r["is_bull"],
        "eth1h_bull_silence":            lambda r: r["eth1h"]>50 and r["silence_eth"],
        "eth1h_flat_silence":            lambda r: -10<=r["eth1h"]<=10 and r["silence_eth"],
    }
    for lbl, fn in combos.items():
        is_short = "short" in lbl
        if is_short:
            cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = [-v-2*FEE_BPS for v in all_cal]
        else:
            cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = all_cal
        perm = permtest(cal_v, pool, N_PERM, SEED, lbl)
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v), "perm": perm}
    return result


# ─── I — BULL_PULLBACK + noisy LONG full analysis ────────────────────────────

def test_i_bull_noisy_long(cal, hold):
    """
    BULL_PULLBACK + noisy (propagation): hold WR=90.9% (N=11).
    This suggests bulls defending even against cascade propagation.
    Try different horizons, feature analysis, perm null on hold.
    """
    all_hold = [r["net_2h"] for r in hold if math.isfinite(r["net_2h"])]
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    result   = {}

    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        bull_noisy = [r for r in rows if r["is_bull"] and not r["silence_eth"]]
        not_bull_noisy = [r for r in rows if not r["is_bull"] and not r["silence_eth"]]
        bull_sil   = [r for r in rows if r["is_bull"] and r["silence_eth"]]

        # horizons
        h2_bn = [r["net_2h"] for r in bull_noisy if math.isfinite(r["net_2h"])]
        h4_bn = [r["net_4h"] for r in bull_noisy if math.isfinite(r.get("net_4h", float("nan")))]

        # sub-filter: bull+noisy+sync low (low cross-asset pressure = bulls defending alone)
        bn_lowsync = [r["net_2h"] for r in bull_noisy if r["sync_k"]<200_000 and math.isfinite(r["net_2h"])]
        bn_hisync  = [r["net_2h"] for r in bull_noisy if r["sync_k"]>=200_000 and math.isfinite(r["net_2h"])]

        # feature profile
        def avg(s, k): v=[r.get(k,float("nan")) for r in s if math.isfinite(r.get(k,float("nan")))]; return r1(sum(v)/len(v)) if v else None

        result[split_lbl] = {
            "bull_noisy_LONG_H2": qs(h2_bn),
            "bull_noisy_LONG_H4": qs(h4_bn),
            "bull_noisy_LONG_lowsync": qs(bn_lowsync),
            "bull_noisy_LONG_hisync":  qs(bn_hisync),
            "bull_noisy_count": len(bull_noisy),
            "bull_silence_count": len(bull_sil),
            "not_bull_noisy_count": len(not_bull_noisy),
            "bull_noisy_profile": {
                "avg_prior4h":  avg(bull_noisy, "prior4h"),
                "avg_eth1h":    avg(bull_noisy, "eth1h"),
                "avg_btc4h":    avg(bull_noisy, "btc4h"),
                "avg_sync_k":   avg(bull_noisy, "sync_k"),
                "avg_vdepth":   avg(bull_noisy, "vdepth"),
                "avg_thresh":   avg(bull_noisy, "thresh"),
            },
        }
    # perm nulls
    bn_cal  = [r["net_2h"] for r in cal  if r["is_bull"] and not r["silence_eth"] and math.isfinite(r["net_2h"])]
    bn_hold = [r["net_2h"] for r in hold if r["is_bull"] and not r["silence_eth"] and math.isfinite(r["net_2h"])]
    result["perm_cal"]  = permtest(bn_cal,  all_cal,  N_PERM, SEED,   "bull_noisy_long_cal")
    result["perm_hold"] = permtest(bn_hold, all_hold, N_PERM, SEED+1, "bull_noisy_long_hold")
    return result


# ─── J — Cross-asset silence ─────────────────────────────────────────────────

def test_j_cross_silence(cal, hold):
    """
    Cross-asset silence: ETH quiet (silence_eth) AND BTC quiet (no large BTC cascade).
    BTC silence threshold: max BTC SELL cascade < 500K in 30-min window.
    Hypothesis: double silence = sellers exhausted across both assets = strongest fade.
    """
    all_cal  = [r["net_2h"] for r in cal  if math.isfinite(r["net_2h"])]
    all_hold = [r["net_2h"] for r in hold if math.isfinite(r["net_2h"])]
    result   = {}

    combos = {
        "eth_silence_only":         lambda r: r["silence_eth"] and not r["silence_btc"],
        "btc_silence_only":         lambda r: not r["silence_eth"] and r["silence_btc"],
        "both_silence":             lambda r: r["silence_both"],
        "both_silence_score3":      lambda r: r["silence_both"] and r["score"]>=3,
        "both_silence_biddep":      lambda r: r["silence_both"] and r["bid_dep"]>0,
        "both_silence_score3_biddep": lambda r: r["silence_both"] and r["score"]>=3 and r["bid_dep"]>0,
        "eth_sil_btc_noisy_short":  lambda r: r["silence_eth"] and not r["silence_btc"]
                                              and not r["silence_eth"],  # noisy eth during btc cascade
        "btc_sil_eth_noisy_short":  lambda r: not r["silence_eth"] and r["silence_btc"]
                                              and not r["is_bull"],
        "neither_silence_short":    lambda r: not r["silence_eth"] and not r["silence_btc"]
                                              and not r["is_bull"],
    }
    for lbl, fn in combos.items():
        is_short = "short" in lbl
        if is_short:
            cal_v  = [(-r["net_2h"]-2*FEE_BPS) for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [(-r["net_2h"]-2*FEE_BPS) for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = [-v-2*FEE_BPS for v in all_cal]
        else:
            cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
            pool   = all_cal
        perm = permtest(cal_v, pool, N_PERM, SEED, lbl)
        result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v), "perm": perm}

    # silence rate breakdown
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        n   = len(rows)
        n_eth_sil  = sum(1 for r in rows if r["silence_eth"])
        n_btc_sil  = sum(1 for r in rows if r["silence_btc"])
        n_both_sil = sum(1 for r in rows if r["silence_both"])
        result[f"rates_{split_lbl}"] = {
            "n_total": n,
            "eth_sil_rate":  r3(n_eth_sil/n),
            "btc_sil_rate":  r3(n_btc_sil/n),
            "both_sil_rate": r3(n_both_sil/n),
        }
    return result


# ─── Render ──────────────────────────────────────────────────────────────────

def r_perm(p):
    return f"p={p['p_right']} real={p['real_t3r']} null_p95={p['null_p95']} N={p.get('n','-')} -> **{p['verdict']}**"

def r_qs(d, prefix=""):
    return (f"N={d['n']} T3R={d.get('t3r','-')} med={d.get('med','-')} "
            f"win={d.get('win','-')} maxL={d.get('maxL','-')}")

def render_md(res):
    sp = res["split"]
    lines = [
        "# S34 Fourth Wave Research Suite",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        f"Cal: {sp['cal_n']} ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} ({sp['hold_start']} to {sp['hold_end']})",
        "",
    ]

    # A
    ta = res["test_a"]
    lines += ["## A. bid_depth=0 Filter Analysis", ""]
    for split_lbl in ("cal", "hold"):
        d = ta[split_lbl]
        lines += [
            f"### {split_lbl}",
            f"Total N={d['n_total']}  bid_zero N={d['n_zero']} ({d['zero_rate']*100:.1f}%)  bid_nonzero N={d['n_nonzero']}",
            f"Silence rate: bid_zero={d['sil_rate_zero']} bid_nonzero={d['sil_rate_nonzero']}",
            f"avg_sync_k: zero={d['avg_sync_k_zero']}  nonzero={d['avg_sync_k_nonzero']}",
            f"avg_vdepth: zero={d['avg_vdepth_zero']}  nonzero={d['avg_vdepth_nonzero']}",
            f"avg_thresh: zero={d['avg_thresh_zero']}  nonzero={d['avg_thresh_nonzero']}",
            "",
            f"| Gate | N | T3R | med | win |",
            f"| --- | ---: | ---: | ---: | ---: |",
        ]
        for key in ["bid_zero_silence_LONG","bid_zero_noisy_SHORT","bid_nonzero_silence_LONG","bid_nonzero_noisy_SHORT"]:
            s = d[key]
            lines.append(f"| {key} | {s['n']} | {s.get('t3r','-')} | {s.get('med','-')} | {s.get('win','-')} |")
        lines.append("")
    p = ta["perm_bid_nonzero_silence"]
    lines += [f"Perm bid_nonzero silence (cal): {r_perm(p)}", ""]

    # B
    tb = res["test_b"]
    lines += ["## B. Ultra-Event: Cluster (prior 2h) + Silence & 4+ Cascades Now", ""]
    lines += ["| Signal | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in tb.items():
        if lbl == "prop_count_gte4_noisy_SHORT":
            c = d["cal"]; h = d["hold"]; p = d["perm"]
            lines.append(f"| prop_count>=4 SHORT | now | {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} |"
                         f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} | {p['p_right']} **{p['verdict']}** |")
            continue
        for gate in ["silence_LONG", "noisy_SHORT"]:
            gd = d[gate]; c = gd["cal"]; h = gd["hold"]; p = gd["perm"]
            lines.append(f"| {lbl} | {gate} | {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} |"
                         f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} | {p['p_right']} **{p['verdict']}** |")
    lines.append("")

    # C
    tc = res["test_c"]
    lines += ["## C. Day-of-Week Permutation Null", ""]
    lines += ["| Day | Sil Cal N | Sil Cal win | Sil Hold N | Sil Hold win | Sil Perm | Short Hold N | Short Hold win | Short Perm |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for day in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        d = tc[day]
        sl = d["silence_LONG"]; sh = d["noisy_SHORT"]
        ps = sl["perm"]; pn = sh["perm"]
        lines.append(f"| {day} |"
                     f" {sl['cal']['n']} | {sl['cal'].get('win','-')} |"
                     f" {sl['hold']['n']} | {sl['hold'].get('win','-')} | {ps['p_right']} **{ps['verdict']}** |"
                     f" {sh['hold']['n']} | {sh['hold'].get('win','-')} | {pn['p_right']} **{pn['verdict']}** |")
    lines.append("")

    # D
    td = res["test_d"]
    lines += ["## D. 5th Signal Candidate: 200K + Cluster + Bear + bid_depth", ""]
    lines += ["| Signal | Cal N | Cal T3R | Cal win | Cal Perm | Hold N | Hold T3R | Hold win | Hold Perm |",
              "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for lbl, d in td.items():
        c = d["cal"]; h = d["hold"]; pc = d["perm_cal"]; ph = d["perm_hold"]
        lines.append(f"| {lbl} |"
                     f" {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} | {pc['p_right']} **{pc['verdict']}** |"
                     f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} | {ph['p_right']} **{ph['verdict']}** |")
    lines.append("")

    # E
    te = res["test_e"]
    lines += ["## E. Ultra-Early (<1min) SHORT Mechanics", ""]
    for split_lbl in ("cal", "hold"):
        d = te[split_lbl]
        up = d["ultra_profile"]; np_ = d["normal_profile"]
        lines += [
            f"### {split_lbl}",
            f"Ultra-early N={up['n']}  Normal-early N={np_['n']}",
            f"| Feature | Ultra (<1min) | Normal (>=1min) |",
            f"| --- | ---: | ---: |",
            f"| avg sync_k | {up['avg_sync_k']} | {np_['avg_sync_k']} |",
            f"| avg vdepth | {up['avg_vdepth']} | {np_['avg_vdepth']} |",
            f"| avg btc4h  | {up['avg_btc4h']} | {np_['avg_btc4h']} |",
            f"| avg prop_count | {up['avg_prop_count']} | {np_['avg_prop_count']} |",
            f"| pct US session | {up['pct_US']} | {np_['pct_US']} |",
            f"| pct sync>=300K | {up['pct_highsync']} | {np_['pct_highsync']} |",
            "",
            f"Ultra SHORT H2: {r_qs(d['ultra_early_short_H2'])}",
            f"Normal SHORT H2: {r_qs(d['normal_early_short_H2'])}",
            "",
        ]

    # F
    tf = res["test_f"]
    lines += ["## F. Refined Portfolio (score>=3 + bid_dep>0 + prop filters)", ""]
    lines += ["| Split | Coverage | LONG N | LONG T3R | LONG win | SHORT N | SHORT T3R | SHORT win | Combined T3R | Combined win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for sl in ("cal", "hold"):
        d = tf[sl]
        l = d["LONG_only"]; s = d["SHORT_only"]; c = d["combined"]
        lines.append(f"| {sl} | {d['coverage']} | {l['n']} | {l.get('t3r','-')} | {l.get('win','-')} |"
                     f" {s['n']} | {s.get('t3r','-')} | {s.get('win','-')} |"
                     f" {c.get('t3r','-')} | {c.get('win','-')} |")
    pc = tf["perm_combined_cal"]; ph = tf["perm_combined_hold"]
    lines += ["", f"Perm (cal):  {r_perm(pc)}", f"Perm (hold): {r_perm(ph)}", ""]

    # G
    tg = res["test_g"]
    lines += ["## G. Wed+Thu US Session + Score>=3 (Best Subset Search)", ""]
    lines += ["| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in tg.items():
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} |"
                     f" {p['p_right']} | **{p['verdict']}** |")
    lines.append("")

    # H
    th = res["test_h"]
    lines += ["## H. ETH 1h Bear + bid_nonzero + Silence Gate", ""]
    lines += ["| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in th.items():
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} |"
                     f" {p['p_right']} | **{p['verdict']}** |")
    lines.append("")

    # I
    ti = res["test_i"]
    lines += ["## I. BULL_PULLBACK + noisy LONG (Full Analysis)", ""]
    for split_lbl in ("cal", "hold"):
        d = ti[split_lbl]
        prof = d["bull_noisy_profile"]
        lines += [
            f"### {split_lbl}: N_bull_noisy={d['bull_noisy_count']}  N_bull_sil={d['bull_silence_count']}",
            f"Profile: prior4h={prof['avg_prior4h']} eth1h={prof['avg_eth1h']} btc4h={prof['avg_btc4h']}"
            f" sync_k={prof['avg_sync_k']} vdepth={prof['avg_vdepth']} thresh={prof['avg_thresh']}",
            f"H2 LONG: {r_qs(d['bull_noisy_LONG_H2'])}",
            f"H4 LONG: {r_qs(d['bull_noisy_LONG_H4'])}",
            f"Low-sync (<200K) H2: {r_qs(d['bull_noisy_LONG_lowsync'])}",
            f"High-sync (>=200K) H2: {r_qs(d['bull_noisy_LONG_hisync'])}",
            "",
        ]
    pc = ti["perm_cal"]; ph = ti["perm_hold"]
    lines += [f"Perm (cal):  {r_perm(pc)}", f"Perm (hold): {r_perm(ph)}", ""]

    # J
    tj = res["test_j"]
    lines += ["## J. Cross-Asset Silence (ETH + BTC both quiet)", ""]
    for split_lbl in ("cal", "hold"):
        d = tj.get(f"rates_{split_lbl}", {})
        lines.append(f"**{split_lbl}** rates: ETH_sil={d.get('eth_sil_rate')} "
                     f"BTC_sil={d.get('btc_sil_rate')} Both_sil={d.get('both_sil_rate')}")
    lines += ["", "| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lbl, d in tj.items():
        if lbl.startswith("rates_"): continue
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {lbl} | {c['n']} | {c.get('t3r','-')} | {c.get('win','-')} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h.get('win','-')} |"
                     f" {p['p_right']} | **{p['verdict']}** |")
    lines += ["",
              "---",
              "## Synthesis — Running Signal Registry",
              "",
              "| # | Signal | Hold WR | Hold T3R | Perm Status |",
              "| --- | --- | ---: | ---: | --- |",
              "| 1 | Silence LONG (30min) | 70.1% | +7733 | p=0.0 PASS |",
              "| 2 | Silence + sync>=200K LONG | 83.1% | +4298 | p=0.0 PASS x2 |",
              "| 3 | noisy_NOT_bull SHORT | 54.9% | +11360 | p=0.0 PASS |",
              "| 4 | prior4h_neg + silence LONG | 76.2% | +6741 | p=0.0 PASS |",
              "| 5 | Combined portfolio | 59.9% | +19952 | p=0.0 PASS |",
              "| ? | 5th signal candidate | ? | ? | See test D |",
              "",
              "RESEARCH_ONLY. No live changes without explicit operator sign-off.",
              ]
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

    print("A: bid_depth=0 filter...")
    ta = test_a_bid_depth_zero(cal, hold)
    print("B: ultra-event (cluster+silence)...")
    tb = test_b_ultra_event(cal, hold)
    print("C: day-of-week perm nulls...")
    tc = test_c_sunday(cal, hold)
    print("D: 5th signal candidate...")
    td = test_d_fifth_signal(cal, hold)
    print("E: ultra-early mechanics...")
    te = test_e_ultra_early(cal, hold)
    print("F: refined portfolio...")
    tf = test_f_refined_portfolio(cal, hold)
    print("G: WedThu US score3...")
    tg = test_g_wedthu_us(cal, hold)
    print("H: eth1h bear + biddep + silence...")
    th = test_h_eth1h_bear(cal, hold)
    print("I: bull+noisy LONG...")
    ti = test_i_bull_noisy_long(cal, hold)
    print("J: cross-asset silence...")
    tj = test_j_cross_silence(cal, hold)

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
        "test_a": ta, "test_b": tb, "test_c": tc, "test_d": td, "test_e": te,
        "test_f": tf, "test_g": tg, "test_h": th, "test_i": ti, "test_j": tj,
    }
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
