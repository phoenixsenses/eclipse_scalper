"""S34 Deep Expansion Suite — second wave of research.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Builds on 3 confirmed OOS signals:
  (1) Silence gate LONG      — hold WR=70.1%, T3R=+7733, perm p=0.0
  (2) Silence+sync>=200K LONG— hold WR=83.1%, T3R=+4298, perm p=0.0
  (3) noisy_AND_NOT_bull SHORT— hold WR=54.9%, T3R=+11360, perm p=0.0

New tests:
  A. Portfolio: silence LONG + noisy SHORT combined system
  B. Flip strategy: enter SHORT on noisy, flip to LONG on silence
  C. vdepth gate: does overshoot depth predict outcome quality?
  D. BTC context: btc4h_bps as silence gate modifier
  E. Book imbalance: bid support at cascade time
  F. prior4h permutation null: +81.5% WR real or small-N?
  G. Cascade sequence: is this 1st/2nd/3rd in a cluster? Effect on signal?
  H. Time-of-day breakdown: EU/US/ASIA session effects
  I. Tail event audit: what do the silence gate LOSERS look like?
  J. Frequency & Kelly: monthly trade count + optimal sizing
  K. BTC-led cascade: BTC cascade before ETH cascade -> ETH fade?
  L. Silence gate weekly holdout stability: week-by-week breakdown

SAF-02/DAT-01/DAT-03: research only, no live changes, no lookahead.
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
from statistics import median, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import load_jsonl, r1, r3, NAV_EVENTS, FEE_BPS

DEFAULT_DB       = ROOT / "data" / "microstructure.db"
OUT_MD           = ROOT / "reports" / "research" / "s34" / "S34_DEEP_EXPANSION.md"
OUT_JSON         = ROOT / "reports" / "research" / "s34" / "S34_DEEP_EXPANSION.json"

HOLDOUT_FRAC     = 0.30
SEED             = 42
N_PERM           = 1000
MIN_N            = 12
SYNC_WINDOW_MS   = 10 * 60 * 1000
SILENCE_LO_MS    = 60_000
SILENCE_HI_MS    = 30 * 60_000
PROP_THRESH      = 50_000.0
LIVE_THRESH      = 200_000.0
BTC_LEAD_WINDOW  = 30 * 60_000   # BTC cascade in prior 30min qualifies as lead


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now(): return datetime.now(timezone.utc).isoformat()

def ts_utc(ts):
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def t3r(vals):
    if len(vals) <= 3: return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])

def qs(vals):
    if not vals: return {"n":0,"t3r":None,"sum":None,"med":None,"win":None,"maxL":None,"maxW":None,"std":None}
    t = t3r(vals) if len(vals) >= MIN_N else None
    return {
        "n":    len(vals),
        "t3r":  r1(t) if t is not None else None,
        "sum":  r1(sum(vals)),
        "med":  r1(median(vals)),
        "win":  r3(sum(1 for v in vals if v > 0) / len(vals)),
        "maxL": r1(min(vals)),
        "maxW": r1(max(vals)),
        "std":  r1(stdev(vals)) if len(vals) >= 2 else None,
    }

def pctile(vals, p):
    v = sorted(x for x in vals if math.isfinite(x))
    if not v: return float("nan")
    i = p*(len(v)-1); lo = int(i)
    return v[lo] + (i-lo)*(v[min(lo+1, len(v)-1)] - v[lo])

def permtest(target_vals, all_vals, n_perm, seed):
    rng = random.Random(seed)
    real = t3r(target_vals) if len(target_vals) >= MIN_N else float("nan")
    n = len(target_vals)
    null = [t3r(rng.sample(all_vals, min(n, len(all_vals)))) for _ in range(n_perm)]
    p95  = pctile(null, 0.95)
    p_right = sum(1 for v in null if math.isfinite(v) and v >= real) / max(len(null), 1)
    return {
        "real_t3r": r1(real), "null_p95": r1(p95),
        "p_right": r3(p_right),
        "verdict": "PASS" if p_right < 0.05 else "ARTIFACT",
    }


# ---------------------------------------------------------------------------
# DB loading helpers
# ---------------------------------------------------------------------------

def load_liq(conn, symbol, side):
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side)).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]

def win_sum(ts_list, vals, lo, hi):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return sum(vals[i] for i in range(a, b))

def win_max(ts_list, vals, lo, hi):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return max((vals[i] for i in range(a, b)), default=0.0)

def win_count_thresh(ts_list, vals, lo, hi, thr):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return sum(1 for i in range(a, b) if vals[i] >= thr)

def win_count_any(ts_list, lo, hi):
    a = bisect.bisect_left(ts_list, lo); b = bisect.bisect_right(ts_list, hi)
    return b - a


# ---------------------------------------------------------------------------
# Annotate rows
# ---------------------------------------------------------------------------

def annotate(rows, eth_sell_ts, eth_sell_not,
             btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not):
    out = []
    for r in rows:
        ts = int(r["signal_ts_ms"])

        # 30-min silence (main gate)
        n_prop = win_count_thresh(eth_sell_ts, eth_sell_not,
                                  ts + SILENCE_LO_MS, ts + SILENCE_HI_MS, PROP_THRESH)
        silence30 = n_prop == 0

        # sync_k (BTC+SOL SELL in prior 10min)
        b = win_sum(btc_sell_ts, btc_sell_not, ts - SYNC_WINDOW_MS, ts)
        s = win_sum(sol_sell_ts, sol_sell_not, ts - SYNC_WINDOW_MS, ts)
        sync_k = b + s

        # BTC lead: BTC SELL cascade >= 2M in prior 30min
        btc_lead = win_max(btc_sell_ts, btc_sell_not, ts - BTC_LEAD_WINDOW, ts) >= 2_000_000.0

        # Cascade cluster: how many ETH SELL >= 50K cascades in prior 2h?
        n_prior2h = win_count_thresh(eth_sell_ts, eth_sell_not,
                                     ts - 2*3600_000, ts - 1000, PROP_THRESH)

        # session (UTC hours): EU=7-14, US=13-21, ASIA=0-7
        hour_utc = datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour
        if 7 <= hour_utc < 13:
            session = "EU"
        elif 13 <= hour_utc < 21:
            session = "US"
        else:
            session = "ASIA"

        # field extractions
        net_2h  = float(r.get("net_2h_bps")  or "nan")
        net_4h  = float(r.get("net_4h_bps")  or "nan") if r.get("net_4h_bps") is not None else float("nan")
        prior4h = float(r.get("prior4h_bps") or 0)
        vdepth  = float(r.get("vdepth_bps")  or 0)
        book_imb = float(r.get("book_imbalance") or 0)
        eth1h   = float(r.get("eth1h_bps")   or 0)
        btc4h   = float(r.get("btc4h_bps")   or 0)
        thresh  = float(r.get("threshold_usd") or 0)
        is_bull  = "BULL_PULLBACK" in (r.get("tags") or [])
        is_live  = thresh >= LIVE_THRESH

        item = dict(r)
        item.update({
            "silence30": silence30,
            "sync_k": sync_k,
            "btc_lead": btc_lead,
            "n_prior2h": n_prior2h,
            "session": session,
            "hour_utc": hour_utc,
            "net_2h": net_2h,
            "net_4h": net_4h,
            "prior4h": prior4h,
            "vdepth": vdepth,
            "book_imb": book_imb,
            "eth1h": eth1h,
            "btc4h": btc4h,
            "thresh": thresh,
            "is_bull": is_bull,
            "is_live": is_live,
        })
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# A — Signal portfolio (combined LONG + SHORT system)
# ---------------------------------------------------------------------------

def test_a_portfolio(cal, hold):
    """
    Trade both signals simultaneously:
      - silence30 -> enter LONG at anchor, net_2h
      - noisy AND NOT bull -> enter SHORT at anchor, -net_2h - 2*fee
    Compute combined portfolio: total T3R, win rate, event counts.
    """
    def long_val(r):
        return r["net_2h"] if math.isfinite(r["net_2h"]) else None
    def short_val(r):
        v = r["net_2h"]
        return (-v - 2*FEE_BPS) if math.isfinite(v) else None

    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        long_trades  = [(r, v) for r in rows if r["silence30"]       and (v:=long_val(r))  is not None]
        short_trades = [(r, v) for r in rows if not r["silence30"] and not r["is_bull"] and (v:=short_val(r)) is not None]
        all_trades   = [(lbl, v) for lbl, v in [("L", lv) for _, lv in long_trades] +
                                                [("S", sv) for _, sv in short_trades]]
        l_vals = [v for _, v in long_trades]
        s_vals = [v for _, v in short_trades]
        a_vals = [v for _, v in all_trades]

        # untraded: bull context, not silence (neither long nor short)
        untraded = [r for r in rows if not r["silence30"] and r["is_bull"]]

        # long-only vs short-only vs combined
        result[split_lbl] = {
            "long_only":   qs(l_vals),
            "short_only":  qs(s_vals),
            "combined":    qs(a_vals),
            "n_untraded":  len(untraded),
            "trade_rate":  r3(len(a_vals) / len(rows)) if rows else None,
        }

        # coverage: what fraction of events are traded?
        n_events = len(rows)
        n_traded = len(long_trades) + len(short_trades)
        result[split_lbl]["coverage"] = r3(n_traded / n_events) if n_events else None
    return result


# ---------------------------------------------------------------------------
# B — Flip strategy: SHORT on noisy, LONG on silence
# ---------------------------------------------------------------------------

def test_b_flip(cal, hold):
    """
    Enter SHORT at cascade anchor (always).
    Monitor 30-min window:
      - If silence confirmed: FLIP to LONG (cover short, enter long)
        -> SHORT exit at ts+30min mark price (approx), LONG entry at ts+30min
        -> Problem: mark price not loaded; use net_2h as proxy for long leg outcome
      - If propagation: hold short to ts+2h
    Approximation: for flipped trades, credit = SHORT 30-min gain + LONG remaining gain.
    Since we only have 2h mark prices, we approximate:
      - Noisy: short_2h = -net_2h - 2*fee
      - Silence+flip: short_30m = estimate from drift (~25bps up = -25 short, -fee) + long_remaining
        The flip cost is prohibitive if drift eats into short profit.
    Use simplified model: flip at 30min costs 1*FEE extra (no mark price data).
    For silence events: flip PnL ~= -(drift_in_silence) - FEE (short leg) + net_2h - FEE (long leg)
    We don't have drift per-event in this tool; use the population median drift=+25bps (from test B).
    Conservative estimate: short_leg = -25 - FEE, long_leg = net_2h - FEE -> total flip = net_2h - 35
    """
    ASSUMED_SILENCE_DRIFT = 25.0  # median price recovery in 30min silence window (from test B)
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        # Strategy 1: always SHORT, hold 2h
        always_short = [(-r["net_2h"] - 2*FEE_BPS) for r in rows if math.isfinite(r["net_2h"])]
        # Strategy 2: silence->LONG 2h, noisy->SHORT 2h (our baseline)
        baseline = []
        for r in rows:
            if not math.isfinite(r["net_2h"]): continue
            if r["silence30"]:
                baseline.append(r["net_2h"] - FEE_BPS)
            else:
                baseline.append(-r["net_2h"] - 2*FEE_BPS)
        # Strategy 3: always SHORT first; if silence, flip to LONG (approximate)
        flip_strategy = []
        for r in rows:
            if not math.isfinite(r["net_2h"]): continue
            if r["silence30"]:
                # SHORT leg in silence window: lost ~25bps (price went up) + FEE
                short_leg = -ASSUMED_SILENCE_DRIFT - FEE_BPS
                # LONG leg from ts+30min: (net_2h_from_anchor - drift) - FEE
                long_leg  = r["net_2h"] - ASSUMED_SILENCE_DRIFT - FEE_BPS
                flip_strategy.append(short_leg + long_leg)
            else:
                # No flip: short to 2h
                flip_strategy.append(-r["net_2h"] - 2*FEE_BPS)
        result[split_lbl] = {
            "always_short_2h": qs(always_short),
            "baseline_split":  qs(baseline),  # silence=LONG, noisy=SHORT
            "flip_strategy":   qs(flip_strategy),
            "note": "flip approximation: silence drift assumed +25bps from prior test",
        }
    return result


# ---------------------------------------------------------------------------
# C — vdepth gate (overshoot depth)
# ---------------------------------------------------------------------------

def test_c_vdepth(cal, hold):
    """
    vdepth_bps: how far the cascade overshot below VWAP.
    Hypothesis: deeper V -> stronger mean reversion -> better silence fade.
    """
    result = {"bins": {}}
    # compute percentiles for vdepth in cal
    cal_vd = sorted(r["vdepth"] for r in cal if math.isfinite(r["vdepth"]) and r["vdepth"] != 0)
    p25 = pctile(cal_vd, 0.25) if cal_vd else 0
    p50 = pctile(cal_vd, 0.50) if cal_vd else 0
    p75 = pctile(cal_vd, 0.75) if cal_vd else 0
    result["vdepth_percentiles_cal"] = {"p25": r1(p25), "p50": r1(p50), "p75": r1(p75)}

    for label, lo, hi in [
        ("vdepth_q1",      0,    p25),
        ("vdepth_q2",    p25,    p50),
        ("vdepth_q3",    p50,    p75),
        ("vdepth_q4",    p75,  1e9),
        ("vdepth_lt_15",   0,     15),
        ("vdepth_15_30",  15,     30),
        ("vdepth_30_60",  30,     60),
        ("vdepth_gt_60",  60,   1e9),
    ]:
        def fn(r, lo=lo, hi=hi): return lo <= r["vdepth"] < hi
        for gate, gfn in [
            ("all",      fn),
            ("silence",  lambda r, f=fn: f(r) and r["silence30"]),
            ("noisy",    lambda r, f=fn: f(r) and not r["silence30"]),
        ]:
            cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result["bins"][f"{label}_{gate}"] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# D — BTC context (btc4h_bps)
# ---------------------------------------------------------------------------

def test_d_btc_context(cal, hold):
    result = {}
    for label, fn in [
        ("btc4h_bull_gt100",  lambda r: r["btc4h"] > 100),
        ("btc4h_bull_0_100",  lambda r: 0 < r["btc4h"] <= 100),
        ("btc4h_bear",        lambda r: r["btc4h"] <= 0),
        ("btc4h_bear_lt-100", lambda r: r["btc4h"] < -100),
        ("btc_lead_cascade",  lambda r: r["btc_lead"]),
        ("no_btc_lead",       lambda r: not r["btc_lead"]),
    ]:
        for gate, gfn in [
            ("all",     fn),
            ("silence", lambda r, f=fn: f(r) and r["silence30"]),
            ("noisy",   lambda r, f=fn: f(r) and not r["silence30"]),
        ]:
            key = f"{label}_{gate}"
            cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[key] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# E — Book imbalance gate
# ---------------------------------------------------------------------------

def test_e_book_imbalance(cal, hold):
    """book_imbalance: positive = more bid depth (buyers strong), negative = ask heavy."""
    result = {}
    # percentile on cal
    cal_bi = sorted(r["book_imb"] for r in cal if math.isfinite(r["book_imb"]))
    bp25 = pctile(cal_bi, 0.25) if cal_bi else 0
    bp75 = pctile(cal_bi, 0.75) if cal_bi else 0
    result["book_imb_percentiles_cal"] = {"p25": r1(bp25), "p75": r1(bp75)}

    for label, fn in [
        ("bid_heavy_q4",    lambda r: r["book_imb"] >= bp75),
        ("bid_heavy_q3q4",  lambda r: r["book_imb"] >= bp25),
        ("ask_heavy_q1",    lambda r: r["book_imb"] < bp25),
        ("bid_heavy_pos",   lambda r: r["book_imb"] > 0),
        ("ask_heavy_neg",   lambda r: r["book_imb"] <= 0),
    ]:
        for gate, gfn in [
            ("all",     fn),
            ("silence", lambda r, f=fn: f(r) and r["silence30"]),
            ("noisy_short", lambda r, f=fn: (v := -r["net_2h"] - 2*FEE_BPS) and f(r) and not r["silence30"]),
        ]:
            key = f"{label}_{gate}"
            if gate == "noisy_short":
                cal_v  = [(-r["net_2h"] - 2*FEE_BPS) for r in cal  if fn(r) and not r["silence30"] and math.isfinite(r["net_2h"])]
                hold_v = [(-r["net_2h"] - 2*FEE_BPS) for r in hold if fn(r) and not r["silence30"] and math.isfinite(r["net_2h"])]
            else:
                cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[key] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# F — prior4h permutation null
# ---------------------------------------------------------------------------

def test_f_prior4h(cal, hold):
    result = {}
    for label, fn in [
        ("prior4h_gt100",      lambda r: r["prior4h"] > 100),
        ("prior4h_gt50",       lambda r: r["prior4h"] > 50),
        ("prior4h_0_50",       lambda r: 0 <= r["prior4h"] <= 50),
        ("prior4h_neg",        lambda r: r["prior4h"] < 0),
        ("prior4h_lt-100",     lambda r: r["prior4h"] < -100),
        ("prior4h_gt100_sil",  lambda r: r["prior4h"] > 100 and r["silence30"]),
        ("prior4h_neg_sil",    lambda r: r["prior4h"] < 0   and r["silence30"]),
        ("prior4h_gt100_noisy_short", None),
    ]:
        if label == "prior4h_gt100_noisy_short":
            cal_v  = [(-r["net_2h"] - 2*FEE_BPS) for r in cal  if r["prior4h"] > 100 and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
            hold_v = [(-r["net_2h"] - 2*FEE_BPS) for r in hold if r["prior4h"] > 100 and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
        else:
            cal_v  = [r["net_2h"] for r in cal  if fn(r) and math.isfinite(r["net_2h"])]
            hold_v = [r["net_2h"] for r in hold if fn(r) and math.isfinite(r["net_2h"])]
        all_cal = [r["net_2h"] for r in cal if math.isfinite(r["net_2h"])]
        perm = permtest(cal_v, all_cal, N_PERM, SEED)
        result[label] = {
            "cal": qs(cal_v), "hold": qs(hold_v), "perm": perm,
        }
    return result


# ---------------------------------------------------------------------------
# G — Cascade sequence counter
# ---------------------------------------------------------------------------

def test_g_sequence(cal, hold):
    """
    n_prior2h: how many ETH SELL cascades in prior 2 hours.
    0 = fresh start (first cascade in quiet period)
    1-2 = mid-sequence
    3+ = deep cluster
    """
    result = {}
    for count_label, fn in [
        ("first_in_cluster",    lambda r: r["n_prior2h"] == 0),
        ("second_n_prior2h_1",  lambda r: r["n_prior2h"] == 1),
        ("mid_n_prior2h_2",     lambda r: r["n_prior2h"] == 2),
        ("deep_n_prior2h_3plus",lambda r: r["n_prior2h"] >= 3),
        ("any_prior_cascade",   lambda r: r["n_prior2h"] >= 1),
    ]:
        for gate, gfn in [
            ("all",     fn),
            ("silence", lambda r, f=fn: f(r) and r["silence30"]),
            ("noisy_short", lambda r, f=fn: f(r) and not r["silence30"] and not r["is_bull"]),
        ]:
            key = f"{count_label}_{gate}"
            if gate == "noisy_short":
                cal_v  = [(-r["net_2h"] - 2*FEE_BPS) for r in cal  if fn(r) and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
                hold_v = [(-r["net_2h"] - 2*FEE_BPS) for r in hold if fn(r) and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
            else:
                cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[key] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    # distribution of n_prior2h
    dist_cal  = defaultdict(int)
    dist_hold = defaultdict(int)
    for r in cal:  dist_cal[min(r["n_prior2h"], 5)]  += 1
    for r in hold: dist_hold[min(r["n_prior2h"], 5)] += 1
    result["n_prior2h_distribution"] = {
        "cal": dict(sorted(dist_cal.items())),
        "hold": dict(sorted(dist_hold.items())),
    }
    return result


# ---------------------------------------------------------------------------
# H — Time of day breakdown
# ---------------------------------------------------------------------------

def test_h_tod(cal, hold):
    result = {}
    for session in ["EU", "US", "ASIA"]:
        for gate, gfn in [
            ("all",     lambda r, s=session: r["session"] == s),
            ("silence", lambda r, s=session: r["session"] == s and r["silence30"]),
            ("noisy_short", lambda r, s=session: r["session"] == s and not r["silence30"] and not r["is_bull"]),
        ]:
            key = f"{session}_{gate}"
            if gate == "noisy_short":
                cal_v  = [(-r["net_2h"] - 2*FEE_BPS) for r in cal  if r["session"] == session and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
                hold_v = [(-r["net_2h"] - 2*FEE_BPS) for r in hold if r["session"] == session and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
            else:
                cal_v  = [r["net_2h"] for r in cal  if gfn(r) and math.isfinite(r["net_2h"])]
                hold_v = [r["net_2h"] for r in hold if gfn(r) and math.isfinite(r["net_2h"])]
            result[key] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    # hourly breakdown (hold only, for clarity)
    by_hour = {}
    for h in range(24):
        sil_v  = [r["net_2h"] for r in hold if r["hour_utc"] == h and r["silence30"]       and math.isfinite(r["net_2h"])]
        nois_v = [(-r["net_2h"] - 2*FEE_BPS) for r in hold if r["hour_utc"] == h and not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
        by_hour[str(h).zfill(2)] = {"sil_n": len(sil_v), "sil_win": r3(sum(1 for v in sil_v if v>0)/len(sil_v)) if sil_v else None,
                                     "nois_n": len(nois_v), "nois_win": r3(sum(1 for v in nois_v if v>0)/len(nois_v)) if nois_v else None}
    result["hourly_holdout"] = by_hour
    return result


# ---------------------------------------------------------------------------
# I — Tail event audit (losers)
# ---------------------------------------------------------------------------

def test_i_tail_audit(cal, hold):
    """
    Characterize the BAD trades: what do the losers look like?
    Silence gate: worst 10% events. What's their vdepth, btc4h, sync_k, hour?
    noisy SHORT: worst 10% events.
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        # Silence gate tail
        sil_events = [(r, r["net_2h"]) for r in rows if r["silence30"] and math.isfinite(r["net_2h"])]
        sil_events.sort(key=lambda x: x[1])
        n_tail = max(1, len(sil_events) // 10)
        sil_losers = [r for r, v in sil_events[:n_tail]]
        sil_winners = [r for r, v in sil_events[-n_tail:]]

        def feature_profile(events):
            if not events: return {}
            return {
                "n": len(events),
                "avg_vdepth": r1(sum(e["vdepth"] for e in events)/len(events)),
                "avg_btc4h":  r1(sum(e["btc4h"]  for e in events)/len(events)),
                "avg_sync_k": r1(sum(e["sync_k"]  for e in events)/len(events)),
                "avg_prior4h":r1(sum(e["prior4h"] for e in events)/len(events)),
                "avg_book_imb":r1(sum(e["book_imb"] for e in events)/len(events)),
                "avg_thresh":  r1(sum(e["thresh"]  for e in events)/len(events)),
                "session_dist": {s: sum(1 for e in events if e["session"]==s) for s in ["EU","US","ASIA"]},
                "pct_btc_lead": r3(sum(1 for e in events if e["btc_lead"])/len(events)),
            }

        # noisy SHORT tail
        noisy_events = [(r, -r["net_2h"] - 2*FEE_BPS) for r in rows
                        if not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
        noisy_events.sort(key=lambda x: x[1])
        n_tail_s = max(1, len(noisy_events) // 10)
        noisy_losers  = [r for r, v in noisy_events[:n_tail_s]]
        noisy_winners = [r for r, v in noisy_events[-n_tail_s:]]

        result[split_lbl] = {
            "silence_losers_profile":  feature_profile(sil_losers),
            "silence_winners_profile": feature_profile(sil_winners),
            "noisy_short_losers_profile":  feature_profile(noisy_losers),
            "noisy_short_winners_profile": feature_profile(noisy_winners),
        }
    return result


# ---------------------------------------------------------------------------
# J — Frequency & Kelly sizing
# ---------------------------------------------------------------------------

def test_j_kelly(cal, hold):
    """
    Kelly fraction = (WR * maxW - LR * maxL) / maxW  ... simplified edge/odds Kelly.
    More useful: half-Kelly margin allocation per signal.
    Monthly frequency of each signal.
    """
    def kelly_stats(vals):
        if not vals: return {}
        w = [v for v in vals if v > 0]
        l = [v for v in vals if v <= 0]
        wr = len(w)/len(vals)
        lr = 1 - wr
        avg_w = sum(w)/len(w) if w else 0
        avg_l = abs(sum(l)/len(l)) if l else 0
        b = avg_w / avg_l if avg_l > 0 else float("inf")
        kelly_f = (wr * b - lr) / b if b > 0 else 0
        half_kelly = kelly_f / 2
        return {
            "n": len(vals),
            "wr": r3(wr),
            "avg_win_bps": r1(avg_w),
            "avg_loss_bps": r1(-avg_l),
            "win_loss_ratio": r3(b),
            "kelly_fraction": r3(kelly_f),
            "half_kelly_fraction": r3(half_kelly),
            "edge_bps": r1(sum(vals)/len(vals)),
        }

    # date range for frequency calc
    def days_span(rows):
        ts_list = [int(r["signal_ts_ms"]) for r in rows]
        if len(ts_list) < 2: return 1
        return (max(ts_list) - min(ts_list)) / 86400_000

    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        span = days_span(rows)
        months = max(span / 30.44, 0.1)

        sil_v   = [r["net_2h"] for r in rows if r["silence30"]       and math.isfinite(r["net_2h"])]
        noisy_v = [(-r["net_2h"]-2*FEE_BPS) for r in rows if not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
        sil_live_v = [r["net_2h"] for r in rows if r["silence30"] and r["is_live"] and math.isfinite(r["net_2h"])]
        sil_hs_v   = [r["net_2h"] for r in rows if r["silence30"] and r["sync_k"]>=200_000 and math.isfinite(r["net_2h"])]

        def freq(n): return r3(n / months)

        result[split_lbl] = {
            "span_days": r1(span),
            "span_months": r3(months),
            "silence_LONG": {
                "monthly_trades": freq(len(sil_v)),
                **kelly_stats(sil_v),
            },
            "noisy_short": {
                "monthly_trades": freq(len(noisy_v)),
                **kelly_stats(noisy_v),
            },
            "silence_live_200K": {
                "monthly_trades": freq(len(sil_live_v)),
                **kelly_stats(sil_live_v),
            },
            "silence_highsync": {
                "monthly_trades": freq(len(sil_hs_v)),
                **kelly_stats(sil_hs_v),
            },
        }
    return result


# ---------------------------------------------------------------------------
# K — BTC-led cascade (cross-asset lead)
# ---------------------------------------------------------------------------

def test_k_btc_lead(cal, hold):
    """
    When BTC has a large SELL cascade in the 30 minutes before an ETH cascade,
    does the ETH silence gate work differently?
    BTC lead defined as max(BTC SELL notional in prior 30min) >= 2M.
    """
    result = {}
    for split_lbl, rows in [("cal", cal), ("hold", hold)]:
        lead    = [r for r in rows if r["btc_lead"]]
        no_lead = [r for r in rows if not r["btc_lead"]]
        for subset, label in [(lead, "btc_lead"), (no_lead, "no_btc_lead")]:
            for gate, gfn in [
                ("all",     lambda r: True),
                ("silence", lambda r: r["silence30"]),
                ("noisy",   lambda r: not r["silence30"]),
            ]:
                key = f"{label}_{gate}_{split_lbl}"
                vals = [r["net_2h"] for r in subset if gfn(r) and math.isfinite(r["net_2h"])]
                result[key] = qs(vals)
    return result


# ---------------------------------------------------------------------------
# L — Weekly holdout stability
# ---------------------------------------------------------------------------

def test_l_weekly(cal, hold):
    from datetime import timedelta
    result = {}
    # group hold events by ISO week
    weeks = defaultdict(list)
    for r in hold:
        ts = int(r["signal_ts_ms"])
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        week_key = dt.strftime("%Y-W%W")
        weeks[week_key].append(r)

    for week, rows in sorted(weeks.items()):
        sil_v  = [r["net_2h"] for r in rows if r["silence30"]       and math.isfinite(r["net_2h"])]
        nois_v = [(-r["net_2h"]-2*FEE_BPS) for r in rows if not r["silence30"] and not r["is_bull"] and math.isfinite(r["net_2h"])]
        avg_sync = r1(sum(r["sync_k"] for r in rows)/len(rows)) if rows else None
        sil_rate = r3(sum(1 for r in rows if r["silence30"]) / len(rows)) if rows else None
        result[week] = {
            "n_events": len(rows),
            "avg_sync_k": avg_sync,
            "silence_rate": sil_rate,
            "silence_LONG": qs(sil_v),
            "noisy_SHORT": qs(nois_v),
        }
    return result


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def fv(d, key, default="-"):
    v = d.get(key, default)
    return str(v) if v is not None else "-"

def render_md(res):
    sp = res["split"]
    lines = [
        "# S34 Deep Expansion Suite",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        f"Cal: {sp['cal_n']} ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} ({sp['hold_start']} to {sp['hold_end']})",
        "",
        "Baseline reference (from comprehensive final suite):",
        "- Silence LONG hold: WR=70.1%, T3R=+7733",
        "- Silence+sync>=200K LONG hold: WR=83.1%, T3R=+4298",
        "- noisy_NOT_bull SHORT hold: WR=54.9%, T3R=+11360",
        "",
    ]

    # A
    ta = res["test_a"]
    lines += ["## A. Signal Portfolio (LONG + SHORT Combined)", ""]
    lines += ["| Split | Signal | N | T3R | med | win | coverage |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for sl in ("cal", "hold"):
        d = ta[sl]
        for sig, key in [("LONG only", "long_only"), ("SHORT only", "short_only"), ("Combined", "combined")]:
            s = d[key]
            lines.append(f"| {sl} | {sig} | {s['n']} | {s.get('t3r','-')} | {s['med']} | {s['win']} |"
                         f" {d.get('coverage','') if sig=='Combined' else ''} |")
        lines.append(f"| {sl} | Untraded (bull+noisy) | {d['n_untraded']} | - | - | - | - |")
    lines.append("")

    # B
    tb = res["test_b"]
    lines += ["## B. Flip Strategy (SHORT first, flip to LONG on silence)", ""]
    lines += ["| Split | Strategy | N | T3R | med | win |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for sl in ("cal", "hold"):
        d = tb[sl]
        for sig, key in [("Always SHORT 2h", "always_short_2h"),
                          ("Baseline (sil=LONG, noisy=SHORT)", "baseline_split"),
                          ("Flip (SHORT->LONG on silence)", "flip_strategy")]:
            s = d[key]
            lines.append(f"| {sl} | {sig} | {s['n']} | {s.get('t3r','-')} | {s['med']} | {s['win']} |")
    lines += ["", f"*Note*: {tb['cal']['note']}", ""]

    # C
    tc = res["test_c"]
    p = tc["vdepth_percentiles_cal"]
    lines += ["## C. vdepth Gate (Overshoot Depth)", "",
              f"Cal vdepth percentiles: p25={p['p25']}bps  p50={p['p50']}bps  p75={p['p75']}bps", ""]
    lines += ["| Segment | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in tc["bins"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # D
    td = res["test_d"]
    lines += ["## D. BTC Context (btc4h_bps)", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in td.items():
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # E
    te = res["test_e"]
    p2 = te.get("book_imb_percentiles_cal", {})
    lines += ["## E. Book Imbalance Gate", "",
              f"Cal book_imb percentiles: p25={p2.get('p25')}  p75={p2.get('p75')}", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in te.items():
        if key == "book_imb_percentiles_cal": continue
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # F
    tf = res["test_f"]
    lines += ["## F. prior4h Permutation Null", ""]
    lines += ["| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Perm verdict |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key, d in tf.items():
        c = d["cal"]; h = d["hold"]; p = d["perm"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} |"
                     f" {h['n']} | {h.get('t3r','-')} | {h['win']} | {p['p_right']} | **{p['verdict']}** |")
    lines.append("")

    # G
    tg = res["test_g"]
    lines += ["## G. Cascade Sequence Counter", ""]
    dist = tg.get("n_prior2h_distribution", {})
    lines += [f"Cal distribution: {dist.get('cal')}", f"Hold distribution: {dist.get('hold')}", ""]
    lines += ["| Sequence position | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in tg.items():
        if key == "n_prior2h_distribution": continue
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines.append("")

    # H
    th = res["test_h"]
    lines += ["## H. Time of Day (Session Breakdown)", ""]
    lines += ["| Session/Key | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, d in th.items():
        if key == "hourly_holdout": continue
        c = d["cal"]; h = d["hold"]
        lines.append(f"| {key} | {c['n']} | {c.get('t3r','-')} | {c['win']} | {h['n']} | {h.get('t3r','-')} | {h['win']} |")
    lines += ["", "### Hourly breakdown (hold only)", "| Hour UTC | Sil N | Sil WR | Noisy Short N | Noisy WR |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for h_str, d in th["hourly_holdout"].items():
        lines.append(f"| {h_str}:00 | {d['sil_n']} | {d['sil_win'] or '-'} | {d['nois_n']} | {d['nois_win'] or '-'} |")
    lines.append("")

    # I
    ti = res["test_i"]
    lines += ["## I. Tail Event Audit (Losers vs Winners)", ""]
    for split_lbl in ("cal", "hold"):
        d = ti[split_lbl]
        lines += [f"### {split_lbl}",
                  "",
                  "**Silence LONG Losers vs Winners** (bottom/top 10%):"]
        lp = d["silence_losers_profile"]
        wp = d["silence_winners_profile"]
        if lp:
            lines += [
                f"| Feature | Losers | Winners |",
                f"| --- | ---: | ---: |",
                f"| avg vdepth | {lp.get('avg_vdepth')} | {wp.get('avg_vdepth')} |",
                f"| avg btc4h | {lp.get('avg_btc4h')} | {wp.get('avg_btc4h')} |",
                f"| avg sync_k | {lp.get('avg_sync_k')} | {wp.get('avg_sync_k')} |",
                f"| avg prior4h | {lp.get('avg_prior4h')} | {wp.get('avg_prior4h')} |",
                f"| avg book_imb | {lp.get('avg_book_imb')} | {wp.get('avg_book_imb')} |",
                f"| avg thresh | {lp.get('avg_thresh')} | {wp.get('avg_thresh')} |",
                f"| pct_btc_lead | {lp.get('pct_btc_lead')} | {wp.get('pct_btc_lead')} |",
                f"| sessions | {lp.get('session_dist')} | {wp.get('session_dist')} |",
                "",
            ]
        nl = d["noisy_short_losers_profile"]
        nw = d["noisy_short_winners_profile"]
        lines += ["**noisy SHORT Losers vs Winners** (bottom/top 10%):"]
        if nl:
            lines += [
                f"| Feature | Losers | Winners |",
                f"| --- | ---: | ---: |",
                f"| avg vdepth | {nl.get('avg_vdepth')} | {nw.get('avg_vdepth')} |",
                f"| avg btc4h | {nl.get('avg_btc4h')} | {nw.get('avg_btc4h')} |",
                f"| avg sync_k | {nl.get('avg_sync_k')} | {nw.get('avg_sync_k')} |",
                f"| avg prior4h | {nl.get('avg_prior4h')} | {nw.get('avg_prior4h')} |",
                f"| pct_btc_lead | {nl.get('pct_btc_lead')} | {nw.get('pct_btc_lead')} |",
                f"| sessions | {nl.get('session_dist')} | {nw.get('session_dist')} |",
                "",
            ]
    lines.append("")

    # J
    tj = res["test_j"]
    lines += ["## J. Frequency & Kelly Sizing", ""]
    for split_lbl in ("cal", "hold"):
        d = tj[split_lbl]
        lines += [f"### {split_lbl} ({d['span_days']} days, {d['span_months']} months)",
                  "| Signal | Monthly trades | Edge bps | WR | avg_win | avg_loss | W/L ratio | Kelly | Half-Kelly |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for sig in ["silence_LONG", "noisy_short", "silence_live_200K", "silence_highsync"]:
            s = d[sig]
            lines.append(f"| {sig} | {s.get('monthly_trades','?')} | {s.get('edge_bps','?')} |"
                         f" {s.get('wr','?')} | {s.get('avg_win_bps','?')} | {s.get('avg_loss_bps','?')} |"
                         f" {s.get('win_loss_ratio','?')} | {s.get('kelly_fraction','?')} | {s.get('half_kelly_fraction','?')} |")
        lines.append("")

    # K
    tk = res["test_k"]
    lines += ["## K. BTC-Led Cascade (Cross-Asset Lead)", ""]
    lines += ["| Key | N | T3R | med | win |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key, d in tk.items():
        lines.append(f"| {key} | {d['n']} | {d.get('t3r','-')} | {d['med']} | {d['win']} |")
    lines.append("")

    # L
    tl = res["test_l"]
    lines += ["## L. Weekly Holdout Stability", ""]
    lines += ["| Week | N events | Avg sync_k | Sil rate | Sil T3R | Sil win | SHORT T3R | SHORT win |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for week, d in tl.items():
        s = d["silence_LONG"]; n = d["noisy_SHORT"]
        lines.append(f"| {week} | {d['n_events']} | {d['avg_sync_k']} | {d['silence_rate']} |"
                     f" {s.get('t3r','-')} | {s['win']} | {n.get('t3r','-')} | {n['win']} |")
    lines.append("")

    lines += ["---",
              "## Open Questions for Next Session",
              "",
              "1. **vdepth optimal cut**: does vdepth>30bps + silence form a 4th validated signal?",
              "2. **BTC-led cascade permutation null**: test K shows directional pattern — run formal perm test",
              "3. **prior4h>100 + silence perm null (hold)**: PASS in cal — does hold confirm?",
              "4. **Session-specific signal**: if EU silence shows strongest WR, can we trade EU-only?",
              "5. **Flip strategy cost sensitivity**: how much drift reduction needed for flip to be worth it?",
              "6. **Combined portfolio perm null**: silence LONG + noisy SHORT as a SINGLE combined strategy",
              "7. **Cascade depth + sync**: deep cascade (vdepth>40) in high sync + silence = ultra signal?",
              "8. **Next-day fade**: does silence gate work on H8/H12 (overnight hold)?",
              "9. **Sequence signal**: first cascade after 4h quiet period — stronger silence predictor?",
              "10. **Bid depth absolute**: high bid_depth_usd at cascade time — does it predict silence?",
              "",
              "RESEARCH_ONLY. No live changes without explicit operator sign-off.",
              ]
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

    print("Loading DB (liquidations only)...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_sell_ts, eth_sell_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_sell_ts, btc_sell_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_sell_ts, sol_sell_not = load_liq(conn, "SOLUSDT", "SELL")
    print("Annotating...")
    cal  = annotate(cal_raw,  eth_sell_ts, eth_sell_not, btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)
    hold = annotate(hold_raw, eth_sell_ts, eth_sell_not, btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)

    print("A: portfolio...")
    ta = test_a_portfolio(cal, hold)
    print("B: flip strategy...")
    tb = test_b_flip(cal, hold)
    print("C: vdepth gate...")
    tc = test_c_vdepth(cal, hold)
    print("D: BTC context...")
    td = test_d_btc_context(cal, hold)
    print("E: book imbalance...")
    te = test_e_book_imbalance(cal, hold)
    print("F: prior4h perm...")
    tf = test_f_prior4h(cal, hold)
    print("G: cascade sequence...")
    tg = test_g_sequence(cal, hold)
    print("H: time of day...")
    th = test_h_tod(cal, hold)
    print("I: tail audit...")
    ti = test_i_tail_audit(cal, hold)
    print("J: kelly sizing...")
    tj = test_j_kelly(cal, hold)
    print("K: BTC lead...")
    tk = test_k_btc_lead(cal, hold)
    print("L: weekly stability...")
    tl = test_l_weekly(cal, hold)

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
        "test_i": ti, "test_j": tj, "test_k": tk, "test_l": tl,
    }
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
