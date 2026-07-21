"""S34 State Machine Deep Research — 10 questions before live.

1.  Stop loss for SHORT — optimal level, tail protection
2.  BTC cascade threshold sensitivity — is 500K optimal?
3.  Which BTC cascade to use — first / largest / last
4.  ETH cascade notional sweet spot — 200-500K vs 500K+ vs 1M+
5.  Sequential cascade exhaustion — n2h effect on WR
6.  Silence window sensitivity — 20/30/45/60min
7.  Session breakdown — Asia/Europe/US WR for LONG and SHORT
8.  BTC regime — btc4h>0 vs btc4h<0 for NEITHER SHORT
9.  P&L distribution and max drawdown (holdout)
10. NOISY recovery — can score/filter rescue the 185 excluded events?
"""
from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
DEFAULT_DB  = ROOT / "data" / "microstructure.db"
OUT_JSON    = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_DEEP.json"
OUT_MD      = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_DEEP.md"

LIVE_THRESH    = 200_000.0
SIL_LO_MS      = 60_000
SIL_HI_MS      = 30 * 60_000
PROP_THRESH    = 50_000.0
BTC_THRESH     = 500_000.0
ULTRA_MS       = 60_000
FEE_BPS        = 5.0
SYNC_WIN_MS    = 10 * 60_000
HOLDOUT_FRAC   = 0.30
CASCADE_WIN_MS = 5 * 60_000   # 5-min window for running notional


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def load_events():
    rows = []
    with NAV_EVENTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def load_liq(conn, sym, side):
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (sym, side),
    ).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]


def win_cnt(ts, v, lo, hi, thr):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(1 for i in range(a, b) if v[i] >= thr)


def win_sum(ts, v, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(v[i] for i in range(a, b))


def win_max(ts, v, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return max((v[i] for i in range(a, b)), default=0.0)


def first_above(ts, v, lo, hi, thr):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    for i in range(a, b):
        if v[i] >= thr:
            return int(ts[i]), float(v[i])
    return None, None


def all_above(ts, v, lo, hi, thr):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return [(int(ts[i]), float(v[i])) for i in range(a, b) if v[i] >= thr]


def mark_before(mk_ts, mk_px, t):
    idx = bisect.bisect_right(mk_ts, t) - 1
    return mk_px[idx] if idx >= 0 else None


def max_in(mk_ts, mk_px, lo, hi):
    a = bisect.bisect_left(mk_ts, lo)
    b = bisect.bisect_right(mk_ts, hi)
    return max((mk_px[i] for i in range(a, b)), default=None)


def min_in(mk_ts, mk_px, lo, hi):
    a = bisect.bisect_left(mk_ts, lo)
    b = bisect.bisect_right(mk_ts, hi)
    return min((mk_px[i] for i in range(a, b)), default=None)


def wr(vals):
    if not vals:
        return None
    return round(sum(1 for v in vals if v > 0) / len(vals), 3)


def stats(vals):
    if not vals:
        return {"n": 0, "wr": None, "mean": None, "median": None, "t3r": None}
    sv = sorted(vals)
    t3r = sum(sv[:-3]) if len(sv) > 3 else sum(sv)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "mean": round(mean(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(t3r, 0),
    }


def classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not,
             sil_hi_ms=SIL_HI_MS, btc_thr=BTC_THRESH):
    ts   = int(row["signal_ts_ms"])
    thr  = float(row.get("threshold_usd") or 0)
    net2 = float(row.get("net_2h_bps") or "nan")
    net4v = row.get("net_4h_bps")
    net4 = float(net4v) if net4v is not None else net2
    tags = row.get("tags") or []

    if not math.isfinite(net2) or thr < LIVE_THRESH:
        return None

    n_prop     = win_cnt(eth_ts, eth_not, ts + SIL_LO_MS, ts + sil_hi_ms, PROP_THRESH)
    sil_eth    = n_prop == 0
    btc_1st_ts, _ = first_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + sil_hi_ms, btc_thr)
    sil_btc    = btc_1st_ts is None
    bull       = "BULL_PULLBACK" in tags

    if sil_eth:
        state = "SILENCE"
    elif not sil_btc:
        state = "NEITHER"
    else:
        state = "NOISY"
    if bull:
        state += "_BULL"

    b4h     = float(row.get("btc4h_bps") or 0)
    vd      = float(row.get("vdepth_bps") or 0)
    bid     = float(row.get("bid_depth_usd") or 0)
    ts_dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour    = ts_dt.hour
    weekday = ts_dt.weekday() < 5
    sess_us = 13 <= hour < 21
    sync_k  = (win_sum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts) +
               win_sum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts))
    n2h     = win_cnt(eth_ts, eth_not, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
    running_notional = win_sum(eth_ts, eth_not, ts - CASCADE_WIN_MS, ts)

    score = sum([int(sil_eth), int(n2h >= 3), int(b4h < 0),
                 int(vd >= 30), int(sess_us), int(sync_k >= 200_000)])

    session = ("ASIA" if 0 <= hour < 7 else
               "EUROPE" if 7 <= hour < 13 else
               "US" if 13 <= hour < 21 else "OFF")

    return {
        "ts": ts, "net2": net2, "net4": net4,
        "state": state, "sil_eth": sil_eth, "sil_btc": sil_btc, "bull": bull,
        "btc_1st_ts": btc_1st_ts,
        "score": score, "b4h": b4h, "vd": vd, "bid": bid,
        "n2h": n2h, "sync_k": sync_k, "weekday": weekday,
        "sess_us": sess_us, "session": session, "hour": hour,
        "running_notional": running_notional,
    }


def main() -> int:
    events = load_events()
    print(f"Loaded {len(events)} NAV_EVENTS")

    print("Loading liquidation arrays...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")

        classed = []
        for row in events:
            c = classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not)
            if c:
                classed.append(c)
        n_cal = int(len(classed) * (1 - HOLDOUT_FRAC))
        for i, c in enumerate(classed):
            c["is_hold"] = i >= n_cal
        print(f"Classified: {len(classed)} (cal={n_cal}, hold={len(classed)-n_cal})")

        sil  = [c for c in classed if c["state"] == "SILENCE"]
        nei  = [c for c in classed if c["state"] == "NEITHER"]
        nois = [c for c in classed if c["state"] == "NOISY"]

        # Mark prices (loaded once, bounded to event range)
        all_ts = [c["ts"] for c in classed]
        mk_lo = min(all_ts) - 60_000
        mk_hi = max(all_ts) + 4 * 3600_000 + 60_000
        print(f"Loading ETH mark prices [{iso(mk_lo)} -> {iso(mk_hi)}]...")
        mk_rows = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT'"
            " AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms", (mk_lo, mk_hi)
        ).fetchall()
        mk_ts = [int(r[0]) for r in mk_rows]
        mk_px = [float(r[1]) for r in mk_rows]
        print(f"  Loaded {len(mk_ts):,} mark price rows")

    out = {"generated_utc": utc_now(), "fee_bps": FEE_BPS}

    # ── 1. Stop loss for SHORT ────────────────────────────────────────────────
    print("\n=== 1. Stop Loss for SHORT ===")
    stop_levels = [50, 75, 100, 125, 150, 175, 200, 250]
    nei_hold = [c for c in nei if c["is_hold"]]

    stop_results = {}
    for sl in stop_levels:
        nets = []
        triggered = 0
        for c in nei_hold:
            t0 = c["ts"]
            px0 = mark_before(mk_ts, mk_px, t0)
            if not px0 or px0 <= 0:
                nets.append(-c["net2"] - FEE_BPS)
                continue
            # Worst adverse for SHORT = max price rise in 2h
            hi = max_in(mk_ts, mk_px, t0, t0 + 2 * 3600_000)
            if hi is None:
                nets.append(-c["net2"] - FEE_BPS)
                continue
            max_adv_bps = (hi - px0) / px0 * 10_000.0
            if max_adv_bps >= sl:
                triggered += 1
                nets.append(-sl - FEE_BPS)   # stopped out at nominal
            else:
                nets.append(-c["net2"] - FEE_BPS)
        s = stats(nets)
        s["stop_trigger_rate"] = round(triggered / len(nei_hold), 3) if nei_hold else 0
        stop_results[str(sl)] = s
        print(f"  SL={sl:>3}bps: WR={s['wr']:.1%} mean={s['mean']:+.1f} "
              f"T3R={s['t3r']:+.0f} triggered={s['stop_trigger_rate']:.1%}")
    out["Q1_stop_loss_short"] = stop_results

    # Symmetric: also check SILENCE LONG stop (reference)
    sil_hold = [c for c in sil if c["is_hold"]]
    sil_stop_results = {}
    for sl in stop_levels:
        nets = []
        triggered = 0
        for c in sil_hold:
            t0 = c["ts"]
            px0 = mark_before(mk_ts, mk_px, t0)
            if not px0 or px0 <= 0:
                nets.append(c["net4"] - FEE_BPS)
                continue
            lo = min_in(mk_ts, mk_px, t0, t0 + 4 * 3600_000)
            if lo is None:
                nets.append(c["net4"] - FEE_BPS)
                continue
            max_adv_bps = (px0 - lo) / px0 * 10_000.0
            if max_adv_bps >= sl:
                triggered += 1
                nets.append(-sl - FEE_BPS)
            else:
                nets.append(c["net4"] - FEE_BPS)
        s = stats(nets)
        s["stop_trigger_rate"] = round(triggered / len(sil_hold), 3) if sil_hold else 0
        sil_stop_results[str(sl)] = s
    out["Q1_stop_loss_long_reference"] = sil_stop_results

    # ── 2. BTC threshold sensitivity ─────────────────────────────────────────
    print("\n=== 2. BTC Threshold Sensitivity ===")
    btc_thresholds = [200_000, 300_000, 500_000, 750_000, 1_000_000]
    btc_thr_results = {}
    print(f"{'BTC_THR':<12} {'N_all':>7} {'WR_all':>8} {'N_hold':>8} {'WR_hold':>9} {'T3R_hold':>10}")
    for btc_thr in btc_thresholds:
        # Reclassify only NEITHER condition
        nei_thr = []
        for row in events:
            c = classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not, btc_thr=btc_thr)
            if c and c["state"] == "NEITHER":
                nei_thr.append(c)
        n_cal_t = int(len(nei_thr) * (1 - HOLDOUT_FRAC))
        nei_thr_hold = [c for i, c in enumerate(sorted(nei_thr, key=lambda x: x["ts"])) if i >= n_cal_t]
        all_v  = [-c["net2"] - FEE_BPS for c in nei_thr]
        hold_v = [-c["net2"] - FEE_BPS for c in nei_thr_hold]
        s = stats(hold_v)
        print(f"{btc_thr:<12,} {len(nei_thr):>7} {wr(all_v) or 0:>7.1%} "
              f"{s['n']:>8} {s['wr'] or 0:>8.1%}  {s['t3r'] or 0:>+9.0f}")
        btc_thr_results[str(btc_thr)] = {"n_all": len(nei_thr), **s}
    out["Q2_btc_threshold"] = btc_thr_results

    # ── 3. Which BTC cascade — first / largest / last ────────────────────────
    print("\n=== 3. Which BTC Cascade to Use ===")
    entry_comparison = {"first": [], "largest": [], "last": []}
    for c in nei:
        ts   = c["ts"]
        candidates = all_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + SIL_HI_MS, BTC_THRESH)
        if not candidates:
            continue
        first_t  = candidates[0][0]
        last_t   = candidates[-1][0]
        largest_t = max(candidates, key=lambda x: x[1])[0]

        px_entry = mark_before(mk_ts, mk_px, ts)  # ETH cascade entry price (same for all)
        for label, t_entry in [("first", first_t), ("largest", largest_t), ("last", last_t)]:
            px_t = mark_before(mk_ts, mk_px, t_entry)
            if px_entry and px_t and px_entry > 0:
                slippage_bps = (px_t - px_entry) / px_entry * 10_000.0
            else:
                slippage_bps = 0.0
            # P&L for SHORT: depends on exit price 2h from THIS entry
            # Approximate: use net_2h adjusted for entry timing shift
            # net_2h_bps is measured from ETH cascade; if BTC entry is T_delta later,
            # actual SHORT P&L ≈ -(price_at_ETH+2h - price_at_BTC_entry)
            # We don't have that exactly; use slippage as entry cost difference
            net_adj = -c["net2"] - FEE_BPS + slippage_bps  # worse entry = slippage subtracted from SHORT
            entry_comparison[label].append({
                "slippage_bps": round(slippage_bps, 1),
                "net_approx": round(net_adj, 1),
                "is_hold": c["is_hold"],
            })

    print(f"{'Entry':>10} {'N':>5} {'Avg_slip':>10} {'WR_approx':>11} {'Mean_net':>10}")
    entry_results = {}
    for label in ["first", "largest", "last"]:
        rows_e = entry_comparison[label]
        hold_e = [r for r in rows_e if r["is_hold"]]
        slips  = [r["slippage_bps"] for r in rows_e]
        nets_h = [r["net_approx"] for r in hold_e]
        print(f"{label:>10} {len(rows_e):>5} {mean(slips):>+9.1f}bps "
              f"{wr(nets_h) or 0:>10.1%}  {mean(nets_h):>+9.1f}" if nets_h else
              f"{label:>10} {len(rows_e):>5} {mean(slips):>+9.1f}bps N/A")
        entry_results[label] = {
            "n": len(rows_e), "n_hold": len(hold_e),
            "avg_slippage_bps": round(mean(slips), 1) if slips else None,
            "hold_wr": wr(nets_h),
            "hold_mean": round(mean(nets_h), 1) if nets_h else None,
        }
    out["Q3_btc_entry_choice"] = entry_results

    # ── 4. ETH notional sweet spot ───────────────────────────────────────────
    print("\n=== 4. ETH Cascade Notional Sweet Spot ===")
    buckets = [
        ("200K-500K",  200_000,  500_000),
        ("500K-1M",    500_000, 1_000_000),
        ("1M-2M",    1_000_000, 2_000_000),
        ("2M+",      2_000_000, 1e15),
    ]
    notional_results = {}
    print(f"{'Bucket':<14} {'N_sil':>7} {'WR_sil':>8} {'N_nei':>7} {'WR_nei':>8}")
    for label, lo, hi in buckets:
        sil_b  = [c for c in sil  if lo <= c["running_notional"] < hi]
        nei_b  = [c for c in nei  if lo <= c["running_notional"] < hi]
        sil_bh = [c for c in sil_b if c["is_hold"]]
        nei_bh = [c for c in nei_b if c["is_hold"]]
        sil_v  = [c["net4"] - FEE_BPS for c in sil_bh]
        nei_v  = [-c["net2"] - FEE_BPS for c in nei_bh]
        wr_s   = wr(sil_v)
        wr_n   = wr(nei_v)
        print(f"{label:<14} {len(sil_b):>7} {(wr_s or 0):>7.1%} {len(nei_b):>7} {(wr_n or 0):>7.1%}")
        notional_results[label] = {
            "sil_n_all": len(sil_b), "sil_n_hold": len(sil_bh), "sil_hold_wr": wr_s,
            "nei_n_all": len(nei_b), "nei_n_hold": len(nei_bh), "nei_hold_wr": wr_n,
        }
    out["Q4_notional_buckets"] = notional_results

    # ── 5. Sequential cascade exhaustion (n2h) ───────────────────────────────
    print("\n=== 5. n2h (Prior 2h Cascade Count) Effect ===")
    n2h_results = {}
    print(f"{'n2h':>5} {'N_sil':>7} {'WR_sil_h':>10} {'N_nei':>7} {'WR_nei_h':>10}")
    for n2h_val in range(0, 8):
        sil_b  = [c for c in sil  if c["n2h"] == n2h_val]
        nei_b  = [c for c in nei  if c["n2h"] == n2h_val]
        sil_v  = [c["net4"] - FEE_BPS for c in sil_b if c["is_hold"]]
        nei_v  = [-c["net2"] - FEE_BPS for c in nei_b if c["is_hold"]]
        print(f"{n2h_val:>5} {len(sil_b):>7} {(wr(sil_v) or 0):>9.1%} "
              f"{len(nei_b):>7} {(wr(nei_v) or 0):>9.1%}")
        n2h_results[str(n2h_val)] = {
            "sil_n": len(sil_b), "sil_hold_wr": wr(sil_v),
            "nei_n": len(nei_b), "nei_hold_wr": wr(nei_v),
        }
    # Cumulative n2h>=
    print(f"\n{'n2h>=':>6} {'N_nei':>7} {'WR_nei_h':>10} {'T3R_nei_h':>12}")
    n2h_cum = {}
    for thr in range(0, 6):
        nei_b = [c for c in nei if c["n2h"] >= thr]
        nei_v = [-c["net2"] - FEE_BPS for c in nei_b if c["is_hold"]]
        s = stats(nei_v)
        print(f"{thr:>6} {len(nei_b):>7} {(s['wr'] or 0):>9.1%}  {(s['t3r'] or 0):>+11.0f}")
        n2h_cum[f">={thr}"] = {"nei_n_all": len(nei_b), **s}
    out["Q5_n2h_effect"] = {"by_exact": n2h_results, "cumulative": n2h_cum}

    # ── 6. Silence window sensitivity ────────────────────────────────────────
    print("\n=== 6. Silence Window Sensitivity ===")
    sil_windows_min = [15, 20, 30, 45, 60]
    win_results = {}
    print(f"{'Window':>10} {'N_sil':>7} {'WR_sil_h':>10} {'N_nei':>7} {'WR_nei_h':>10}")
    for w_min in sil_windows_min:
        w_ms = w_min * 60_000
        sil_w, nei_w = [], []
        for row in events:
            c = classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not, sil_hi_ms=w_ms)
            if not c:
                continue
            if c["state"] == "SILENCE":
                sil_w.append(c)
            elif c["state"] == "NEITHER":
                nei_w.append(c)
        n_c = int(len(sil_w) * (1 - HOLDOUT_FRAC))
        sil_w.sort(key=lambda x: x["ts"])
        nei_w.sort(key=lambda x: x["ts"])
        sil_wh = [c for i, c in enumerate(sil_w) if i >= n_c]
        n_c2 = int(len(nei_w) * (1 - HOLDOUT_FRAC))
        nei_wh = [c for i, c in enumerate(nei_w) if i >= n_c2]
        sil_v = [c["net4"] - FEE_BPS for c in sil_wh]
        nei_v = [-c["net2"] - FEE_BPS for c in nei_wh]
        print(f"{w_min:>8}min {len(sil_w):>7} {(wr(sil_v) or 0):>9.1%} "
              f"{len(nei_w):>7} {(wr(nei_v) or 0):>9.1%}")
        win_results[f"{w_min}min"] = {
            "sil_n_all": len(sil_w), "sil_n_hold": len(sil_wh), "sil_hold_wr": wr(sil_v),
            "nei_n_all": len(nei_w), "nei_n_hold": len(nei_wh), "nei_hold_wr": wr(nei_v),
        }
    out["Q6_silence_window"] = win_results

    # ── 7. Session breakdown ─────────────────────────────────────────────────
    print("\n=== 7. Session Breakdown ===")
    sessions = ["ASIA", "EUROPE", "US", "OFF"]
    sess_results = {}
    print(f"{'Session':<10} {'N_sil':>7} {'WR_sil_h':>10} {'N_nei':>7} {'WR_nei_h':>10}")
    for sess in sessions:
        sil_s  = [c for c in sil  if c["session"] == sess]
        nei_s  = [c for c in nei  if c["session"] == sess]
        sil_v  = [c["net4"] - FEE_BPS for c in sil_s if c["is_hold"]]
        nei_v  = [-c["net2"] - FEE_BPS for c in nei_s if c["is_hold"]]
        print(f"{sess:<10} {len(sil_s):>7} {(wr(sil_v) or 0):>9.1%} "
              f"{len(nei_s):>7} {(wr(nei_v) or 0):>9.1%}")
        sess_results[sess] = {
            "sil_n_all": len(sil_s), "sil_n_hold": len(sil_v),
            "sil_hold_wr": wr(sil_v),
            "nei_n_all": len(nei_s), "nei_n_hold": len(nei_v),
            "nei_hold_wr": wr(nei_v),
        }
    out["Q7_session"] = sess_results

    # ── 8. BTC regime (btc4h > 0 vs < 0) ────────────────────────────────────
    print("\n=== 8. BTC Regime Effect on NEITHER SHORT ===")
    nei_btc_up   = [c for c in nei if c["b4h"] >= 0]
    nei_btc_down = [c for c in nei if c["b4h"] < 0]
    for label, evs in [("btc4h>=0 (UP)", nei_btc_up), ("btc4h<0  (DOWN)", nei_btc_down)]:
        hold_v = [-c["net2"] - FEE_BPS for c in evs if c["is_hold"]]
        s = stats(hold_v)
        print(f"  {label}: N_all={len(evs)} hold={s}")
    regime_out = {
        "btc_up":   {"n_all": len(nei_btc_up),
                     **stats([-c["net2"] - FEE_BPS for c in nei_btc_up if c["is_hold"]])},
        "btc_down": {"n_all": len(nei_btc_down),
                     **stats([-c["net2"] - FEE_BPS for c in nei_btc_down if c["is_hold"]])},
    }
    # Also cross: score>=3 by regime
    for label, evs in [("score3_btc_up", [c for c in nei if c["score"] >= 3 and c["b4h"] >= 0]),
                       ("score3_btc_dn", [c for c in nei if c["score"] >= 3 and c["b4h"] < 0])]:
        hold_v = [-c["net2"] - FEE_BPS for c in evs if c["is_hold"]]
        s = stats(hold_v)
        print(f"  {label}: N_all={len(evs)} hold={s}")
        regime_out[label] = {"n_all": len(evs), **s}
    out["Q8_btc_regime"] = regime_out

    # ── 9. P&L distribution and max drawdown ─────────────────────────────────
    print("\n=== 9. P&L Distribution and Max Drawdown ===")
    # Combined portfolio: SILENCE LONG + NEITHER(score>=3) SHORT, holdout only
    hold_portfolio = []
    for c in classed:
        if not c["is_hold"]:
            continue
        if c["state"] == "SILENCE":
            hold_portfolio.append({"ts": c["ts"], "signal": "LONG", "net": c["net4"] - FEE_BPS})
        elif c["state"] == "NEITHER" and c["score"] >= 3:
            hold_portfolio.append({"ts": c["ts"], "signal": "SHORT", "net": -c["net2"] - FEE_BPS})
    hold_portfolio.sort(key=lambda x: x["ts"])

    nets = [p["net"] for p in hold_portfolio]
    if nets:
        # Running P&L
        cum = []
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for v in nets:
            running += v
            cum.append(running)
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        # Worst consecutive loss streaks
        worst_streaks = []
        streak_loss = 0.0
        streak_len  = 0
        for v in nets:
            if v < 0:
                streak_loss += v
                streak_len  += 1
            else:
                if streak_len > 0:
                    worst_streaks.append((streak_len, streak_loss))
                streak_loss = 0.0
                streak_len  = 0
        if streak_len > 0:
            worst_streaks.append((streak_len, streak_loss))
        worst_streaks.sort(key=lambda x: x[1])

        # Percentiles
        sv = sorted(nets)
        p5  = sv[max(0, int(len(sv) * 0.05))]
        p10 = sv[max(0, int(len(sv) * 0.10))]
        p25 = sv[max(0, int(len(sv) * 0.25))]
        p75 = sv[int(len(sv) * 0.75)]
        p90 = sv[min(len(sv)-1, int(len(sv) * 0.90))]

        print(f"Portfolio holdout: N={len(nets)} sum={sum(nets):+.0f}bps WR={wr(nets):.1%}")
        print(f"  Mean={mean(nets):+.1f} Median={median(nets):+.1f}")
        print(f"  p5={p5:+.1f} p10={p10:+.1f} p25={p25:+.1f} p75={p75:+.1f} p90={p90:+.1f}")
        print(f"  Max drawdown: {max_dd:.0f} bps")
        print(f"  Worst streak: {worst_streaks[:3]}")

        pnl_out = {
            "n": len(nets), "sum_bps": round(sum(nets), 1), "wr": wr(nets),
            "mean": round(mean(nets), 1), "median": round(median(nets), 1),
            "p5": round(p5, 1), "p10": round(p10, 1), "p25": round(p25, 1),
            "p75": round(p75, 1), "p90": round(p90, 1),
            "max_drawdown_bps": round(max_dd, 1),
            "worst_streaks": [{"len": s[0], "loss_bps": round(s[1], 1)} for s in worst_streaks[:5]],
        }
    else:
        pnl_out = {"n": 0}
    out["Q9_pnl_distribution"] = pnl_out

    # ── 10. NOISY recovery ───────────────────────────────────────────────────
    print("\n=== 10. NOISY Recovery ===")
    nois_hold = [c for c in nois if c["is_hold"]]
    print(f"NOISY total: {len(nois)} holdout: {len(nois_hold)}")
    noisy_out = {}
    filters = [
        ("base",              lambda c: True),
        ("score>=2",          lambda c: c["score"] >= 2),
        ("score>=3",          lambda c: c["score"] >= 3),
        ("score>=4",          lambda c: c["score"] >= 4),
        ("vd>=30",            lambda c: c["vd"] >= 30),
        ("btc4h<0",           lambda c: c["b4h"] < 0),
        ("US_session",        lambda c: c["sess_us"]),
        ("score>=3+btc4h<0",  lambda c: c["score"] >= 3 and c["b4h"] < 0),
        ("score>=3+US",       lambda c: c["score"] >= 3 and c["sess_us"]),
        ("vd>=30+btc4h<0",    lambda c: c["vd"] >= 30 and c["b4h"] < 0),
    ]
    print(f"{'Filter':<22} {'N_all':>7} {'WR_all':>8} {'N_hold':>8} {'WR_hold':>9} {'T3R_hold':>10}")
    print("-" * 70)
    for label, fn in filters:
        all_f  = [c for c in nois if fn(c)]
        hold_f = [c for c in all_f if c["is_hold"]]
        all_v  = [-c["net2"] - FEE_BPS for c in all_f]
        hold_v = [-c["net2"] - FEE_BPS for c in hold_f]
        s = stats(hold_v)
        print(f"{label:<22} {len(all_f):>7} {(wr(all_v) or 0):>7.1%} "
              f"{s['n']:>8} {(s['wr'] or 0):>8.1%}  {(s['t3r'] or 0):>+9.0f}")
        noisy_out[label] = {"n_all": len(all_f), **s}
    out["Q10_noisy_recovery"] = noisy_out

    # ── Write output ─────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(_render_md(out), encoding="utf-8")
    print(f"\nOutput: {OUT_JSON}")
    print(f"Output: {OUT_MD}")
    return 0


def _render_md(r: dict) -> str:
    lines = ["# S34 State Machine Deep Research", "",
             f"Generated: `{r['generated_utc']}`  |  Fee: {r['fee_bps']} bps", ""]

    def _table(rows):
        lines.extend(rows)
        lines.append("")

    # Q1
    lines += ["## Q1. Stop Loss for SHORT", ""]
    q1 = r.get("Q1_stop_loss_short", {})
    _table(["| SL bps | N | WR | Mean | T3R | Trigger% |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {sl} | {s['n']} | {s['wr']:.1%} | {s['mean']:+.1f} | {s['t3r']:+.0f} "
            f"| {s['stop_trigger_rate']:.1%} |"
            for sl, s in q1.items()])

    # Q2
    lines += ["## Q2. BTC Threshold Sensitivity", ""]
    q2 = r.get("Q2_btc_threshold", {})
    _table(["| BTC Thr | N all | N hold | WR hold | Mean hold | T3R hold |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {thr} | {s['n_all']} | {s['n']} | {(s['wr'] or 0):.1%} "
            f"| {(s['mean'] or 0):+.1f} | {(s['t3r'] or 0):+.0f} |"
            for thr, s in q2.items()])

    # Q3
    lines += ["## Q3. BTC Entry Choice (First / Largest / Last)", ""]
    q3 = r.get("Q3_btc_entry_choice", {})
    _table(["| Entry | N | N hold | Avg Slippage | Hold WR | Hold Mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['n']} | {v['n_hold']} | {(v['avg_slippage_bps'] or 0):+.1f}bps "
            f"| {(v['hold_wr'] or 0):.1%} | {(v['hold_mean'] or 0):+.1f} |"
            for k, v in q3.items()])

    # Q4
    lines += ["## Q4. ETH Notional Sweet Spot", ""]
    q4 = r.get("Q4_notional_buckets", {})
    _table(["| Bucket | N sil | WR sil hold | N nei | WR nei hold |",
            "| --- | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['sil_n_all']} | {(v['sil_hold_wr'] or 0):.1%} "
            f"| {v['nei_n_all']} | {(v['nei_hold_wr'] or 0):.1%} |"
            for k, v in q4.items()])

    # Q5
    lines += ["## Q5. n2h (Sequential Cascade) Effect on NEITHER", ""]
    q5 = r.get("Q5_n2h_effect", {}).get("cumulative", {})
    _table(["| n2h >= | N NEITHER all | WR hold | Mean hold | T3R hold |",
            "| ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['nei_n_all']} | {(v['wr'] or 0):.1%} "
            f"| {(v['mean'] or 0):+.1f} | {(v['t3r'] or 0):+.0f} |"
            for k, v in q5.items()])

    # Q6
    lines += ["## Q6. Silence Window Sensitivity", ""]
    q6 = r.get("Q6_silence_window", {})
    _table(["| Window | N sil | WR sil hold | N nei | WR nei hold |",
            "| --- | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['sil_n_all']} | {(v['sil_hold_wr'] or 0):.1%} "
            f"| {v['nei_n_all']} | {(v['nei_hold_wr'] or 0):.1%} |"
            for k, v in q6.items()])

    # Q7
    lines += ["## Q7. Session Breakdown", ""]
    q7 = r.get("Q7_session", {})
    _table(["| Session | N sil | WR sil hold | N nei | WR nei hold |",
            "| --- | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['sil_n_all']} | {(v['sil_hold_wr'] or 0):.1%} "
            f"| {v['nei_n_all']} | {(v['nei_hold_wr'] or 0):.1%} |"
            for k, v in q7.items()])

    # Q8
    lines += ["## Q8. BTC Regime Effect", ""]
    q8 = r.get("Q8_btc_regime", {})
    _table(["| Condition | N all | N hold | WR hold | Mean | T3R |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['n_all']} | {v['n']} | {(v['wr'] or 0):.1%} "
            f"| {(v['mean'] or 0):+.1f} | {(v['t3r'] or 0):+.0f} |"
            for k, v in q8.items()])

    # Q9
    lines += ["## Q9. P&L Distribution and Max Drawdown (Holdout Portfolio)", ""]
    q9 = r.get("Q9_pnl_distribution", {})
    if q9.get("n"):
        lines += [f"Portfolio: SILENCE LONG + NEITHER(score≥3) SHORT  |  N={q9['n']}",
                  f"Sum={q9['sum_bps']:+.0f}bps  WR={q9['wr']:.1%}  Mean={q9['mean']:+.1f}  "
                  f"Median={q9['median']:+.1f}", "",
                  f"**Max drawdown: {q9['max_drawdown_bps']:.0f} bps**", "",
                  "| Percentile | bps |",
                  "| --- | ---: |",
                  f"| p5 | {q9['p5']:+.1f} |",
                  f"| p10 | {q9['p10']:+.1f} |",
                  f"| p25 | {q9['p25']:+.1f} |",
                  f"| p75 | {q9['p75']:+.1f} |",
                  f"| p90 | {q9['p90']:+.1f} |", ""]
        lines += ["Worst consecutive loss streaks:", ""]
        for s in q9.get("worst_streaks", []):
            lines.append(f"- {s['len']} trades: {s['loss_bps']:+.0f} bps")
        lines.append("")

    # Q10
    lines += ["## Q10. NOISY Recovery", ""]
    q10 = r.get("Q10_noisy_recovery", {})
    _table(["| Filter | N all | N hold | WR hold | Mean hold | T3R hold |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"] +
           [f"| {k} | {v['n_all']} | {v['n']} | {(v['wr'] or 0):.1%} "
            f"| {(v['mean'] or 0):+.1f} | {(v['t3r'] or 0):+.0f} |"
            for k, v in q10.items()])

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
