"""S34 State Machine Pre-Live Audit.

Answers four questions before going live with the full LONG+SHORT state machine:

A. Event distribution: how many events go to SILENCE/NEITHER/NOISY/OTHER daily?
B. Ultra-early BTC trap: does BTC cascade < 60s after ETH cascade hurt NEITHER WR?
C. LONG + NEITHER overlap: if a NEITHER fires during an active LONG, flip or hold?
D. Score filter impact on NEITHER SHORT.

Uses NAV_EVENTS + liquidations DB. No lookahead, DAT-01 compliant.
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
from statistics import median, mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
DEFAULT_DB  = ROOT / "data" / "microstructure.db"
OUT_JSON    = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_AUDIT.json"
OUT_MD      = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_AUDIT.md"

# Thresholds (match validated research)
LIVE_THRESH      = 200_000.0
SIL_GATE_LO_MS   = 60_000        # 1 min
SIL_GATE_HI_MS   = 30 * 60_000   # 30 min
PROP_THRESH      = 50_000.0       # ETH follow-on threshold
BTC_THRESH       = 500_000.0      # BTC cascade threshold
ULTRA_EARLY_MS   = 60_000         # BTC < 60s = ultra-early
FEE_BPS          = 5.0
SYNC_WIN_MS      = 10 * 60_000
HOLDOUT_FRAC     = 0.30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def load_events() -> list[dict]:
    rows = []
    with NAV_EVENTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def load_liq_arrays(conn: sqlite3.Connection, sym: str, side: str):
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (sym, side),
    ).fetchall()
    ts   = [int(r[0]) for r in rows]
    not_ = [float(r[1]) for r in rows]
    return ts, not_


def win_cnt(ts, vals, lo, hi, thr):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(1 for i in range(a, b) if vals[i] >= thr)


def win_max(ts, vals, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return max((vals[i] for i in range(a, b)), default=0.0)


def win_sum(ts, vals, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(vals[i] for i in range(a, b))


def first_above(ts, vals, lo, hi, thr):
    """First ts in [lo, hi] where notional >= thr. Returns None if not found."""
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    for i in range(a, b):
        if vals[i] >= thr:
            return int(ts[i])
    return None


def load_mark_prices(conn: sqlite3.Connection, sym: str, start_ms: int, end_ms: int):
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (sym, start_ms, end_ms),
    ).fetchall()
    ts  = [int(r[0]) for r in rows]
    px  = [float(r[1]) for r in rows]
    return ts, px


def mark_at_or_before(ts_arr, px_arr, t):
    idx = bisect.bisect_right(ts_arr, t) - 1
    return px_arr[idx] if idx >= 0 else None


def classify_event(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not):
    """Classify a single NAV event into state + compute score."""
    ts    = int(row["signal_ts_ms"])
    thr   = float(row.get("threshold_usd") or 0)
    net2  = float(row.get("net_2h_bps") or "nan")
    net4v = row.get("net_4h_bps")
    net4  = float(net4v) if net4v is not None else net2
    tags  = row.get("tags") or []

    if not math.isfinite(net2) or thr < LIVE_THRESH:
        return None  # skip

    # Silence classification
    n_prop  = win_cnt(eth_ts, eth_not, ts + SIL_GATE_LO_MS, ts + SIL_GATE_HI_MS, PROP_THRESH)
    sil_eth = n_prop == 0

    # BTC cascade in window
    btc_cascade_ts = first_above(btc_ts, btc_not, ts + SIL_GATE_LO_MS, ts + SIL_GATE_HI_MS, BTC_THRESH)
    sil_btc = btc_cascade_ts is None

    # Ultra-early BTC (< 60s)
    btc_ultra_ts = first_above(btc_ts, btc_not, ts, ts + ULTRA_EARLY_MS, BTC_THRESH)
    has_ultra_btc = btc_ultra_ts is not None

    # Context flags
    bull = "BULL_PULLBACK" in tags

    # State
    if sil_eth:
        state = "SILENCE"
    elif not sil_btc:
        state = "NEITHER"   # ETH noisy + BTC noisy
    else:
        state = "NOISY"     # ETH noisy + BTC quiet

    if bull:
        state = state + "_BULL"  # exclude from SHORT

    # Score features
    b4h      = float(row.get("btc4h_bps") or 0)
    vd       = float(row.get("vdepth_bps") or 0)
    bid      = float(row.get("bid_depth_usd") or 0)
    ts_dt    = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour     = ts_dt.hour
    weekday  = ts_dt.weekday() < 5
    sess_us  = 13 <= hour < 21
    sync_k   = win_sum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts) + win_sum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts)
    n2h      = win_cnt(eth_ts, eth_not, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)

    score = sum([
        int(sil_eth),
        int(n2h >= 3),
        int(b4h < 0),
        int(vd >= 30),
        int(sess_us),
        int(sync_k >= 200_000),
    ])

    return {
        "ts": ts, "date": ts_dt.date().isoformat(), "hour": hour,
        "net2": net2, "net4": net4,
        "state": state, "sil_eth": sil_eth, "sil_btc": sil_btc,
        "bull": bull,
        "btc_cascade_ts": btc_cascade_ts,
        "btc_cascade_delay_ms": (btc_cascade_ts - ts) if btc_cascade_ts else None,
        "has_ultra_btc": has_ultra_btc,
        "btc_ultra_ts": btc_ultra_ts,
        "score": score, "bid": bid, "weekday": weekday, "sess_us": sess_us,
        "vd": vd, "b4h": b4h, "n2h": n2h, "sync_k": sync_k,
    }


def wr_stats(vals):
    if not vals:
        return {"n": 0, "wr": None, "mean": None, "median": None, "t3r": None}
    wins = sum(1 for v in vals if v > 0)
    sv   = sorted(vals)
    t3r  = sum(sv[:-3]) if len(sv) > 3 else sum(sv)
    return {
        "n": len(vals),
        "wr": round(wins / len(vals), 3),
        "mean": round(mean(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(t3r, 0),
    }


def main() -> int:
    events = load_events()
    print(f"Loaded {len(events)} NAV_EVENTS")

    print("Loading liquidation arrays from DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq_arrays(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq_arrays(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq_arrays(conn, "SOLUSDT", "SELL")
        print(f"  ETH SELL: {len(eth_ts):,}  BTC SELL: {len(btc_ts):,}  SOL SELL: {len(sol_ts):,}")

        # Classify all events
        classified = []
        for row in events:
            c = classify_event(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not)
            if c is not None:
                classified.append(c)
        print(f"Classified: {len(classified)} events (>= 200K, finite net_2h)")

        # Holdout split
        n_cal  = int(len(classified) * (1 - HOLDOUT_FRAC))
        for i, c in enumerate(classified):
            c["is_holdout"] = i >= n_cal

        # ─── A. Distribution ────────────────────────────────────────────────────
        print("\n=== A. Event Distribution ===")
        state_counts: dict[str, list] = defaultdict(list)
        for c in classified:
            state_counts[c["state"]].append(c)

        print(f"{'State':<20} {'N':>6} {'N_hold':>8} {'WR_long':>8} {'WR_short':>9}")
        print("-" * 55)
        dist_out = {}
        for state in sorted(state_counts.keys()):
            evs  = state_counts[state]
            hold = [e for e in evs if e["is_holdout"]]
            # LONG = net4 - fee; SHORT = -net2 - fee
            long_vals  = [e["net4"] - FEE_BPS for e in hold]
            short_vals = [-e["net2"] - FEE_BPS for e in hold]
            wr_l = sum(1 for v in long_vals if v > 0) / len(long_vals) if long_vals else None
            wr_s = sum(1 for v in short_vals if v > 0) / len(short_vals) if short_vals else None
            print(f"{state:<20} {len(evs):>6} {len(hold):>8} "
                  f"{wr_l*100:>7.1f}%" if wr_l else f"{state:<20} {len(evs):>6} {len(hold):>8}     N/A")
            dist_out[state] = {
                "n_total": len(evs), "n_holdout": len(hold),
                "hold_wr_long": round(wr_l, 3) if wr_l else None,
                "hold_wr_short": round(wr_s, 3) if wr_s else None,
            }

        # Daily distribution
        by_date: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in classified:
            by_date[c["date"]][c["state"]] += 1
        dates = sorted(by_date.keys())
        per_day_total  = [sum(by_date[d].values()) for d in dates]
        per_day_sil    = [by_date[d].get("SILENCE", 0) for d in dates]
        per_day_nei    = [by_date[d].get("NEITHER", 0) for d in dates]
        print(f"\nDaily: total avg={mean(per_day_total):.1f}/day  silence avg={mean(per_day_sil):.1f}/day  neither avg={mean(per_day_nei):.1f}/day")
        print(f"  max/day total={max(per_day_total)}  silence={max(per_day_sil)}  neither={max(per_day_nei)}")

        # ─── B. Ultra-early BTC trap ─────────────────────────────────────────
        print("\n=== B. Ultra-Early BTC Trap (NEITHER events) ===")
        neither_evs = [c for c in classified if c["state"] == "NEITHER"]
        ultra_evs   = [c for c in neither_evs if c["has_ultra_btc"]]
        normal_evs  = [c for c in neither_evs if not c["has_ultra_btc"]]

        # WR for SHORT: profit = -net2 - fee
        ultra_hold  = [c for c in ultra_evs  if c["is_holdout"]]
        normal_hold = [c for c in normal_evs if c["is_holdout"]]
        ultra_vals  = [-c["net2"] - FEE_BPS for c in ultra_hold]
        normal_vals = [-c["net2"] - FEE_BPS for c in normal_hold]

        print(f"NEITHER total: {len(neither_evs)}  ultra_early BTC (<60s): {len(ultra_evs)}  normal: {len(normal_evs)}")
        print(f"Hold ultra:  {wr_stats(ultra_vals)}")
        print(f"Hold normal: {wr_stats(normal_vals)}")

        # BTC cascade delay distribution for normal NEITHER
        delays_ms = [c["btc_cascade_delay_ms"] for c in neither_evs if c["btc_cascade_delay_ms"] is not None]
        delay_buckets = {"<2min": 0, "2-5min": 0, "5-15min": 0, "15-30min": 0}
        for d in delays_ms:
            if d < 2 * 60_000: delay_buckets["<2min"] += 1
            elif d < 5 * 60_000: delay_buckets["2-5min"] += 1
            elif d < 15 * 60_000: delay_buckets["5-15min"] += 1
            else: delay_buckets["15-30min"] += 1
        print(f"BTC delay distribution: {delay_buckets}")
        if delays_ms:
            print(f"  median delay: {median(delays_ms)/60000:.1f}min  mean: {mean(delays_ms)/60000:.1f}min")

        btc_b_out = {
            "neither_n": len(neither_evs),
            "ultra_early_n": len(ultra_evs),
            "normal_n": len(normal_evs),
            "hold_ultra": wr_stats(ultra_vals),
            "hold_normal": wr_stats(normal_vals),
            "delay_distribution": delay_buckets,
            "median_delay_min": round(median(delays_ms) / 60000, 1) if delays_ms else None,
        }

        # ─── C. LONG + NEITHER overlap (flip vs hold) ─────────────────────────
        print("\n=== C. LONG + NEITHER Overlap ===")
        silence_evs = [c for c in classified if c["state"] == "SILENCE"]
        neither_sorted = sorted([c for c in classified if c["state"] == "NEITHER"], key=lambda x: x["ts"])
        neither_ts_arr = [c["ts"] for c in neither_sorted]

        # Load ETH mark prices in bulk for the overlap windows
        if silence_evs:
            sil_min_ts = min(c["ts"] for c in silence_evs)
            sil_max_ts = max(c["ts"] for c in silence_evs) + 4 * 3600_000 + 60_000
            print(f"Loading ETH mark prices for overlap windows...")
            mk_ts, mk_px = load_mark_prices(conn, "ETHUSDT", sil_min_ts - 60_000, sil_max_ts)
            print(f"  Loaded {len(mk_ts):,} mark price rows")
        else:
            mk_ts, mk_px = [], []

        overlap_results = []
        n_overlap = 0
        for sil in silence_evs:
            t_entry  = sil["ts"]
            t_exit4h = t_entry + 4 * 3600_000
            # Find NEITHER events in [t_entry+60s, t_entry+4h]
            lo = bisect.bisect_left(neither_ts_arr, t_entry + 60_000)
            hi = bisect.bisect_right(neither_ts_arr, t_exit4h)
            overlapping = neither_sorted[lo:hi]
            if not overlapping:
                continue
            n_overlap += 1
            # Take the first NEITHER signal
            nei = overlapping[0]
            t_neither = nei["ts"]

            # HOLD: keep LONG full 4h
            hold_net = sil["net4"] - FEE_BPS

            # FLIP: exit LONG at NEITHER time, enter SHORT
            # Compute early LONG exit return using mark prices
            px_entry  = mark_at_or_before(mk_ts, mk_px, t_entry)
            px_at_nei = mark_at_or_before(mk_ts, mk_px, t_neither)
            if px_entry and px_at_nei and px_entry > 0:
                long_partial_bps = (px_at_nei - px_entry) / px_entry * 10_000.0
            else:
                long_partial_bps = None

            short_net = -nei["net2"] - FEE_BPS  # SHORT 2h from NEITHER time
            if long_partial_bps is not None:
                flip_net = (long_partial_bps - FEE_BPS) + short_net  # exit LONG + enter SHORT
            else:
                flip_net = None

            overlap_results.append({
                "sil_ts": t_entry,
                "nei_ts": t_neither,
                "delay_min": round((t_neither - t_entry) / 60_000, 1),
                "hold_net": round(hold_net, 1),
                "flip_net": round(flip_net, 1) if flip_net is not None else None,
                "long_partial_bps": round(long_partial_bps, 1) if long_partial_bps is not None else None,
                "short_net": round(short_net, 1),
                "is_holdout": sil["is_holdout"],
            })

        print(f"SILENCE events with overlapping NEITHER in 4h window: {n_overlap} / {len(silence_evs)}")
        hold_vals_c  = [r["hold_net"] for r in overlap_results if r["is_holdout"]]
        flip_vals_c  = [r["flip_net"] for r in overlap_results if r["is_holdout"] and r["flip_net"] is not None]
        print(f"Hold scenario: {wr_stats(hold_vals_c)}")
        print(f"Flip scenario: {wr_stats(flip_vals_c)}")
        if hold_vals_c and flip_vals_c:
            hold_sum = sum(hold_vals_c)
            flip_sum = sum(flip_vals_c)
            print(f"Sum diff (flip - hold): {flip_sum - hold_sum:+.0f} bps over {len(flip_vals_c)} events")

        overlap_hold_c   = {"scenario": "HOLD_LONG", **wr_stats(hold_vals_c)}
        overlap_flip_c   = {"scenario": "FLIP_SHORT", **wr_stats(flip_vals_c)}

        # Full dataset (cal + hold combined)
        all_hold_vals = [r["hold_net"] for r in overlap_results]
        all_flip_vals = [r["flip_net"] for r in overlap_results if r["flip_net"] is not None]
        print(f"All (cal+hold): hold={wr_stats(all_hold_vals)['wr']} flip={wr_stats(all_flip_vals)['wr']}")

        c_out = {
            "n_silence_events": len(silence_evs),
            "n_with_overlap": n_overlap,
            "overlap_rate": round(n_overlap / len(silence_evs), 3) if silence_evs else 0,
            "holdout_hold": overlap_hold_c,
            "holdout_flip": overlap_flip_c,
            "sample": overlap_results[:10],  # first 10 for inspection
        }

        # ─── D. Score filter for NEITHER SHORT ─────────────────────────────────
        print("\n=== D. Score Filter for NEITHER SHORT ===")
        # For NEITHER (sil_eth=0), max score = 5
        by_score: dict[int, list[float]] = defaultdict(list)
        by_score_hold: dict[int, list[float]] = defaultdict(list)
        for c in neither_evs:
            net = -c["net2"] - FEE_BPS
            by_score[c["score"]].append(net)
            if c["is_holdout"]:
                by_score_hold[c["score"]].append(net)

        print(f"{'Score':<8} {'N_all':>7} {'WR_all':>8} {'N_hold':>8} {'WR_hold':>9} {'T3R_hold':>10}")
        print("-" * 55)
        d_out_scores = {}
        for sc in sorted(by_score.keys()):
            all_v  = by_score[sc]
            hold_v = by_score_hold.get(sc, [])
            wr_a   = sum(1 for v in all_v if v > 0) / len(all_v) if all_v else 0
            wr_h   = sum(1 for v in hold_v if v > 0) / len(hold_v) if hold_v else None
            sv     = sorted(hold_v)
            t3r    = sum(sv[:-3]) if len(sv) > 3 else sum(sv)
            print(f"{sc:<8} {len(all_v):>7} {wr_a:>7.1%} {len(hold_v):>8} "
                  f"{wr_h:>8.1%}  {t3r:>+9.0f}" if wr_h is not None else
                  f"{sc:<8} {len(all_v):>7} {wr_a:>7.1%} {len(hold_v):>8}      N/A")
            d_out_scores[str(sc)] = {"n_all": len(all_v), "n_hold": len(hold_v),
                                      "wr_all": round(wr_a, 3), "wr_hold": round(wr_h, 3) if wr_h else None}

        # Cumulative: score >= threshold
        print(f"\n{'Score>=':>8} {'N_all':>7} {'WR_all':>8} {'N_hold':>8} {'WR_hold':>9} {'T3R_hold':>10}")
        print("-" * 55)
        d_out_cum = {}
        for thr in range(0, 6):
            all_v  = [v for sc, vals in by_score.items() if sc >= thr for v in vals]
            hold_v = [v for sc, vals in by_score_hold.items() if sc >= thr for v in vals]
            if not all_v: continue
            wr_a   = sum(1 for v in all_v if v > 0) / len(all_v)
            wr_h   = sum(1 for v in hold_v if v > 0) / len(hold_v) if hold_v else None
            sv     = sorted(hold_v)
            t3r    = sum(sv[:-3]) if len(sv) > 3 else sum(sv)
            print(f"{thr:<8} {len(all_v):>7} {wr_a:>7.1%} {len(hold_v):>8} "
                  f"{wr_h:>8.1%}  {t3r:>+9.0f}" if wr_h is not None else
                  f"{thr:<8} {len(all_v):>7} {wr_a:>7.1%} {len(hold_v):>8}      N/A")
            d_out_cum[f">={thr}"] = {"n_all": len(all_v), "n_hold": len(hold_v),
                                      "wr_all": round(wr_a, 3), "wr_hold": round(wr_h, 3) if wr_h else None,
                                      "t3r_hold": round(t3r, 0)}

        # ─── Final report ───────────────────────────────────────────────────────
        report = {
            "generated_utc": utc_now(),
            "n_events_total": len(events),
            "n_events_classified": len(classified),
            "n_events_holdout": sum(1 for c in classified if c["is_holdout"]),
            "fee_bps": FEE_BPS,
            "A_distribution": {
                "by_state": dist_out,
                "daily_avg_total": round(mean(per_day_total), 1),
                "daily_avg_silence": round(mean(per_day_sil), 1),
                "daily_avg_neither": round(mean(per_day_nei), 1),
                "daily_max_total": max(per_day_total),
                "daily_max_silence": max(per_day_sil),
                "daily_max_neither": max(per_day_nei),
            },
            "B_ultra_early_btc": btc_b_out,
            "C_long_neither_overlap": c_out,
            "D_score_filter_neither": {
                "by_score_exact": d_out_scores,
                "by_score_cumulative": d_out_cum,
            },
        }

        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

        md = _render_md(report, overlap_results)
        OUT_MD.write_text(md, encoding="utf-8")
        print(f"\nOutput: {OUT_JSON}")
        print(f"Output: {OUT_MD}")
        return 0


def _render_md(r: dict, overlap_details: list) -> str:
    lines = ["# S34 State Machine Pre-Live Audit", "",
             f"Generated: `{r['generated_utc']}`", "",
             f"Events: {r['n_events_classified']} classified (>=200K, finite net_2h) "
             f"from {r['n_events_total']} total. Holdout: {r['n_events_holdout']}.", ""]

    # A
    a = r["A_distribution"]
    lines += ["## A. Event Distribution", "",
              f"Daily average: **{a['daily_avg_total']}/day** total  |  "
              f"SILENCE {a['daily_avg_silence']}/day  |  NEITHER {a['daily_avg_neither']}/day", "",
              "| State | N total | N holdout | Hold WR (LONG) | Hold WR (SHORT) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for state, d in a["by_state"].items():
        wr_l = f"{d['hold_wr_long']:.1%}" if d["hold_wr_long"] else "N/A"
        wr_s = f"{d['hold_wr_short']:.1%}" if d["hold_wr_short"] else "N/A"
        lines.append(f"| {state} | {d['n_total']} | {d['n_holdout']} | {wr_l} | {wr_s} |")
    lines.append("")

    # B
    b = r["B_ultra_early_btc"]
    lines += ["## B. Ultra-Early BTC Trap (NEITHER SHORT)", "",
              f"NEITHER total: {b['neither_n']}  |  Ultra-early BTC (<60s): {b['ultra_early_n']}  |  Normal: {b['normal_n']}", "",
              "| Condition | N holdout | WR | Mean bps | T3R |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for label, key in [("Ultra-early BTC (<60s)", "hold_ultra"), ("Normal BTC timing", "hold_normal")]:
        s = b[key]
        wr  = f"{s['wr']:.1%}" if s["wr"] is not None else "N/A"
        mn  = f"{s['mean']:+.1f}" if s["mean"] is not None else "N/A"
        t3r = f"{s['t3r']:+.0f}" if s["t3r"] is not None else "N/A"
        lines.append(f"| {label} | {s['n']} | {wr} | {mn} | {t3r} |")
    lines.append("")
    d_dist = b.get("delay_distribution", {})
    lines.append(f"BTC cascade delay: {d_dist}  median={b.get('median_delay_min')}min")
    lines.append("")

    # C
    c = r["C_long_neither_overlap"]
    lines += ["## C. LONG + NEITHER Overlap (Flip vs Hold)", "",
              f"SILENCE events: {c['n_silence_events']}  |  With overlapping NEITHER in 4h: "
              f"{c['n_with_overlap']} ({c['overlap_rate']:.1%})", "",
              "| Scenario | N holdout | WR | Mean bps | T3R |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for scen in ["holdout_hold", "holdout_flip"]:
        s   = c[scen]
        wr  = f"{s['wr']:.1%}" if s.get("wr") is not None else "N/A"
        mn  = f"{s['mean']:+.1f}" if s.get("mean") is not None else "N/A"
        t3r = f"{s['t3r']:+.0f}" if s.get("t3r") is not None else "N/A"
        lines.append(f"| {s['scenario']} | {s['n']} | {wr} | {mn} | {t3r} |")
    lines.append("")

    # D
    d = r["D_score_filter_neither"]
    lines += ["## D. Score Filter for NEITHER SHORT", "",
              "Score = n2h>=3 + btc4h<0 + vdepth>=30 + US_session + sync_k>=200K (max=5, sil_eth always 0)", "",
              "### By exact score", "",
              "| Score | N all | WR all | N hold | WR hold |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for sc, s in d["by_score_exact"].items():
        wr_a = f"{s['wr_all']:.1%}"
        wr_h = f"{s['wr_hold']:.1%}" if s["wr_hold"] else "N/A"
        lines.append(f"| {sc} | {s['n_all']} | {wr_a} | {s['n_hold']} | {wr_h} |")
    lines += ["", "### Cumulative (score >= threshold)", "",
              "| Score >= | N all | WR all | N hold | WR hold | T3R hold |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for thr, s in d["by_score_cumulative"].items():
        wr_a = f"{s['wr_all']:.1%}"
        wr_h = f"{s['wr_hold']:.1%}" if s["wr_hold"] else "N/A"
        t3r  = f"{s['t3r_hold']:+.0f}" if s.get("t3r_hold") is not None else "N/A"
        lines.append(f"| {thr} | {s['n_all']} | {wr_a} | {s['n_hold']} | {wr_h} | {t3r} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
