"""S34 Next-Round Research — T1-T6 comprehensive backtest.

Tests (all single-process, sequential):
  T1  SHORT_NOISY_BTC1M  : noisy ETH follow-on + BTC>=Xm → SHORT extraction
  T2  C_score_relax narrow: base_score1 LONG on strict live-like universe
  T3  btc4h vs btc7d      : which gate is more stable?
  T4  ASIA session        : ASIA-only LONG deep-dive
  T5  Notional bands      : running_notional filter (200K-300K, 300K-500K, 500K+)
  T6  SHORT expansion     : BTC threshold, delay, hold-time sweep

Output:
  reports/research/s34/S34_NEXT_TESTS_V1.json
  reports/research/s34/S34_NEXT_TESTS_V1.md
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH   = ROOT / "data" / "microstructure.db"
OUT_DIR   = ROOT / "reports" / "research" / "s34"
OUT_JSON  = OUT_DIR / "S34_NEXT_TESTS_V1.json"
OUT_MD    = OUT_DIR / "S34_NEXT_TESTS_V1.md"

ETH_THRESH   = 200_000.0
PROP_THRESH  =  50_000.0
LOOKBACK_MS  = 400 * 24 * 3600_000   # full history
FEE_BPS      = 5.0
MC_ITER      = 1000
random.seed(42)

# ── DB helpers ────────────────────────────────────────────────────────────────

def _scalar(conn, sql, params):
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row else 0.0

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def prior_bps(conn, sym, ts_ms, lookback_ms):
    row0 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms - lookback_ms)).fetchone()
    row1 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    if row0 and row1 and float(row0[0]) > 0:
        return (float(row1[0]) - float(row0[0])) / float(row0[0]) * 10_000.0
    return 0.0

def mark_at_or_after(marks, ts_ms):
    """marks is a MarkIndex from load_mark_index."""
    r = marks.at_or_after(ts_ms)
    return float(r[1]) if r else None

def mark_at_or_before(marks, ts_ms):
    r = marks.at_or_before(ts_ms)
    return float(r[1]) if r else None

def session_name(ts_ms):
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= hour < 13:   return "EUROPE"
    if 13 <= hour < 21:  return "US"
    return "OFF"

def dow_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()

def hour_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour

# ── Stats helpers ─────────────────────────────────────────────────────────────

def stats(vals):
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "t3r": None, "worst": None, "tail_n": 0}
    n = len(vals)
    wins = sum(1 for v in vals if v > 0)
    s = sorted(vals)
    t3 = vals[:]
    t3.sort(reverse=True)
    t3r = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    return {
        "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(sum(vals) / n, 1),
        "t3r": round(t3r, 1),
        "worst": round(s[0], 1),
        "tail_n": sum(1 for v in vals if v < -100),
        "per_month": round(n / max(1, total_months), 1),
    }

def mc_p(obs_avg, vals, n_iter=MC_ITER):
    if len(vals) < 4:
        return None
    rng = random.Random(0)
    count = sum(1 for _ in range(n_iter)
                if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / len(vals) >= obs_avg)
    return round(count / n_iter, 3)

def holdout_stats(vals, split=0.7):
    """70% dev / 30% holdout chronological split."""
    n = len(vals)
    cut = max(1, int(n * split))
    dev = vals[:cut]
    ho  = vals[cut:]
    return {
        "dev_n": len(dev), "dev_avg": round(sum(dev)/len(dev),1) if dev else None,
        "dev_wr": round(100*sum(1 for v in dev if v>0)/len(dev),1) if dev else None,
        "ho_n": len(ho), "ho_avg": round(sum(ho)/len(ho),1) if ho else None,
        "ho_wr": round(100*sum(1 for v in ho if v>0)/len(ho),1) if ho else None,
        "ho_sum": round(sum(ho),1) if ho else None,
    }

def full_stat(vals, label=""):
    if not vals:
        return {"label": label, "n": 0}
    s = stats(vals)
    s["label"] = label
    s["mc_p"] = mc_p(s["avg"] or 0, vals)
    s.update(holdout_stats(vals))
    return s

# ── Feature computation (point-in-time) ──────────────────────────────────────

def compute_features(conn, ts, sync_k, n2h):
    book  = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep  = float(book.get("vdepth_bps") or 0) if book else 0.0
    b4h   = prior_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    hour  = hour_of(ts)
    sess_us = 13 <= hour < 21
    score = sum([
        int(n2h >= 3),
        int(b4h < 0),
        int(vdep >= 30),
        int(sess_us),
        int(sync_k >= 200_000),
    ])
    eth1h = prior_bps(conn, "ETHUSDT", ts, 3600_000)
    bull  = eth1h > 20.0 and b4h > 50.0
    btc7d = prior_bps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000)
    btc3d = prior_bps(conn, "BTCUSDT", ts, 3 * 24 * 3600_000)
    return {
        "score": score, "b4h": b4h, "vdep": vdep, "bull": bull,
        "btc7d": btc7d, "btc3d": btc3d, "btc4h": b4h,
    }

# ── Outcome computation ───────────────────────────────────────────────────────

def long_outcome(marks, entry_ts, hold_ms):
    p0 = mark_at_or_after(marks, entry_ts)
    if p0 is None or p0 <= 0:
        return None
    p1 = mark_at_or_before(marks, entry_ts + hold_ms)
    if p1 is None:
        return None
    return round((p1 - p0) / p0 * 10_000.0 - FEE_BPS, 2)

def short_outcome(marks, entry_ts, hold_ms):
    p0 = mark_at_or_after(marks, entry_ts)
    if p0 is None or p0 <= 0:
        return None
    p1 = mark_at_or_before(marks, entry_ts + hold_ms)
    if p1 is None:
        return None
    return round(-(p1 - p0) / p0 * 10_000.0 - FEE_BPS, 2)

# ── Main data build ───────────────────────────────────────────────────────────

def build_anchors(conn):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    lo = now_ms - LOOKBACK_MS
    print(f"Loading liquidations from {datetime.fromtimestamp(lo/1000,tz=timezone.utc).date()} ...")
    liqs = load_liquidations(conn, "ETHUSDT", "SELL", lo, now_ms)
    print(f"  {len(liqs)} ETH SELL liq rows loaded")
    anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                  thresholds=(ETH_THRESH,), accel_window_sec=30)
    print(f"  {len(anchors)} anchors reconstructed")
    return anchors

def build_events(conn, anchors, marks_eth):
    """Build feature-rich event list from all anchors."""
    events = []
    total = len(anchors)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print(f"  feature [{i+1}/{total}] ...")
        ts   = int(anc.anchor_ts_ms)
        rn   = float(anc.running_notional)
        p0   = mark_at_or_after(marks_eth, ts)
        if p0 is None:
            continue
        sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts)
                + liq_sum(conn, "SOLUSDT", "SELL", ts - 10*60_000, ts))
        n2h  = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
        feat = compute_features(conn, ts, sync_k, n2h)
        sess = session_name(ts)
        dow  = dow_of(ts)
        hour = hour_of(ts)
        blocked_us = (sess == "US" and hour in {13, 14})
        long_elig = (
            not feat["bull"]
            and sess != "EUROPE"
            and not blocked_us
            and dow not in {0, 2}
            and sync_k < 200_000
            and feat["btc7d"] < 0
            and (feat["score"] + 1) >= 3
        )
        short_elig = (
            not feat["bull"]
            and sess != "EUROPE"
            and dow != 6
            and feat["score"] >= 4
        )
        # Noisy follow-on detection (T1)
        noisy_ts  = liq_first_ts(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 30*60_000, PROP_THRESH)
        noisy     = noisy_ts is not None
        # BTC confirm at different thresholds (T1 / T6)
        btc_max_30m = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000, ts + 30*60_000)
        btc_max_60m = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000, ts + 60*60_000)
        btc_max_10d = liq_max(conn, "BTCUSDT", "SELL", ts + 10*60_000, ts + 30*60_000)
        btc_max_15d = liq_max(conn, "BTCUSDT", "SELL", ts + 15*60_000, ts + 60*60_000)
        events.append({
            "ts": ts, "rn": rn, "p0": p0,
            "sync_k": sync_k, "n2h": n2h,
            "score": feat["score"], "btc4h": feat["btc4h"],
            "btc7d": feat["btc7d"], "btc3d": feat["btc3d"],
            "bull": feat["bull"], "sess": sess, "dow": dow, "hour": hour,
            "blocked_us": blocked_us,
            "long_elig": long_elig, "short_elig": short_elig,
            "noisy": noisy, "noisy_ts": noisy_ts,
            "btc_max_5d_30m": btc_max_30m,
            "btc_max_5d_60m": btc_max_60m,
            "btc_max_10d_30m": btc_max_10d,
            "btc_max_15d_60m": btc_max_15d,
        })
    return events

def add_outcomes(events, marks_eth):
    """Compute outcomes for all hold times needed."""
    print(f"Computing outcomes for {len(events)} events ...")
    for ev in events:
        ts = ev["ts"]
        ev["long_4h"] = long_outcome(marks_eth, ts, 4 * 3600_000)
        ev["long_2h"] = long_outcome(marks_eth, ts, 2 * 3600_000)
        ev["short_2h"] = short_outcome(marks_eth, ts, 2 * 3600_000)
        ev["short_90m"] = short_outcome(marks_eth, ts, 90 * 60_000)
        ev["short_1h"] = short_outcome(marks_eth, ts, 3600_000)
        # Noisy entry SHORT outcomes (entry at noisy_ts, not anchor_ts)
        if ev.get("noisy_ts"):
            nt = ev["noisy_ts"]
            ev["short_noisy_2h"] = short_outcome(marks_eth, nt, 2 * 3600_000)
            ev["short_noisy_1h"] = short_outcome(marks_eth, nt, 3600_000)
        else:
            ev["short_noisy_2h"] = None
            ev["short_noisy_1h"] = None

# ── Filters ───────────────────────────────────────────────────────────────────

def base_long(ev):
    """Current live gate: sync<200K + btc7d<0 + score>=2 + not bull + not EUROPE + not Mon/Wed + not US 13-14."""
    return (
        not ev["bull"]
        and ev["sess"] != "EUROPE"
        and not ev["blocked_us"]
        and ev["dow"] not in {0, 2}
        and ev["sync_k"] < 200_000
        and ev["btc7d"] < 0
        and ev["score"] >= 2  # long_score = score+1 >= 3
    )

def base_short_elig(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and ev["dow"] != 6 and ev["score"] >= 4)

# ── T1: SHORT_NOISY_BTC1M ─────────────────────────────────────────────────────

def run_T1(events):
    print("\n=== T1: SHORT_NOISY_BTC1M ===")
    results = {}

    # Subsets: noisy ETH + BTC>=Xm (different thresholds and windows)
    configs = [
        ("noisy_btc1m_5d30m",  "noisy + BTC>=1M (5min delay, 30m window)", 1_000_000, "5d_30m", "short_noisy_2h"),
        ("noisy_btc2m_5d30m",  "noisy + BTC>=2M (5min delay, 30m window)", 2_000_000, "5d_30m", "short_noisy_2h"),
        ("noisy_btc500k_5d30m","noisy + BTC>=500K (5min delay, 30m window)", 500_000,  "5d_30m", "short_noisy_2h"),
        ("noisy_btc1m_5d30m_1h","noisy + BTC>=1M  2h vs 1h hold", 1_000_000, "5d_30m", "short_noisy_1h"),
        ("noisy_btc1m_10d30m", "noisy + BTC>=1M (10min delay, 30m window)", 1_000_000, "10d_30m","short_noisy_2h"),
    ]

    btc_key_map = {
        "5d_30m":  "btc_max_5d_30m",
        "5d_60m":  "btc_max_5d_60m",
        "10d_30m": "btc_max_10d_30m",
        "15d_60m": "btc_max_15d_60m",
    }

    # Base: noisy LONG (what we're inverting)
    noisy_long = [ev["long_4h"] for ev in events if ev.get("noisy") and ev.get("long_4h") is not None]
    results["NOISY_LONG_base"] = full_stat(noisy_long, "LONG if noisy (all, no BTC filter)")

    # All noisy events SHORT
    noisy_short_2h = [ev["short_noisy_2h"] for ev in events
                      if ev.get("noisy") and ev.get("short_noisy_2h") is not None]
    results["NOISY_SHORT_all_2h"] = full_stat(noisy_short_2h, "SHORT at noisy entry, all noisy events, 2h")

    for name, label, btc_thr, btc_win, outcome_key in configs:
        bkey = btc_key_map[btc_win]
        vals = [ev[outcome_key] for ev in events
                if ev.get("noisy") and ev.get(outcome_key) is not None
                and ev.get(bkey, 0) >= btc_thr]
        results[name] = full_stat(vals, label)
        s = results[name]
        print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} T3R={s.get('t3r')} MC_p={s.get('mc_p')}")

    # Noisy + BTC1M + score filter
    for score_thr in [3, 4]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev.get("noisy") and ev.get("short_noisy_2h") is not None
                and ev.get("btc_max_5d_30m", 0) >= 1_000_000
                and ev.get("score", 0) >= score_thr]
        key = f"noisy_btc1m_score{score_thr}"
        results[key] = full_stat(vals, f"noisy+BTC>=1M+score>={score_thr}, 2h SHORT")
        s = results[key]
        print(f"  {results[key]['label']}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")

    return results

# ── T2: C_score_relax narrow ──────────────────────────────────────────────────

def run_T2(events):
    print("\n=== T2: C_score_relax narrow ===")
    results = {}

    # Current live gate reference
    live_vals = [ev["long_4h"] for ev in events if base_long(ev) and ev.get("long_4h") is not None]
    results["current_live_LONG"] = full_stat(live_vals, "Current live LONG gate (strict)")

    # Relaxation: remove btc7d requirement
    no_btc7d = [ev["long_4h"] for ev in events
                if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                    and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                    and ev["score"] >= 2 and ev.get("long_4h") is not None)]
    results["relax_no_btc7d"] = full_stat(no_btc7d, "LONG: remove btc7d gate")

    # Relaxation: add score=1 events (long_score=2, still btc7d<0)
    score1_added = [ev["long_4h"] for ev in events
                    if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                        and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                        and ev["btc7d"] < 0 and ev["score"] >= 1
                        and ev.get("long_4h") is not None)]
    results["relax_score1_btc7d"] = full_stat(score1_added, "LONG: score>=1 + btc7d<0")

    # Added-only for score1
    score1_only = [ev["long_4h"] for ev in events
                   if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                       and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                       and ev["btc7d"] < 0 and ev["score"] == 1
                       and ev.get("long_4h") is not None)]
    results["score1_events_only"] = full_stat(score1_only, "LONG: score=1 events only (added vs current)")

    # Remove Mon block only
    no_mon_block = [ev["long_4h"] for ev in events
                    if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                        and ev["dow"] not in {2}  # only Wed blocked
                        and ev["sync_k"] < 200_000 and ev["btc7d"] < 0
                        and ev["score"] >= 2 and ev.get("long_4h") is not None)]
    results["relax_no_mon_block"] = full_stat(no_mon_block, "LONG: remove Mon block only")

    for k, s in results.items():
        print(f"  {s['label']}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")
    return results

# ── T3: btc4h vs btc7d ───────────────────────────────────────────────────────

def run_T3(events):
    print("\n=== T3: btc4h vs btc7d ===")
    results = {}
    base = lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                       and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                       and ev["score"] >= 2 and ev.get("long_4h") is not None)

    configs = [
        ("btc7d_lt0",      "btc7d<0",               lambda ev: ev["btc7d"] < 0),
        ("btc4h_lt0",      "btc4h<0",               lambda ev: ev["btc4h"] < 0),
        ("btc3d_lt0",      "btc3d<0",               lambda ev: ev["btc3d"] < 0),
        ("both_btc4h_7d",  "btc4h<0 AND btc7d<0",   lambda ev: ev["btc4h"] < 0 and ev["btc7d"] < 0),
        ("btc4h_or_7d",    "btc4h<0 OR btc7d<0",    lambda ev: ev["btc4h"] < 0 or ev["btc7d"] < 0),
        ("no_mom_gate",    "no BTC momentum gate",   lambda ev: True),
        ("btc7d_lt500",    "btc7d<+500 (relaxed)",   lambda ev: ev["btc7d"] < 500),
        ("btc4h_lt0_no7d", "btc4h<0 (no btc7d)",    lambda ev: ev["btc4h"] < 0),
        ("btc7d_neg500_0", "btc7din(-500,0)",         lambda ev: -500 < ev["btc7d"] < 0),
    ]

    for name, label, flt in configs:
        vals = [ev["long_4h"] for ev in events if base(ev) and flt(ev)]
        results[name] = full_stat(vals, label)
        s = results[name]
        print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} T3R={s.get('t3r')} MC_p={s.get('mc_p')}")

    # Month stability: by quarter for btc4h vs btc7d
    for name, label, flt in [("btc7d_lt0","btc7d<0",lambda ev: ev["btc7d"]<0),
                               ("btc4h_lt0","btc4h<0",lambda ev: ev["btc4h"]<0)]:
        monthly: dict[str, list] = {}
        for ev in events:
            if not base(ev) or not flt(ev) or ev.get("long_4h") is None:
                continue
            mo = datetime.fromtimestamp(ev["ts"]/1000, tz=timezone.utc).strftime("%Y-%m")
            monthly.setdefault(mo, []).append(ev["long_4h"])
        pos_months = sum(1 for v in monthly.values() if sum(v)/len(v) > 0)
        results[f"{name}_month_stability"] = {
            "label": f"{label} month stability",
            "months_positive": pos_months, "total_months": len(monthly),
        }
        print(f"  {label} months positive: {pos_months}/{len(monthly)}")
    return results

# ── T4: ASIA session ─────────────────────────────────────────────────────────

def run_T4(events):
    print("\n=== T4: ASIA session LONG ===")
    results = {}
    base = lambda ev: (not ev["bull"] and ev["sync_k"] < 200_000
                       and ev["score"] >= 2 and ev.get("long_4h") is not None)

    configs = [
        ("asia_btc7d",  "ASIA sess + btc7d<0",           lambda ev: ev["sess"]=="OFF" and ev["btc7d"]<0),
        ("asia_btc4h",  "ASIA sess + btc4h<0 (no btc7d)", lambda ev: ev["sess"]=="OFF" and ev["btc4h"]<0),
        ("us_btc7d",    "US sess + btc7d<0",              lambda ev: ev["sess"]=="US" and not ev["blocked_us"] and ev["btc7d"]<0),
        ("us_btc4h",    "US sess + btc4h<0",              lambda ev: ev["sess"]=="US" and not ev["blocked_us"] and ev["btc4h"]<0),
        ("all_sess_7d", "All sess + btc7d<0 (no dow/sess block)", lambda ev: ev["btc7d"]<0),
        ("asia_no_mom", "ASIA sess + no btc mom gate",    lambda ev: ev["sess"]=="OFF"),
    ]

    for name, label, flt in configs:
        vals = [ev["long_4h"] for ev in events if base(ev) and flt(ev) and ev["dow"] not in {0,2}]
        results[name] = full_stat(vals, label)
        s = results[name]
        print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")
    return results

# ── T5: Notional bands ────────────────────────────────────────────────────────

def run_T5(events):
    print("\n=== T5: Notional bands ===")
    results = {}
    base = lambda ev: (base_long(ev) and ev.get("long_4h") is not None)

    bands = [
        ("200k_300k", "rn 200K–300K", 200_000, 300_000),
        ("300k_500k", "rn 300K–500K", 300_000, 500_000),
        ("500k_1m",   "rn 500K–1M",   500_000, 1_000_000),
        ("1m_plus",   "rn >1M",       1_000_000, 1e15),
        ("all",       "rn all",        0,       1e15),
    ]

    for name, label, lo, hi in bands:
        vals = [ev["long_4h"] for ev in events if base(ev) and lo <= ev["rn"] < hi]
        results[name] = full_stat(vals, label)
        s = results[name]
        print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")

    # Also without gate (just sync<200K) to see full picture
    for name, label, lo, hi in bands[:4]:
        vals = [ev["long_4h"] for ev in events
                if not ev["bull"] and ev["sync_k"] < 200_000
                and ev["score"] >= 2 and ev.get("long_4h") is not None
                and lo <= ev["rn"] < hi]
        results[f"nogate_{name}"] = full_stat(vals, f"no-btc7d | {label}")
        s = results[f"nogate_{name}"]
        print(f"  no-btc7d | {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')}")
    return results

# ── T6: SHORT gate expansion ─────────────────────────────────────────────────

def run_T6(events):
    print("\n=== T6: SHORT gate expansion ===")
    results = {}

    # BTC threshold × delay × hold time sweep
    btc_thrs = [(500_000, "500K"), (1_000_000, "1M"), (2_000_000, "2M")]
    delays_hold = [
        ("5d_30m",  "5d30m",  "short_2h"),
        ("5d_30m",  "5d30m",  "short_90m"),
        ("5d_30m",  "5d30m",  "short_1h"),
        ("10d_30m", "10d30m", "short_2h"),
        ("15d_60m", "15d60m", "short_2h"),
    ]
    btc_key_map = {
        "5d_30m":  "btc_max_5d_30m",
        "10d_30m": "btc_max_10d_30m",
        "15d_60m": "btc_max_15d_60m",
    }
    hold_label = {"short_2h": "2h", "short_90m": "90m", "short_1h": "1h"}

    base = lambda ev: base_short_elig(ev)
    for btc_thr, btc_lbl in btc_thrs:
        for bwin, bwin_lbl, outcome_key in delays_hold:
            bkey = btc_key_map[bwin]
            vals = [ev[outcome_key] for ev in events
                    if base(ev) and ev.get(bkey, 0) >= btc_thr and ev.get(outcome_key) is not None]
            name = f"short_btc{btc_lbl}_delay{bwin_lbl}_hold{hold_label[outcome_key]}"
            label = f"SHORT BTC>={btc_lbl} delay {bwin_lbl} hold {hold_label[outcome_key]}"
            results[name] = full_stat(vals, label)
            s = results[name]
            print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")

    # Combined LONG+SHORT best candidates
    for long_gate_name, long_gate_fn in [
        ("current", base_long),
        ("score1_btc7d", lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE"
                                     and not ev["blocked_us"] and ev["dow"] not in {0,2}
                                     and ev["sync_k"] < 200_000 and ev["btc7d"] < 0
                                     and ev["score"] >= 1)),
    ]:
        for btc_thr, btc_lbl, bwin in [(1_000_000,"1M","5d_30m"),(2_000_000,"2M","5d_30m")]:
            bkey = btc_key_map[bwin]
            longs = [ev["long_4h"] for ev in events if long_gate_fn(ev) and ev.get("long_4h") is not None
                     and not ev.get("noisy")]
            shorts = [ev["short_2h"] for ev in events
                      if base(ev) and ev.get(bkey, 0) >= btc_thr and ev.get("short_2h") is not None]
            combo = longs + shorts
            name = f"combo_{long_gate_name}_short_btc{btc_lbl}"
            label = f"COMBO {long_gate_name}_LONG + SHORT BTC>={btc_lbl}"
            results[name] = full_stat(combo, label)
            s = results[name]
            print(f"  {label}: N={s.get('n',0)} WR={s.get('wr')} avg={s.get('avg')} MC_p={s.get('mc_p')}")
    return results

# ── Report generation ─────────────────────────────────────────────────────────

def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep    = "|" + "|".join("---" for _ in cols) + "|"
    lines  = [header, sep]
    for r in rows:
        def fmt(k):
            v = r.get(k)
            if v is None: return "—"
            if isinstance(v, float): return f"{v:+.1f}" if abs(v) < 10_000 else f"{v:.0f}"
            return str(v)
        lines.append("| " + " | ".join(fmt(k) for k, _ in cols) + " |")
    return "\n".join(lines)

def results_to_md(all_results, date_str):
    sections = []
    sections.append(f"# S34 Next-Round Tests — {date_str}\n")
    sections.append("> Single-process backtest. MC=1000 random-sign permutation. 70/30 chronological holdout.")
    sections.append(f"> Total anchors in universe: {all_results.get('meta',{}).get('total_anchors','?')}\n")

    cols = [
        ("label","Label"),("n","N"),("wr","WR%"),("avg","Avg bps"),
        ("t3r","T3R"),("worst","Worst"),("tail_n","Tail<-100"),
        ("per_month","N/mo"),("mc_p","MC p"),
        ("ho_n","HO N"),("ho_wr","HO WR%"),("ho_avg","HO avg"),("ho_sum","HO sum"),
    ]

    for section_key, section_label in [
        ("T1","T1 — SHORT_NOISY_BTC1M"),
        ("T2","T2 — C_score_relax narrow"),
        ("T3","T3 — btc4h vs btc7d"),
        ("T4","T4 — ASIA session"),
        ("T5","T5 — Notional bands"),
        ("T6","T6 — SHORT gate expansion"),
    ]:
        data = all_results.get(section_key, {})
        if not data:
            continue
        rows = [v for v in data.values() if isinstance(v, dict) and "n" in v]
        sections.append(f"\n## {section_label}\n")
        sections.append(md_table(rows, cols))
    return "\n".join(sections)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global total_months
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("S34 Next-Round Tests V1")
    print(f"DB: {DB_PATH}")

    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        anchors = build_anchors(conn)
        if not anchors:
            print("ERROR: no anchors found"); return

        # Determine total months for per_month stat
        tss = [int(a.anchor_ts_ms) for a in anchors]
        span_days = (max(tss) - min(tss)) / 86_400_000
        total_months = max(1.0, span_days / 30.0)
        print(f"Span: {span_days:.0f} days = {total_months:.1f} months")

        marks_eth = load_mark_index(conn, "ETHUSDT")
        print("Building feature events ...")
        events = build_events(conn, anchors, marks_eth)
        print(f"  {len(events)} events with features")
        add_outcomes(events, marks_eth)

    print(f"\nRunning tests on {len(events)} events ...")
    all_results = {
        "meta": {
            "total_anchors": len(anchors),
            "total_events": len(events),
            "total_months": round(total_months, 1),
            "span_days": round(span_days, 0),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "T1": run_T1(events),
        "T2": run_T2(events),
        "T3": run_T3(events),
        "T4": run_T4(events),
        "T5": run_T5(events),
        "T6": run_T6(events),
    }

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md_content = results_to_md(all_results, date_str)

    OUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(md_content, encoding="utf-8")

    print(f"\nSaved:\n  {OUT_JSON}\n  {OUT_MD}")
    print("\nDone.")

if __name__ == "__main__":
    main()
