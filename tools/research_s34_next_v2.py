"""S34 Next Tests V2 — T1 through T9 comprehensive backtest.

T1  SEQ_noisy_with_btc1m SHORT  -- danger-state inversion, full gauntlet
T2  C_score_relax narrow        -- score+1>=2 LONG + SHORT BTC>=1M delay10
T3  Echo cascade timing         -- 2nd ETH SELL window sweep (4 windows)
T4  BTC trend gate comparison   -- btc3d vs btc4h vs btc7d vs combinations
T5  Score component LOO         -- leave-one-out, which component matters most
T6  Funding rate gate           -- crowded vs short-squeeze environment
T7  Cascade burst quality       -- rate / prebuildup filter
T8  Pre-cascade drift           -- was ETH already moving 30m before?
T9  Weekend / Friday timing     -- Friday US-close + full Sat block
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
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

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_NEXT_V2.json"
OUT_MD   = OUT_DIR / "S34_NEXT_V2.md"

ETH_THRESH   = 200_000.0
PROP_THRESH  =  50_000.0
LOOKBACK_MS  = 400 * 24 * 3600_000
FEE_BPS      = 5.0
MC_ITER      = 1000
TOTAL_MONTHS = 4.5

random.seed(42)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def mark_bps(conn, sym, ts_ms, lookback_ms):
    r0 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms - lookback_ms)).fetchone()
    r1 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0]) - float(r0[0])) / float(r0[0]) * 10_000.0
    return 0.0

def funding_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT funding_rate FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= h < 13:
        return "EUROPE"
    if 13 <= h < 21:
        return "US"
    return "OFF"

def dow_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()

def hour_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(vals, label=""):
    if not vals:
        return {"label": label, "n": 0}
    n = len(vals)
    wins = sum(1 for v in vals if v > 0)
    sv = sorted(vals)
    t3 = sorted(vals, reverse=True)
    t3r = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    avg = sum(vals) / n
    rng = random.Random(0)
    mc_p = None
    if n >= 4:
        ct = sum(1 for _ in range(MC_ITER)
                 if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / n >= avg)
        mc_p = round(ct / MC_ITER, 3)
    cut = max(1, int(n * 0.7))
    ho = vals[cut:]
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "t3r": round(t3r, 1),
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in vals if v < -100),
        "per_month": round(n / TOTAL_MONTHS, 1),
        "mc_p": mc_p,
        "ho_n": len(ho),
        "ho_avg": round(sum(ho) / len(ho), 1) if ho else None,
        "ho_wr": round(100 * sum(1 for v in ho if v > 0) / len(ho), 1) if ho else None,
        "ho_sum": round(sum(ho), 1) if ho else None,
    }

# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------

def long_out(marks, entry_ts, hold_ms):
    r0 = marks.at_or_after(entry_ts)
    r1 = marks.at_or_before(entry_ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return round((float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000.0 - FEE_BPS, 2)

def short_out(marks, entry_ts, hold_ms):
    r0 = marks.at_or_after(entry_ts)
    r1 = marks.at_or_before(entry_ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return round(-(float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000.0 - FEE_BPS, 2)

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_score(conn, ts, sync_k, n2h):
    b4h = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    sess_us = 13 <= hour < 21
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30)
            + int(sess_us) + int(sync_k >= 200_000))

def build_events(conn, anchors, marks_eth):
    events = []
    total = len(anchors)
    anchor_ts_set = sorted(int(a.anchor_ts_ms) for a in anchors)
    print("  Building features for %d anchors ..." % total)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print("    [%d/%d]" % (i + 1, total))
        ts  = int(anc.anchor_ts_ms)
        rn  = float(anc.running_notional)
        p0r = marks_eth.at_or_after(ts)
        if p0r is None:
            continue

        # Core
        sync_k  = (liq_sum(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts)
                 + liq_sum(conn, "SOLUSDT", "SELL", ts - 10*60_000, ts))
        n2h     = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
        score   = compute_score(conn, ts, sync_k, n2h)
        btc4h   = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        btc7d   = mark_bps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000)
        btc3d   = mark_bps(conn, "BTCUSDT", ts, 3 * 24 * 3600_000)
        btc1d   = mark_bps(conn, "BTCUSDT", ts, 1 * 24 * 3600_000)
        eth1h   = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        eth30m  = mark_bps(conn, "ETHUSDT", ts, 30 * 60_000)
        bull    = eth1h > 20.0 and btc4h > 50.0
        sess    = session_name(ts)
        dow     = dow_of(ts)
        hour    = hour_of(ts)
        blocked_us = (sess == "US" and hour in {13, 14})

        # Funding
        funding = funding_at(conn, "ETHUSDT", ts)

        # Cascade density / prebuildup
        density_24h = liq_cnt(conn, "ETHUSDT", "SELL",
                               ts - 24*3600_000, ts - 300_000, ETH_THRESH)
        prebuildup  = liq_cnt(conn, "ETHUSDT", "SELL",
                               ts - 30*60_000,   ts - 1000, PROP_THRESH)
        burst_5m    = liq_cnt(conn, "ETHUSDT", "SELL",
                               ts, ts + 5*60_000, PROP_THRESH)

        # Failed cascade
        p5m = marks_eth.at_or_before(ts + 5 * 60_000)
        failed_cascade = False
        if p5m and float(p0r[1]) > 0:
            failed_cascade = (float(p5m[1]) - float(p0r[1])) / float(p0r[1]) * 10_000 > 0

        # Noisy follow-on
        noisy_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                                  ts + 60_000, ts + 30*60_000, PROP_THRESH)
        noisy = (noisy_ts is not None)

        # BTC confirm windows (relative to anchor ts)
        btc_5d_30m  = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000,  ts + 30*60_000)
        btc_5d_60m  = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000,  ts + 60*60_000)
        btc_10d_30m = liq_max(conn, "BTCUSDT", "SELL", ts + 10*60_000, ts + 30*60_000)
        btc_10d_60m = liq_max(conn, "BTCUSDT", "SELL", ts + 10*60_000, ts + 60*60_000)

        # Score components individually
        b4h_val = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        book    = book_features_at(conn, "ETHUSDT", ts, 30)
        vdep    = float(book.get("vdepth_bps") or 0) if book else 0.0
        sc_n2h    = int(n2h >= 3)
        sc_btc4h  = int(b4h_val < 0)
        sc_vdepth = int(vdep >= 30)
        sc_us     = int(13 <= hour < 21)
        sc_synck  = int(sync_k >= 200_000)

        # Gate eligibility (current live rule)
        long_elig  = (not bull and sess != "EUROPE" and not blocked_us
                      and dow not in {0, 2} and sync_k < 200_000
                      and btc7d < 0 and (score + 1) >= 3)
        short_elig = (not bull and sess != "EUROPE" and dow != 6 and score >= 4)

        events.append({
            "ts": ts, "rn": rn,
            "sync_k": sync_k, "n2h": n2h, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d, "btc1d": btc1d,
            "eth1h": eth1h, "eth30m": eth30m,
            "bull": bull, "sess": sess, "dow": dow, "hour": hour, "blocked_us": blocked_us,
            "funding": funding,
            "density_24h": density_24h, "prebuildup": prebuildup, "burst_5m": burst_5m,
            "failed_cascade": failed_cascade,
            "noisy": noisy, "noisy_ts": noisy_ts,
            "btc_5d_30m": btc_5d_30m, "btc_5d_60m": btc_5d_60m,
            "btc_10d_30m": btc_10d_30m, "btc_10d_60m": btc_10d_60m,
            "sc_n2h": sc_n2h, "sc_btc4h": sc_btc4h, "sc_vdepth": sc_vdepth,
            "sc_us": sc_us, "sc_synck": sc_synck,
            "long_elig": long_elig, "short_elig": short_elig,
        })
    return events, anchor_ts_set

def add_outcomes(events, marks_eth):
    print("  Computing outcomes ...")
    for ev in events:
        ts = ev["ts"]
        ev["long_4h"]  = long_out(marks_eth, ts, 4 * 3600_000)
        ev["long_2h"]  = long_out(marks_eth, ts, 2 * 3600_000)
        ev["short_2h"] = short_out(marks_eth, ts, 2 * 3600_000)
        nt = ev.get("noisy_ts")
        ev["short_noisy_2h"] = short_out(marks_eth, nt, 2 * 3600_000) if nt else None
        ev["short_noisy_90m"] = short_out(marks_eth, nt, 90 * 60_000) if nt else None

def add_echo_flags(events, anchor_ts_set):
    """Mark events that are echo cascades (prior anchor within window)."""
    ts_list = sorted(anchor_ts_set)
    for ev in events:
        ts = ev["ts"]
        for lo_m, hi_m, key in [
            (10, 30, "echo_10_30"),
            (15, 45, "echo_15_45"),
            (20, 60, "echo_20_60"),
            (30, 90, "echo_30_90"),
        ]:
            lo_ms = ts - hi_m * 60_000
            hi_ms = ts - lo_m * 60_000
            prior = any(lo_ms <= t < hi_ms for t in ts_list if t != ts)
            ev[key] = prior

# ---------------------------------------------------------------------------
# Base filters
# ---------------------------------------------------------------------------

def base_long(ev):
    """Current live LONG gate (score+1>=3 means score>=2)."""
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
            and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
            and ev["btc7d"] < 0 and ev["score"] >= 2)

def base_short(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] != 6 and ev["score"] >= 4)

# ---------------------------------------------------------------------------
# T1 — SEQ_noisy_with_btc1m SHORT inversion
# ---------------------------------------------------------------------------

def run_T1(events):
    print("\n=== T1: SEQ_noisy_with_btc1m SHORT inversion ===")
    R = {}

    # Baseline: all noisy events SHORT at noisy entry, 2h hold
    all_noisy = [ev["short_noisy_2h"] for ev in events
                 if ev["noisy"] and ev.get("short_noisy_2h") is not None]
    R["T1_baseline_all_noisy"] = stats(all_noisy, "SHORT_NOISY all (no BTC filter) 2h")

    # BTC threshold sweep with 5min delay
    for thr, lbl in [(500_000,"500K"), (1_000_000,"1M"), (2_000_000,"2M")]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= thr]
        R["T1_btc%s_delay5" % lbl] = stats(vals, "SHORT_NOISY BTC>=%s delay5m 2h" % lbl)

    # BTC threshold sweep with 10min delay
    for thr, lbl in [(500_000,"500K"), (1_000_000,"1M"), (2_000_000,"2M")]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_10d_30m"] >= thr]
        R["T1_btc%s_delay10" % lbl] = stats(vals, "SHORT_NOISY BTC>=%s delay10m 2h" % lbl)

    # BTC>=1M delay5 + score gate
    for sc in [3, 4]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= 1_000_000 and ev["score"] >= sc]
        R["T1_btc1m_d5_sc%d" % sc] = stats(vals, "SHORT_NOISY BTC>=1M d5 score>=%d 2h" % sc)

    # BTC>=1M delay5 + 90min hold
    vals = [ev["short_noisy_90m"] for ev in events
            if ev["noisy"] and ev.get("short_noisy_90m") is not None
            and ev["btc_5d_30m"] >= 1_000_000]
    R["T1_btc1m_d5_90m"] = stats(vals, "SHORT_NOISY BTC>=1M delay5m 90m hold")

    # BTC>=1M delay5 + session filter (not EUROPE already; add US-only)
    vals_us = [ev["short_noisy_2h"] for ev in events
               if ev["noisy"] and ev.get("short_noisy_2h") is not None
               and ev["btc_5d_30m"] >= 1_000_000 and ev["sess"] == "US"]
    R["T1_btc1m_d5_us_only"] = stats(vals_us, "SHORT_NOISY BTC>=1M delay5m US only 2h")

    # BTC>=1M delay10, 60min window
    vals = [ev["short_noisy_2h"] for ev in events
            if ev["noisy"] and ev.get("short_noisy_2h") is not None
            and ev["btc_10d_60m"] >= 1_000_000]
    R["T1_btc1m_d10_60w"] = stats(vals, "SHORT_NOISY BTC>=1M delay10m 60m-window 2h")

    # Best combined candidate
    vals = [ev["short_noisy_2h"] for ev in events
            if ev["noisy"] and ev.get("short_noisy_2h") is not None
            and ev["btc_5d_30m"] >= 1_000_000
            and not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] != 6]
    R["T1_best_combo"] = stats(vals, "SHORT_NOISY BTC>=1M d5 not_bull not_EU not_sat 2h")

    print("  T1 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-35s N=%-4d WR=%-6s avg=%-8s T3R=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps",
                str(v["t3r"]), v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T2 — C_score_relax_short1m10 narrow gauntlet
# ---------------------------------------------------------------------------

def run_T2(events):
    print("\n=== T2: C_score_relax narrow gauntlet (score+1>=2 LONG + SHORT BTC1M d10) ===")
    R = {}

    def long_relax(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                and ev["btc7d"] < 0 and ev["score"] >= 1)

    # LONG relaxed leg alone
    vals_l = [ev["long_4h"] for ev in events
              if long_relax(ev) and ev.get("long_4h") is not None]
    R["T2_long_relax"] = stats(vals_l, "LONG score>=1 (relaxed) 4h")

    # Current LONG (score>=2) for reference
    vals_cur = [ev["long_4h"] for ev in events
                if base_long(ev) and ev.get("long_4h") is not None]
    R["T2_current_long"] = stats(vals_cur, "LONG score>=2 (current) 4h")

    # Added-only: relax adds N vs current
    cur_ts = {ev["ts"] for ev in events if base_long(ev)}
    vals_added = [ev["long_4h"] for ev in events
                  if long_relax(ev) and ev["ts"] not in cur_ts
                  and ev.get("long_4h") is not None]
    R["T2_long_added_only"] = stats(vals_added, "LONG score=1 added-only 4h")

    # SHORT BTC>=1M delay10 alone
    vals_s = [ev["short_2h"] for ev in events
              if base_short(ev) and ev.get("short_2h") is not None
              and ev["btc_10d_30m"] >= 1_000_000]
    R["T2_short_btc1m_d10"] = stats(vals_s, "SHORT BTC>=1M delay10m 2h")

    # Combined: relaxed LONG + SHORT BTC1M d10
    combined = []
    seen_ts = set()
    for ev in events:
        if long_relax(ev) and ev.get("long_4h") is not None and ev["ts"] not in seen_ts:
            combined.append(ev["long_4h"]); seen_ts.add(ev["ts"])
        if (base_short(ev) and ev.get("short_2h") is not None
                and ev["btc_10d_30m"] >= 1_000_000 and ev["ts"] not in seen_ts):
            combined.append(ev["short_2h"]); seen_ts.add(ev["ts"])
    R["T2_combined_relax_short1m10"] = stats(combined, "COMBINED score_relax_LONG + SHORT_BTC1M_d10")

    # Score=1 specifically (the new events added by relax)
    vals_sc1 = [ev["long_4h"] for ev in events
                if long_relax(ev) and ev["score"] == 1 and ev.get("long_4h") is not None]
    R["T2_long_score1_only"] = stats(vals_sc1, "LONG score=1 events only 4h")

    print("  T2 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s T3R=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps",
                str(v["t3r"]), v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T3 — Echo cascade timing optimization
# ---------------------------------------------------------------------------

def run_T3(events):
    print("\n=== T3: Echo cascade timing (prior ETH SELL >=200K window sweep) ===")
    R = {}

    for key, lbl in [
        ("echo_10_30", "10-30min"),
        ("echo_15_45", "15-45min"),
        ("echo_20_60", "20-60min"),
        ("echo_30_90", "30-90min"),
    ]:
        # LONG at echo anchor
        vals = [ev["long_4h"] for ev in events
                if ev.get(key) and ev.get("long_4h") is not None]
        R["T3_echo_%s_long4h" % key] = stats(vals, "ECHO %s LONG 4h (all)" % lbl)

        # LONG + base_long gate on echo
        vals_g = [ev["long_4h"] for ev in events
                  if ev.get(key) and base_long(ev) and ev.get("long_4h") is not None]
        R["T3_echo_%s_gated" % key] = stats(vals_g, "ECHO %s LONG 4h (gated)" % lbl)

        # SHORT at echo (price exhaustion reversal)
        vals_s = [ev["short_2h"] for ev in events
                  if ev.get(key) and ev.get("short_2h") is not None]
        R["T3_echo_%s_short2h" % key] = stats(vals_s, "ECHO %s SHORT 2h (all)" % lbl)

    # Best echo direction: silence (not noisy) vs noisy
    for key in ["echo_15_45", "echo_20_60"]:
        vals_sil = [ev["long_4h"] for ev in events
                    if ev.get(key) and not ev["noisy"] and ev.get("long_4h") is not None]
        R["T3_%s_silence_long" % key] = stats(vals_sil, "ECHO %s + silence LONG 4h" % key)
        vals_noisy = [ev["long_4h"] for ev in events
                      if ev.get(key) and ev["noisy"] and ev.get("long_4h") is not None]
        R["T3_%s_noisy_long" % key] = stats(vals_noisy, "ECHO %s + noisy LONG 4h" % key)

    print("  T3 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T4 — BTC trend gate comparison
# ---------------------------------------------------------------------------

def run_T4(events):
    print("\n=== T4: BTC trend gate comparison (btc3d vs btc4h vs btc7d) ===")
    R = {}

    def base_nodow_nosync(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE"
                and not ev["blocked_us"] and ev["dow"] not in {0, 2}
                and ev["sync_k"] < 200_000 and ev["score"] >= 2)

    for gname, gfn in [
        ("no_btc_gate",    lambda ev: base_nodow_nosync(ev)),
        ("btc7d_lt0",      lambda ev: base_nodow_nosync(ev) and ev["btc7d"] < 0),
        ("btc3d_lt0",      lambda ev: base_nodow_nosync(ev) and ev["btc3d"] < 0),
        ("btc4h_lt0",      lambda ev: base_nodow_nosync(ev) and ev["btc4h"] < 0),
        ("btc1d_lt0",      lambda ev: base_nodow_nosync(ev) and ev["btc1d"] < 0),
        ("btc3d_OR_btc4h", lambda ev: base_nodow_nosync(ev) and (ev["btc3d"] < 0 or ev["btc4h"] < 0)),
        ("btc4h_OR_btc7d", lambda ev: base_nodow_nosync(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("btc3d_AND_btc4h",lambda ev: base_nodow_nosync(ev) and ev["btc3d"] < 0 and ev["btc4h"] < 0),
        ("btc4h_AND_btc7d",lambda ev: base_nodow_nosync(ev) and ev["btc4h"] < 0 and ev["btc7d"] < 0),
        ("all_three_lt0",  lambda ev: base_nodow_nosync(ev) and ev["btc3d"] < 0 and ev["btc4h"] < 0 and ev["btc7d"] < 0),
    ]:
        vals = [ev["long_4h"] for ev in events
                if gfn(ev) and ev.get("long_4h") is not None]
        R["T4_%s" % gname] = stats(vals, "LONG gate=%s 4h" % gname)

    # btc7d threshold sweep
    for thr in [-500, -200, -100, 0, 100, 500]:
        op = "lt" if thr >= 0 else "lt_neg"
        vals = [ev["long_4h"] for ev in events
                if base_nodow_nosync(ev) and ev["btc7d"] < thr
                and ev.get("long_4h") is not None]
        R["T4_btc7d_lt%d" % thr] = stats(vals, "LONG btc7d<%d 4h" % thr)

    print("  T4 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-35s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T5 — Score component leave-one-out
# ---------------------------------------------------------------------------

def run_T5(events):
    print("\n=== T5: Score component leave-one-out ===")
    R = {}

    # Base: current LONG (btc7d<0 + no dow 0/2 + no EUROPE + no US1314 + sync<200K)
    def base_nosc(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000 and ev["btc7d"] < 0)

    # Score splits: how many events are won by each component
    for comp, ckey in [
        ("n2h>=3",     "sc_n2h"),
        ("btc4h<0",    "sc_btc4h"),
        ("vdepth>=30", "sc_vdepth"),
        ("sess=US",    "sc_us"),
        ("sync_k>=200K","sc_synck"),
    ]:
        vals_has = [ev["long_4h"] for ev in events
                    if base_nosc(ev) and ev["score"] >= 2 and ev[ckey] == 1
                    and ev.get("long_4h") is not None]
        vals_no  = [ev["long_4h"] for ev in events
                    if base_nosc(ev) and ev["score"] >= 2 and ev[ckey] == 0
                    and ev.get("long_4h") is not None]
        R["T5_%s_present" % ckey] = stats(vals_has, "LONG score>=2 + %s present 4h" % comp)
        R["T5_%s_absent"  % ckey] = stats(vals_no,  "LONG score>=2 + %s absent 4h"  % comp)

    # Score threshold sweep (how N and WR change)
    for sc_thr in [1, 2, 3, 4, 5]:
        vals = [ev["long_4h"] for ev in events
                if base_nosc(ev) and ev["score"] >= sc_thr
                and ev.get("long_4h") is not None]
        R["T5_score_gte%d" % sc_thr] = stats(vals, "LONG score>=%d 4h" % sc_thr)

    # Remove individual gate entirely (LOO on gate conditions)
    full_base = [ev["long_4h"] for ev in events
                 if base_long(ev) and ev.get("long_4h") is not None]
    R["T5_full_base"] = stats(full_base, "Full base_long (reference)")

    # Remove btc7d gate (keep all else)
    no_btc7d = [ev["long_4h"] for ev in events
                if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                    and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000 and ev["score"] >= 2)
                and ev.get("long_4h") is not None]
    R["T5_remove_btc7d"] = stats(no_btc7d, "LONG remove btc7d gate 4h")

    # Remove dow block
    no_dow = [ev["long_4h"] for ev in events
              if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                  and ev["sync_k"] < 200_000 and ev["btc7d"] < 0 and ev["score"] >= 2)
              and ev.get("long_4h") is not None]
    R["T5_remove_dow"] = stats(no_dow, "LONG remove dow{0,2} block 4h")

    # Remove US 13-14 block
    no_us1314 = [ev["long_4h"] for ev in events
                 if (not ev["bull"] and ev["sess"] != "EUROPE"
                     and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
                     and ev["btc7d"] < 0 and ev["score"] >= 2)
                 and ev.get("long_4h") is not None]
    R["T5_remove_us1314"] = stats(no_us1314, "LONG remove US13-14 block 4h")

    print("  T5 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T6 — Funding rate gate
# ---------------------------------------------------------------------------

def run_T6(events):
    print("\n=== T6: Funding rate gate ===")
    R = {}

    # Distribution of funding
    fundings = [ev["funding"] for ev in events]
    fund_pct5  = sorted(fundings)[int(len(fundings)*0.05)]
    fund_pct95 = sorted(fundings)[int(len(fundings)*0.95)]
    print("    Funding range: p5=%.5f p95=%.5f" % (fund_pct5, fund_pct95))

    # LONG: effect of funding direction
    for label, fn in [
        ("fund_pos",      lambda ev: ev["funding"] > 0),
        ("fund_neg",      lambda ev: ev["funding"] < 0),
        ("fund_gt001pct", lambda ev: ev["funding"] > 0.0001),
        ("fund_gt003pct", lambda ev: ev["funding"] > 0.0003),
        ("fund_lt0",      lambda ev: ev["funding"] < 0),
        ("fund_lt_neg001",lambda ev: ev["funding"] < -0.0001),
    ]:
        vals_l = [ev["long_4h"] for ev in events
                  if base_long(ev) and fn(ev) and ev.get("long_4h") is not None]
        R["T6_long_%s" % label] = stats(vals_l, "LONG %s 4h" % label)

        vals_s = [ev["short_2h"] for ev in events
                  if base_short(ev) and fn(ev) and ev.get("short_2h") is not None
                  and ev["btc_5d_30m"] >= 2_000_000]
        R["T6_short_%s" % label] = stats(vals_s, "SHORT BTC2M %s 2h" % label)

    # LONG without funding filter (reference)
    vals_ref = [ev["long_4h"] for ev in events
                if base_long(ev) and ev.get("long_4h") is not None]
    R["T6_long_ref"] = stats(vals_ref, "LONG no funding filter (ref)")

    print("  T6 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-35s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T7 — Cascade burst quality
# ---------------------------------------------------------------------------

def run_T7(events):
    print("\n=== T7: Cascade burst quality (prebuildup + burst_5m) ===")
    R = {}

    # prebuildup sweep (count of >=50K ETH SELL in 30min before cascade)
    for pb_thr in [0, 1, 2, 3, 5]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["prebuildup"] >= pb_thr
                and ev.get("long_4h") is not None]
        R["T7_prebuildup_gte%d" % pb_thr] = stats(
            vals, "LONG prebuildup>=%d 4h" % pb_thr)

    # prebuildup = 0 (clean cascade, no prior buildup)
    vals_clean = [ev["long_4h"] for ev in events
                  if base_long(ev) and ev["prebuildup"] == 0
                  and ev.get("long_4h") is not None]
    R["T7_prebuildup_zero"] = stats(vals_clean, "LONG prebuildup=0 (clean) 4h")

    # burst_5m sweep (fast cascade in first 5min)
    for b5 in [0, 1, 2, 3]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["burst_5m"] >= b5
                and ev.get("long_4h") is not None]
        R["T7_burst5m_gte%d" % b5] = stats(vals, "LONG burst_5m>=%d 4h" % b5)

    # Running notional bands
    for lo_n, hi_n, lbl in [(200_000,400_000,"200-400K"), (400_000,700_000,"400-700K"),
                             (700_000,1_200_000,"700K-1.2M"), (1_200_000,999e9,"1.2M+")]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and lo_n <= ev["rn"] < hi_n
                and ev.get("long_4h") is not None]
        R["T7_rn_%s" % lbl.replace("-","_")] = stats(vals, "LONG rn=%s 4h" % lbl)

    # density_24h (how many prior cascades in 24h)
    for d24 in [0, 1, 2, 3]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["density_24h"] >= d24
                and ev.get("long_4h") is not None]
        R["T7_density24h_gte%d" % d24] = stats(vals, "LONG density_24h>=%d 4h" % d24)

    # DOUBLE_CASCADE: density_24h>=1 + prebuildup>=2 (from mega)
    vals_dc = [ev["long_4h"] for ev in events
               if base_long(ev) and ev["density_24h"] >= 1 and ev["prebuildup"] >= 2
               and ev.get("long_4h") is not None]
    R["T7_double_cascade"] = stats(vals_dc, "LONG DOUBLE_CASCADE (d24>=1+pb>=2) 4h")

    print("  T7 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T8 — Pre-cascade drift
# ---------------------------------------------------------------------------

def run_T8(events):
    print("\n=== T8: Pre-cascade drift (eth30m) ===")
    R = {}

    # Distribution
    eth30m_vals = [ev["eth30m"] for ev in events]
    s = sorted(eth30m_vals)
    p25 = s[int(len(s)*0.25)]
    p75 = s[int(len(s)*0.75)]
    print("    eth30m range: min=%.1f p25=%.1f med=%.1f p75=%.1f max=%.1f" % (
        s[0], p25, s[len(s)//2], p75, s[-1]))

    # LONG: split by pre-cascade drift
    for label, fn in [
        ("already_falling_hard", lambda ev: ev["eth30m"] < -30),
        ("already_falling",      lambda ev: ev["eth30m"] < -10),
        ("flat",                 lambda ev: -10 <= ev["eth30m"] <= 10),
        ("already_rising",       lambda ev: ev["eth30m"] > 10),
        ("already_rising_hard",  lambda ev: ev["eth30m"] > 30),
    ]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and fn(ev) and ev.get("long_4h") is not None]
        R["T8_long_%s" % label] = stats(vals, "LONG %s 4h" % label)

    # SHORT: does already-rising ETH predict SHORT opportunity?
    for label, fn in [
        ("rising_hard",  lambda ev: ev["eth30m"] > 30),
        ("rising",       lambda ev: ev["eth30m"] > 10),
        ("falling_hard", lambda ev: ev["eth30m"] < -30),
    ]:
        vals = [ev["short_2h"] for ev in events
                if base_short(ev) and fn(ev) and ev.get("short_2h") is not None
                and ev["btc_5d_30m"] >= 2_000_000]
        R["T8_short_%s" % label] = stats(vals, "SHORT BTC2M %s 2h" % label)

    # eth30m quartile analysis on LONG base
    for lo, hi, lbl in [(s[0],p25,"Q1_falling"),(p25,s[len(s)//2],"Q2_moderate"),
                         (s[len(s)//2],p75,"Q3_flat"),(p75,s[-1],"Q4_rising")]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and lo <= ev["eth30m"] <= hi
                and ev.get("long_4h") is not None]
        R["T8_long_Q_%s" % lbl] = stats(vals, "LONG eth30m %s [%.0f,%.0f] 4h" % (lbl,lo,hi))

    print("  T8 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# T9 — Weekend / Friday timing block
# ---------------------------------------------------------------------------

def run_T9(events):
    print("\n=== T9: Weekend / Friday timing block ===")
    R = {}

    # Current Saturday block already applied (dow=5 not in {0,2} so it passes!)
    # Wait: dow in {0,2} is Mon/Wed block. Sat=5 is NOT blocked in base_long.
    # Let me verify: dow=5 events with base_long gate...

    # DOW distribution
    for d in range(7):
        dname = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]
        vals = [ev["long_4h"] for ev in events
                if ev["dow"] == d and ev.get("long_4h") is not None
                and not ev["bull"] and ev["sess"] != "EUROPE"
                and not ev["blocked_us"] and ev["sync_k"] < 200_000]
        n = len(vals)
        if n > 0:
            wr = round(100*sum(1 for v in vals if v>0)/n,1)
            avg_v = round(sum(vals)/n,1)
            print("    DOW=%s(%d): N=%d WR=%s%% avg=%sbps" % (dname, d, n, wr, avg_v))
        R["T9_dow%d_%s_base" % (d,dname)] = stats(vals, "DOW=%s (gated) 4h" % dname)

    # Full base_long by DOW
    for d in range(7):
        dname = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["dow"] == d and ev.get("long_4h") is not None]
        R["T9_base_long_dow%d" % d] = stats(vals, "base_long DOW=%s 4h" % dname)

    # Saturday full block (add dow=5 to block set)
    vals_nosat = [ev["long_4h"] for ev in events
                  if base_long(ev) and ev["dow"] not in {0, 2, 5}
                  and ev.get("long_4h") is not None]
    R["T9_block_sat"] = stats(vals_nosat, "LONG + block Sat (dow 0,2,5) 4h")

    # Friday US-close block: Friday hour >= 20
    vals_no_fri_close = [ev["long_4h"] for ev in events
                         if base_long(ev) and not (ev["dow"] == 4 and ev["hour"] >= 20)
                         and ev.get("long_4h") is not None]
    R["T9_block_fri_close"] = stats(vals_no_fri_close, "LONG + block Fri>=20UTC 4h")

    # Friday hour-by-hour
    fri_events = [ev for ev in events if ev["dow"] == 4]
    for h in range(0, 24, 3):
        vals = [ev["long_4h"] for ev in fri_events
                if base_long(ev) and h <= ev["hour"] < h+3
                and ev.get("long_4h") is not None]
        if vals:
            R["T9_fri_h%02d" % h] = stats(vals, "Fri h%02d-%02d 4h" % (h, h+3))

    # Block Sat+Sun
    vals_weekday = [ev["long_4h"] for ev in events
                    if base_long(ev) and ev["dow"] not in {0, 2, 5, 6}
                    and ev.get("long_4h") is not None]
    R["T9_weekday_only"] = stats(vals_weekday, "LONG weekday only (block Mon,Wed,Sat,Sun) 4h")

    # Block Mon+Wed+Sat (add Sat to existing)
    vals_nosatmon = [ev["long_4h"] for ev in events
                     if (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
                         and ev["dow"] not in {0, 2, 5} and ev["sync_k"] < 200_000
                         and ev["btc7d"] < 0 and ev["score"] >= 2)
                     and ev.get("long_4h") is not None]
    R["T9_block_mon_wed_sat"] = stats(vals_nosatmon, "LONG block Mon+Wed+Sat 4h")

    print("  T9 done. Results:")
    for k, v in R.items():
        if v["n"] > 0:
            print("    %-40s N=%-4d WR=%-6s avg=%-8s mc_p=%s" % (
                k, v["n"], str(v["wr"])+"%", str(v["avg"])+"bps", v.get("mc_p","?")))
    return R

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def to_md(all_results: dict) -> str:
    lines = ["# S34 Next Tests V2\n"]
    for section, R in all_results.items():
        lines.append("## %s\n" % section)
        lines.append("| Key | N | WR | Avg bps | T3R | MC p | HO avg | HO wr |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in R.items():
            if v["n"] == 0:
                continue
            lines.append("| %s | %d | %s%% | %s | %s | %s | %s | %s |" % (
                k, v["n"], v.get("wr","?"), v.get("avg","?"),
                v.get("t3r","?"), v.get("mc_p","?"),
                v.get("ho_avg","?"), str(v.get("ho_wr","?"))+"%" if v.get("ho_wr") is not None else "?"))
        lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("S34 Next Tests V2 — T1 through T9")
    from datetime import datetime, timezone as _tz
    now_ms = int(datetime.now(_tz.utc).timestamp() * 1000)
    start_ms = now_ms - LOOKBACK_MS

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        print("Loading liquidations + marks ...")
        liqs_eth = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        print("Reconstructing ETH SELL anchors ...")
        anchors = reconstruct_anchors(
            liqs_eth, bucket_sec=300, min_gap_sec=900,
            thresholds=(ETH_THRESH,), accel_window_sec=30)
        ts_span = sorted(int(a.anchor_ts_ms) for a in anchors)
        span_days = (ts_span[-1] - ts_span[0]) / 86_400_000 if len(ts_span) > 1 else 1
        global TOTAL_MONTHS
        TOTAL_MONTHS = max(1.0, span_days / 30.0)
        print("  %d anchors | %.0f days = %.1f months" % (len(anchors), span_days, TOTAL_MONTHS))

        print("Building events ...")
        events, anchor_ts_set = build_events(conn, anchors, marks_eth)
        print("  Events built: %d" % len(events))
        add_outcomes(events, marks_eth)
    add_echo_flags(events, anchor_ts_set)

    print("\nRunning tests ...")
    all_results = {}
    all_results["T1_SEQ_NOISY_SHORT"]    = run_T1(events)
    all_results["T2_SCORE_RELAX_NARROW"] = run_T2(events)
    all_results["T3_ECHO_CASCADE"]       = run_T3(events)
    all_results["T4_BTC_TREND_GATE"]     = run_T4(events)
    all_results["T5_SCORE_LOO"]          = run_T5(events)
    all_results["T6_FUNDING_RATE"]       = run_T6(events)
    all_results["T7_BURST_QUALITY"]      = run_T7(events)
    all_results["T8_PRECASCADE_DRIFT"]   = run_T8(events)
    all_results["T9_WEEKEND_TIMING"]     = run_T9(events)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(to_md(all_results), encoding="utf-8")
    print("\nDone. Output:")
    print("  %s" % OUT_JSON)
    print("  %s" % OUT_MD)

if __name__ == "__main__":
    main()
