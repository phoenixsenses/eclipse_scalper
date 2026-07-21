"""S34 Mega Research Script — A through F comprehensive backtest.

All ideas from the brainstorm session, single process, sequential.

Sections:
  A: Gate variations (A1-A5)
  B: Frequency expansion (B1-B7)
  C: New data dimensions (C1-C7)
  D: New signal structures (D1-D8)
  E: Exit / hold optimization (E1-E5)
  F: Deep dives (F1-F8)

Output:
  reports/research/s34/S34_MEGA_V1.json
  reports/research/s34/S34_MEGA_V1.md
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
OUT_JSON = OUT_DIR / "S34_MEGA_V1.json"
OUT_MD   = OUT_DIR / "S34_MEGA_V1.md"

ETH_THRESH   = 200_000.0
PROP_THRESH  =  50_000.0
LOOKBACK_MS  = 400 * 24 * 3600_000
FEE_BPS      = 5.0
MC_ITER      = 1000

total_months = 4.5
random.seed(42)

# ── DB helpers ────────────────────────────────────────────────────────────────

def _scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def mark_bps(conn, sym, ts_ms, lookback_ms):
    r0 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms - lookback_ms)).fetchone()
    r1 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0]) - float(r0[0])) / float(r0[0]) * 10_000.0
    return 0.0

def funding_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT funding_rate, next_funding_time_ms FROM mark_prices WHERE symbol=? AND ts_ms<=?"
        " ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    if row:
        return float(row[0] or 0), int(row[1] or 0)
    return 0.0, 0

def spread_bps_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT spread_pct FROM book_ticker WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    return float(row[0]) * 10_000.0 if row and row[0] else 0.0

def imbalance_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT book_imbalance FROM book_ticker WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    return float(row[0]) if row and row[0] else 0.0

def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")

def dow_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()

def hour_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour

def minute_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).minute

# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(vals, label=""):
    if not vals:
        return {"label": label, "n": 0}
    n = len(vals)
    wins = sum(1 for v in vals if v > 0)
    sv = sorted(vals)
    t3 = sorted(vals, reverse=True)
    t3r = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    avg = sum(vals) / n
    # Monte Carlo permutation
    rng = random.Random(0)
    mc_p = None
    if n >= 4:
        ct = sum(1 for _ in range(MC_ITER)
                 if sum(rng.choice([-1,1]) * abs(v) for v in vals) / n >= avg)
        mc_p = round(ct / MC_ITER, 3)
    # 70/30 chronological holdout
    cut = max(1, int(n * 0.7))
    ho = vals[cut:]
    ho_avg = round(sum(ho)/len(ho), 1) if ho else None
    ho_wr  = round(100*sum(1 for v in ho if v>0)/len(ho), 1) if ho else None
    ho_sum = round(sum(ho), 1) if ho else None
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "t3r": round(t3r, 1),
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in vals if v < -100),
        "per_month": round(n / total_months, 1),
        "mc_p": mc_p,
        "ho_n": len(ho), "ho_avg": ho_avg, "ho_wr": ho_wr, "ho_sum": ho_sum,
    }

# ── Outcome helpers ───────────────────────────────────────────────────────────

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

# ── Feature builder ───────────────────────────────────────────────────────────

def compute_score(conn, ts, sync_k, n2h):
    b4h = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or book.get("bid_depth_usd", 0) / 1000 if book else 0) if book else 0.0
    hour = hour_of(ts)
    sess_us = 13 <= hour < 21
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30) + int(sess_us) + int(sync_k >= 200_000))

def build_events(conn, anchors, marks_eth, anchor_set_ts=None):
    """Build rich event list. anchor_set_ts: set of anchor ts used for density."""
    events = []
    total = len(anchors)
    prev_anchor_ts = []  # sorted list for density
    print("  Building features for %d anchors ..." % total)
    for i, anc in enumerate(anchors):
        if (i+1) % 200 == 0:
            print("    [%d/%d]" % (i+1, total))
        ts  = int(anc.anchor_ts_ms)
        rn  = float(anc.running_notional)
        accel = str(getattr(anc, 'acceleration_bucket', '') or '')
        p0r = marks_eth.at_or_after(ts)
        if p0r is None:
            continue

        # Core features
        sync_k = (liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
                + liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts))
        n2h = liq_cnt(conn,"ETHUSDT","SELL",ts-2*3600_000,ts-1000,PROP_THRESH)
        score = compute_score(conn, ts, sync_k, n2h)
        btc4h = mark_bps(conn, "BTCUSDT", ts, 4*3600_000)
        btc7d = mark_bps(conn, "BTCUSDT", ts, 7*24*3600_000)
        btc3d = mark_bps(conn, "BTCUSDT", ts, 3*24*3600_000)
        btc5m = mark_bps(conn, "BTCUSDT", ts, 5*60_000)
        eth1h = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        btc1h = mark_bps(conn, "BTCUSDT", ts, 3600_000)
        bull  = eth1h > 20.0 and btc4h > 50.0
        sess  = session_name(ts)
        dow   = dow_of(ts)
        hour  = hour_of(ts)
        minute = minute_of(ts)
        blocked_us = (sess == "US" and hour in {13,14})

        # Funding
        funding, next_fund_ms = funding_at(conn, "ETHUSDT", ts)
        mins_to_fund = (next_fund_ms - ts) / 60_000 if next_fund_ms > ts else 999

        # Book
        spread = spread_bps_at(conn, "ETHUSDT", ts)
        imbalance = imbalance_at(conn, "ETHUSDT", ts)

        # Cascade density: count prior-24h ETH SELL >=200K events
        density_24h = liq_cnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,ETH_THRESH)

        # Pre-buildup: >=50K events in 30min before cascade
        prebuildup = liq_cnt(conn,"ETHUSDT","SELL",ts-30*60_000,ts-1000,PROP_THRESH)

        # Failed cascade: fiyat T+5min orijinalin uzerinde mi?
        p5m = marks_eth.at_or_before(ts + 5*60_000)
        failed_cascade = False
        if p5m and float(p0r[1]) > 0:
            move5m = (float(p5m[1]) - float(p0r[1])) / float(p0r[1]) * 10_000.0
            failed_cascade = move5m > 0  # SELL cascade ama fiyat yukari gitti = fake

        # Exhaustion: 60min sonra hic >=50K yok
        after_liqs = liq_cnt(conn,"ETHUSDT","SELL",ts+1000,ts+60*60_000,PROP_THRESH)
        exhaustion = (after_liqs == 0) and (rn >= 500_000)

        # Noisy follow-on
        noisy_ts = liq_first_ts(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP_THRESH)
        noisy = (noisy_ts is not None)

        # SOL leading: SOL >=100K in 10min before ETH cascade
        sol_lead = liq_max(conn,"SOLUSDT","SELL",ts-10*60_000,ts) >= 100_000

        # BTC confirms at various thresholds/delays
        btc_5d_30m  = liq_max(conn,"BTCUSDT","SELL",ts+5*60_000,ts+30*60_000)
        btc_5d_60m  = liq_max(conn,"BTCUSDT","SELL",ts+5*60_000,ts+60*60_000)
        btc_10d_30m = liq_max(conn,"BTCUSDT","SELL",ts+10*60_000,ts+30*60_000)

        # Vol regime: std of last 6 x 4h marks (24h)
        vol_marks = []
        for k in range(1, 7):
            r = marks_eth.at_or_before(ts - k*4*3600_000)
            if r:
                vol_marks.append(float(r[1]))
        vol_regime = 0.0
        if len(vol_marks) >= 3:
            mn = sum(vol_marks)/len(vol_marks)
            vol_regime = math.sqrt(sum((v-mn)**2 for v in vol_marks)/len(vol_marks)) / mn * 10_000

        # Round-number timing
        near_round = (minute % 30) < 5 or (minute % 30) > 25

        # Gate eligibility
        long_elig = (not bull and sess != "EUROPE" and not blocked_us
                     and dow not in {0,2} and sync_k < 200_000
                     and btc7d < 0 and (score+1) >= 3)
        short_elig = (not bull and sess != "EUROPE" and dow != 6 and score >= 4)

        events.append({
            "ts": ts, "rn": rn, "accel": accel,
            "sync_k": sync_k, "n2h": n2h, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d, "btc5m": btc5m,
            "eth1h": eth1h, "btc1h": btc1h,
            "bull": bull, "sess": sess, "dow": dow, "hour": hour, "minute": minute,
            "blocked_us": blocked_us,
            "funding": funding, "mins_to_fund": mins_to_fund,
            "spread": spread, "imbalance": imbalance,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "failed_cascade": failed_cascade, "exhaustion": exhaustion,
            "noisy": noisy, "noisy_ts": noisy_ts,
            "sol_lead": sol_lead,
            "btc_5d_30m": btc_5d_30m, "btc_5d_60m": btc_5d_60m,
            "btc_10d_30m": btc_10d_30m,
            "vol_regime": vol_regime, "near_round": near_round,
            "long_elig": long_elig, "short_elig": short_elig,
        })
    return events

def add_outcomes(events, marks_eth, marks_btc):
    print("  Computing outcomes ...")
    for ev in events:
        ts = ev["ts"]
        ev["long_4h"]    = long_out(marks_eth, ts, 4*3600_000)
        ev["long_2h"]    = long_out(marks_eth, ts, 2*3600_000)
        ev["long_1h"]    = long_out(marks_eth, ts, 3600_000)
        ev["short_2h"]   = short_out(marks_eth, ts, 2*3600_000)
        ev["short_90m"]  = short_out(marks_eth, ts, 90*60_000)
        ev["short_1h"]   = short_out(marks_eth, ts, 3600_000)
        ev["partial_42"] = None  # partial exit: 50% at 2h + 50% at 4h
        l2 = ev["long_2h"]; l4 = ev["long_4h"]
        if l2 is not None and l4 is not None:
            raw2 = l2 + FEE_BPS; raw4 = l4 + FEE_BPS
            ev["partial_42"] = round((raw2 + raw4) / 2 - FEE_BPS, 2)
        # Noisy entry outcomes
        nt = ev.get("noisy_ts")
        ev["short_noisy_2h"] = short_out(marks_eth, nt, 2*3600_000) if nt else None
        ev["short_noisy_1h"] = short_out(marks_eth, nt, 3600_000) if nt else None

# ── Base filters ──────────────────────────────────────────────────────────────

def base_long(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked_us"]
            and ev["dow"] not in {0,2} and ev["sync_k"] < 200_000
            and ev["btc7d"] < 0 and ev["score"] >= 2)

def base_short(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and ev["dow"] != 6 and ev["score"] >= 4)

# ── Section A: Gate variations ────────────────────────────────────────────────

def run_A(events):
    print("\n=== SECTION A: Gate Variations ===")
    R = {}
    L = lambda ev, o="long_4h": ev.get(o)

    # A1: SHORT_NOISY_BTC1M variations (from T1 — summary only)
    for thr, lbl in [(1_000_000,"1M"),(500_000,"500K"),(2_000_000,"2M")]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= thr]
        R["A1_short_noisy_btc%s"%lbl] = stats(vals, "SHORT_NOISY BTC>=%s 2h"%lbl)

    # A1b: SHORT_NOISY + score gate
    for sc in [3,4]:
        vals = [ev["short_noisy_2h"] for ev in events
                if ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= 1_000_000 and ev["score"] >= sc]
        R["A1_short_noisy_btc1m_sc%d"%sc] = stats(vals, "SHORT_NOISY BTC>=1M score>=%d"%sc)

    # A2: SHORT BTC 2M -> 1M
    for thr, lbl in [(2_000_000,"2M"),(1_000_000,"1M"),(500_000,"500K")]:
        vals = [ev["short_2h"] for ev in events
                if base_short(ev) and ev.get("short_2h") is not None
                and ev["btc_5d_30m"] >= thr]
        R["A2_short_btc%s"%lbl] = stats(vals, "SHORT BTC>=%s delay5m 2h"%lbl)

    # A3: btc3d gate
    vals = [ev["long_4h"] for ev in events
            if base_long(ev) and ev.get("long_4h") is not None]
    R["A3_current_long"] = stats(vals, "Current LONG (btc7d<0)")
    for gate_name, gate_fn in [
        ("btc3d", lambda ev: (not ev["bull"] and ev["sess"]!="EUROPE" and not ev["blocked_us"]
                              and ev["dow"] not in {0,2} and ev["sync_k"]<200_000
                              and ev["btc3d"]<0 and ev["score"]>=2)),
        ("btc4h", lambda ev: (not ev["bull"] and ev["sess"]!="EUROPE" and not ev["blocked_us"]
                              and ev["dow"] not in {0,2} and ev["sync_k"]<200_000
                              and ev["btc4h"]<0 and ev["score"]>=2)),
    ]:
        vals = [ev["long_4h"] for ev in events if gate_fn(ev) and ev.get("long_4h") is not None]
        R["A3_%s_gate"%gate_name] = stats(vals, "LONG %s<0 gate"%gate_name)

    # A4: btc4h vs btc7d combined/or
    for name, fn in [
        ("and", lambda ev: ev["btc4h"]<0 and ev["btc7d"]<0),
        ("or",  lambda ev: ev["btc4h"]<0 or ev["btc7d"]<0),
    ]:
        base = lambda ev, fn=fn: (not ev["bull"] and ev["sess"]!="EUROPE" and not ev["blocked_us"]
                                  and ev["dow"] not in {0,2} and ev["sync_k"]<200_000
                                  and ev["score"]>=2 and fn(ev))
        vals = [ev["long_4h"] for ev in events if base(ev) and ev.get("long_4h") is not None]
        R["A4_btc4h_%s_7d"%name] = stats(vals, "LONG btc4h %s btc7d"%name)

    # A5: notional cap
    for cap in [300_000, 500_000]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev.get("long_4h") is not None and ev["rn"] <= cap]
        R["A5_rn_cap_%dk"%int(cap/1000)] = stats(vals, "LONG rn<=%dk"%int(cap/1000))

    for k, v in R.items():
        print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── Section B: Frequency expansion ───────────────────────────────────────────

def run_B(events, conn, marks_eth):
    print("\n=== SECTION B: Frequency Expansion ===")
    R = {}

    # B1: ETH BUY cascade (already loaded in events_buy - use passed buy_events arg below)
    # Skipped here - handled in separate build

    # B2: BTC lead entry - BTC >=1M cascade then wait for ETH
    # We measure: ETH events with btc_5d_30m >=1M but entered at anchor time
    # vs if we had entered 5min earlier (anchor - 5min) [price approximation]
    vals_normal = [ev["long_4h"] for ev in events if base_long(ev) and ev.get("long_4h") is not None]
    R["B2_normal_entry"] = stats(vals_normal, "B2: Normal anchor entry")

    # B3: SOL sync confirmation
    vals_sol = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["sol_lead"] and ev.get("long_4h") is not None]
    vals_nosol = [ev["long_4h"] for ev in events
                  if base_long(ev) and not ev["sol_lead"] and ev.get("long_4h") is not None]
    R["B3_sol_lead_yes"] = stats(vals_sol, "B3: LONG + SOL lead >=100K")
    R["B3_sol_lead_no"]  = stats(vals_nosol, "B3: LONG + no SOL lead")

    # B4: alt threshold 150K-200K (events would not appear in main set -- skip, need separate load)
    # Approximation: look at events where rn is low
    vals_low = [ev["long_4h"] for ev in events
                if (not ev["bull"] and ev["sess"]!="EUROPE" and not ev["blocked_us"]
                    and ev["dow"] not in {0,2} and ev["sync_k"]<200_000
                    and ev["btc7d"]<0 and ev["score"]>=2
                    and ev.get("long_4h") is not None and ev["rn"] <= 250_000)]
    R["B4_rn_200k_250k"] = stats(vals_low, "B4: rn 200K-250K (lowest band)")

    # B5: echo cascade - 2nd anchor within 45min of first
    # Flag: is this event within 45min of previous same-direction event?
    sorted_ts = sorted([ev["ts"] for ev in events])
    ts_set = set(sorted_ts)
    echo_vals, non_echo_vals = [], []
    for ev in events:
        if not base_long(ev) or ev.get("long_4h") is None: continue
        # Was there a prior anchor in the 15-45min window?
        prev_in_window = any(
            15*60_000 < (ev["ts"] - prior_ts) < 45*60_000
            for prior_ts in sorted_ts
            if prior_ts < ev["ts"]
        )
        if prev_in_window:
            echo_vals.append(ev["long_4h"])
        else:
            non_echo_vals.append(ev["long_4h"])
    R["B5_echo_cascade"]    = stats(echo_vals, "B5: Echo cascade (2nd in 45min)")
    R["B5_non_echo_cascade"]= stats(non_echo_vals, "B5: First cascade (no echo)")

    # B6: OFF session subtest
    vals_off_btc7d = [ev["long_4h"] for ev in events
                      if (not ev["bull"] and ev["sess"]=="OFF" and ev["dow"] not in {0,2}
                          and ev["sync_k"]<200_000 and ev["score"]>=2
                          and ev["btc7d"]<0 and ev.get("long_4h") is not None)]
    vals_off_btc4h = [ev["long_4h"] for ev in events
                      if (not ev["bull"] and ev["sess"]=="OFF" and ev["dow"] not in {0,2}
                          and ev["sync_k"]<200_000 and ev["score"]>=2
                          and ev["btc4h"]<0 and ev.get("long_4h") is not None)]
    R["B6_off_btc7d"] = stats(vals_off_btc7d, "B6: OFF sess + btc7d<0")
    R["B6_off_btc4h"] = stats(vals_off_btc4h, "B6: OFF sess + btc4h<0")

    # B7: partial exit simulation (50% at 2h + 50% at 4h)
    vals_full = [ev["long_4h"] for ev in events if base_long(ev) and ev.get("long_4h") is not None]
    vals_part = [ev["partial_42"] for ev in events
                 if base_long(ev) and ev.get("partial_42") is not None]
    vals_2h   = [ev["long_2h"] for ev in events if base_long(ev) and ev.get("long_2h") is not None]
    R["B7_full_4h"]    = stats(vals_full, "B7: Full 4h hold")
    R["B7_partial_2h_4h"] = stats(vals_part, "B7: Partial 50%@2h + 50%@4h")
    R["B7_full_2h"]    = stats(vals_2h, "B7: Full 2h hold")

    for k, v in R.items():
        print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── Section C: New data dimensions ───────────────────────────────────────────

def run_C(events):
    print("\n=== SECTION C: New Data Dimensions ===")
    R = {}

    # C1: funding rate gate
    # negative funding -> bearish bias -> LONG on cascade better?
    vals_neg_fund = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev.get("long_4h") is not None and ev["funding"] < -0.0001]
    vals_pos_fund = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev.get("long_4h") is not None and ev["funding"] > 0.0001]
    vals_neut_fund = [ev["long_4h"] for ev in events
                      if base_long(ev) and ev.get("long_4h") is not None
                      and -0.0001 <= ev["funding"] <= 0.0001]
    R["C1_fund_negative"] = stats(vals_neg_fund, "C1: LONG + funding<-0.01%")
    R["C1_fund_positive"] = stats(vals_pos_fund, "C1: LONG + funding>+0.01%")
    R["C1_fund_neutral"]  = stats(vals_neut_fund,"C1: LONG + funding neutral")

    # C1b: SHORT with positive funding (bearish = short squeeze risk lower)
    vals_short_pos = [ev["short_2h"] for ev in events
                      if base_short(ev) and ev.get("short_2h") is not None
                      and ev["btc_5d_30m"] >= 1_000_000 and ev["funding"] > 0.0001]
    R["C1_short_pos_fund"] = stats(vals_short_pos, "C1: SHORT BTC>=1M + fund>0.01%")

    # C2: cascade shape (accelerating vs decelerating)
    for shape in ["accelerating", "decelerating", ""]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev.get("long_4h") is not None
                and (shape == "" or shape in ev.get("accel","").lower())]
        lbl = "C2: LONG accel=%s" % (shape if shape else "all")
        R["C2_accel_%s" % (shape or "all")] = stats(vals, lbl)

    # C3: BTC/ETH notional ratio
    for min_ratio in [2, 5, 10]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev.get("long_4h") is not None
                and ev["rn"] > 0 and (ev["btc_5d_30m"] / ev["rn"]) >= min_ratio]
        R["C3_btc_eth_ratio_%d" % min_ratio] = stats(vals, "C3: LONG btc/eth ratio>=%d" % min_ratio)

    # C4: spread gate
    spread_p50 = sorted([ev["spread"] for ev in events if ev["spread"] > 0])
    p50_spread = spread_p50[len(spread_p50)//2] if spread_p50 else 5.0
    vals_tight = [ev["long_4h"] for ev in events
                  if base_long(ev) and ev.get("long_4h") is not None
                  and 0 < ev["spread"] <= p50_spread]
    vals_wide  = [ev["long_4h"] for ev in events
                  if base_long(ev) and ev.get("long_4h") is not None
                  and ev["spread"] > p50_spread]
    R["C4_spread_tight"] = stats(vals_tight, "C4: LONG spread tight (<=p50=%.1f bps)" % p50_spread)
    R["C4_spread_wide"]  = stats(vals_wide,  "C4: LONG spread wide (>p50)")

    # C5: bid depth recovery - use prebuildup as proxy for order book pressure
    vals_hi_pre = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev.get("long_4h") is not None and ev["prebuildup"] >= 3]
    vals_lo_pre = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev.get("long_4h") is not None and ev["prebuildup"] < 3]
    R["C5_prebuildup_hi"] = stats(vals_hi_pre, "C5: LONG high prebuildup (>=3)")
    R["C5_prebuildup_lo"] = stats(vals_lo_pre, "C5: LONG low prebuildup (<3)")

    # C6: hour-level (not session bucket)
    hourly = defaultdict(list)
    for ev in events:
        if base_long(ev) and ev.get("long_4h") is not None:
            hourly[ev["hour"]].append(ev["long_4h"])
    R["C6_hourly"] = {h: {"n": len(v), "wr": round(100*sum(1 for x in v if x>0)/len(v),1),
                           "avg": round(sum(v)/len(v),1)} for h, v in sorted(hourly.items())}

    # C7: ETH/BTC divergence gate
    for div_type, fn in [
        ("eth_strong", lambda ev: ev["eth1h"] > 0 and ev["btc1h"] < 0),  # ETH up BTC down -> diverge
        ("btc_strong", lambda ev: ev["btc1h"] > 0 and ev["eth1h"] < 0),
        ("both_down",  lambda ev: ev["eth1h"] < 0 and ev["btc1h"] < 0),
    ]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and fn(ev) and ev.get("long_4h") is not None]
        R["C7_%s" % div_type] = stats(vals, "C7: LONG %s" % div_type)

    for k, v in R.items():
        if isinstance(v, dict) and "n" in v:
            print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── Section D: New signal structures ─────────────────────────────────────────

def run_D(events):
    print("\n=== SECTION D: New Signal Structures ===")
    R = {}

    # D1: failed cascade (price went UP in first 5min after SELL cascade)
    vals_real = [ev["long_4h"] for ev in events
                 if base_long(ev) and not ev["failed_cascade"] and ev.get("long_4h") is not None]
    vals_fake = [ev["long_4h"] for ev in events
                 if base_long(ev) and ev["failed_cascade"] and ev.get("long_4h") is not None]
    R["D1_real_cascade"]  = stats(vals_real, "D1: LONG real cascade (price down at T+5m)")
    R["D1_failed_cascade"]= stats(vals_fake, "D1: LONG failed cascade (price up at T+5m)")
    # Can failed cascade -> SHORT?
    vals_fake_short = [ev["short_2h"] for ev in events
                       if base_long(ev) and ev["failed_cascade"] and ev.get("short_2h") is not None]
    R["D1_failed_cascade_short"] = stats(vals_fake_short, "D1: SHORT on failed cascade")

    # D2: cascade density
    for dens in [0, 1, 2, 3]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and ev["density_24h"] == dens and ev.get("long_4h") is not None]
        R["D2_dens_%d" % dens] = stats(vals, "D2: LONG density24h==%d" % dens)
    vals_dens_hi = [ev["long_4h"] for ev in events
                    if base_long(ev) and ev["density_24h"] >= 3 and ev.get("long_4h") is not None]
    R["D2_dens_3plus"] = stats(vals_dens_hi, "D2: LONG density>=3 (crowded day)")

    # D3: pre-cascade buildup
    for pre in [(0,0,"none"),(1,2,"light"),(3,9,"moderate"),(10,99,"heavy")]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and pre[0]<=ev["prebuildup"]<=pre[1] and ev.get("long_4h") is not None]
        R["D3_pre_%s" % pre[2]] = stats(vals, "D3: LONG prebuildup %s (%d-%d)" % (pre[2], pre[0], pre[1]))

    # D4: SOL lead
    vals_sol_yes = [ev["long_4h"] for ev in events
                    if base_long(ev) and ev["sol_lead"] and ev.get("long_4h") is not None]
    vals_sol_no  = [ev["long_4h"] for ev in events
                    if base_long(ev) and not ev["sol_lead"] and ev.get("long_4h") is not None]
    R["D4_sol_lead_yes"] = stats(vals_sol_yes, "D4: LONG + SOL>=100K lead")
    R["D4_sol_lead_no"]  = stats(vals_sol_no,  "D4: LONG + no SOL lead")

    # D5: cross-directional (BUY BTC while ETH SELL - would need btc buy liq data)
    # Approximation: btc5m > 0 while eth sell cascade
    vals_btc_up = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev["btc5m"] > 20 and ev.get("long_4h") is not None]
    vals_btc_dn = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev["btc5m"] < -20 and ev.get("long_4h") is not None]
    R["D5_btc5m_up_ETH_cascade"]  = stats(vals_btc_up, "D5: LONG ETH SELL but BTC rising fast")
    R["D5_btc5m_down_ETH_cascade"]= stats(vals_btc_dn, "D5: LONG ETH SELL + BTC also down")

    # D6: funding payment timing
    vals_near_fund = [ev["long_4h"] for ev in events
                      if base_long(ev) and ev["mins_to_fund"] < 30 and ev.get("long_4h") is not None]
    vals_far_fund  = [ev["long_4h"] for ev in events
                      if base_long(ev) and ev["mins_to_fund"] >= 30 and ev["mins_to_fund"] < 999 and ev.get("long_4h") is not None]
    R["D6_near_funding"] = stats(vals_near_fund, "D6: LONG <30min to funding payment")
    R["D6_far_funding"]  = stats(vals_far_fund,  "D6: LONG 30min+ to funding")

    # D7: weekend liquidity (Fri night vs Sat morning)
    vals_fri_night = [ev["long_4h"] for ev in events
                      if not ev["bull"] and ev["dow"]==4 and ev["hour"]>=20
                      and ev["sync_k"]<200_000 and ev["score"]>=2 and ev.get("long_4h") is not None]
    vals_sat_morn  = [ev["long_4h"] for ev in events
                      if not ev["bull"] and ev["dow"]==5 and ev["hour"]<12
                      and ev["sync_k"]<200_000 and ev["score"]>=2 and ev.get("long_4h") is not None]
    R["D7_fri_night"] = stats(vals_fri_night, "D7: LONG Fri night (dow=4, h>=20)")
    R["D7_sat_morn"]  = stats(vals_sat_morn,  "D7: LONG Sat morning (dow=5, h<12)")

    # D8: double cascade (silence confirmed -> second cascade within 2h)
    # Approximation: density>=2 in 2h before this anchor
    vals_double = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev["density_24h"] >= 1 and ev.get("long_4h") is not None
                   and ev["prebuildup"] >= 2]
    R["D8_double_cascade"] = stats(vals_double, "D8: LONG double cascade setup (density+prebuildup)")

    # D8b: exhaustion signal (large cascade, then silence for 60min)
    vals_exhaust = [ev["long_4h"] for ev in events
                    if base_long(ev) and ev["exhaustion"] and ev.get("long_4h") is not None]
    vals_noexhaust = [ev["long_4h"] for ev in events
                      if base_long(ev) and not ev["exhaustion"] and ev.get("long_4h") is not None]
    R["D8b_exhaustion"]    = stats(vals_exhaust, "D8b: LONG exhaustion (>=500K then 60min quiet)")
    R["D8b_no_exhaustion"] = stats(vals_noexhaust,"D8b: LONG no exhaustion")

    for k, v in R.items():
        if isinstance(v, dict) and "n" in v:
            print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── Section E: Exit optimization ─────────────────────────────────────────────

def run_E(events):
    print("\n=== SECTION E: Exit Optimization ===")
    R = {}

    # E1: Adaptive hold: silence=4h, noisy=2h, neither=3h
    vals_adapt = []
    for ev in events:
        if not base_long(ev): continue
        if ev["noisy"]:
            v = ev.get("long_2h")
        else:
            v = ev.get("long_4h")
        if v is not None:
            vals_adapt.append(v)
    vals_full4h = [ev["long_4h"] for ev in events if base_long(ev) and ev.get("long_4h") is not None]
    R["E1_adaptive_hold"] = stats(vals_adapt, "E1: Adaptive hold (noisy=2h, else=4h)")
    R["E1_full_4h"]       = stats(vals_full4h, "E1: Full 4h hold (baseline)")

    # E2: partial exit at 2h, rest at 4h
    vals_part = [ev["partial_42"] for ev in events
                 if base_long(ev) and ev.get("partial_42") is not None]
    R["E2_partial_2h4h"] = stats(vals_part, "E2: Partial 50% at 2h + 50% at 4h")

    # E3: BE->TP: if +50 bps in first 2h, exit; else hold to 4h
    vals_be = []
    for ev in events:
        if not base_long(ev): continue
        l2 = ev.get("long_2h"); l4 = ev.get("long_4h")
        if l2 is None or l4 is None: continue
        if l2 + FEE_BPS >= 50.0:  # already +50 bps at 2h (before fee adjust)
            vals_be.append(l2)
        else:
            vals_be.append(l4)
    R["E3_be_tp_50"] = stats(vals_be, "E3: BE->TP: exit at 2h if >50bps, else 4h")

    # E4: stop sweep simulation: what % of events would hit various stops
    for stop_bps in [50, 100, 150, 200]:
        # Simulate: if worst intraday bps < -stop, cap loss at -stop
        # Approximation: use worst of long_2h and compare
        capped = []
        for ev in events:
            if not base_long(ev): continue
            l4 = ev.get("long_4h")
            if l4 is None: continue
            if l4 < -(stop_bps - FEE_BPS):
                capped.append(-stop_bps)
            else:
                capped.append(l4)
        R["E4_stop_%d" % stop_bps] = stats(capped, "E4: Simulated -%d bps stop" % stop_bps)

    # E5: LONG -> noisy+BTC1M -> flip to SHORT (transition)
    vals_flip = []
    for ev in events:
        if not base_long(ev): continue
        if ev["noisy"] and ev.get("btc_5d_30m",0) >= 1_000_000:
            # Flip to SHORT from noisy entry point
            v = ev.get("short_noisy_2h")
        else:
            v = ev.get("long_4h")
        if v is not None:
            vals_flip.append(v)
    R["E5_long_flip_short_noisy"] = stats(vals_flip, "E5: LONG->flip SHORT if noisy+BTC1M")

    for k, v in R.items():
        print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── Section F: Deep dives ─────────────────────────────────────────────────────

def run_F(events, events_buy):
    print("\n=== SECTION F: Deep Dives ===")
    R = {}

    # F1: order flow imbalance gate
    # book_imbalance > 0 = bid side dominant (bullish pressure at top of book)
    vals_imb_bull = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev["imbalance"] > 0.1 and ev.get("long_4h") is not None]
    vals_imb_bear = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev["imbalance"] < -0.1 and ev.get("long_4h") is not None]
    R["F1_imbalance_bid_heavy"] = stats(vals_imb_bull, "F1: LONG + book_imbalance>0.1 (bid heavy)")
    R["F1_imbalance_ask_heavy"] = stats(vals_imb_bear, "F1: LONG + book_imbalance<-0.1 (ask heavy)")

    # F2: BTC 5min momentum gate
    vals_btc5_neg = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev["btc5m"] < 0 and ev.get("long_4h") is not None]
    vals_btc5_pos = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev["btc5m"] > 0 and ev.get("long_4h") is not None]
    R["F2_btc5m_neg"] = stats(vals_btc5_neg, "F2: LONG + BTC 5min momentum < 0")
    R["F2_btc5m_pos"] = stats(vals_btc5_pos, "F2: LONG + BTC 5min momentum > 0")

    # F3: exhaustion signal (already in D8b - reference)
    vals_exhaust = [ev["long_4h"] for ev in events
                    if base_long(ev) and ev["exhaustion"] and ev.get("long_4h") is not None]
    R["F3_exhaustion_500k"] = stats(vals_exhaust, "F3: Exhaustion (>=500K + 60min silence)")
    # Exhaustion without gate
    vals_exhaust_all = [ev["long_4h"] for ev in events
                        if not ev["bull"] and ev["exhaustion"] and ev.get("long_4h") is not None]
    R["F3_exhaustion_all_no_gate"] = stats(vals_exhaust_all, "F3: Exhaustion (no gate, all events)")

    # F4: ETH BUY cascade -> SHORT
    if events_buy:
        vals_buy_short = [ev.get("short_2h") for ev in events_buy
                          if not ev["bull"] and ev["sess"] != "EUROPE"
                          and ev["score"] >= 2 and ev.get("short_2h") is not None]
        vals_buy_short_4h = [ev.get("short_4h") for ev in events_buy
                              if not ev["bull"] and ev["sess"] != "EUROPE"
                              and ev["score"] >= 2 and ev.get("short_4h") is not None]
        R["F4_buy_cascade_short_2h"] = stats(vals_buy_short, "F4: ETH BUY cascade -> SHORT 2h")
        R["F4_buy_cascade_short_4h"] = stats(vals_buy_short_4h, "F4: ETH BUY cascade -> SHORT 4h")
    else:
        R["F4_buy_cascade_short_2h"] = {"label": "F4: ETH BUY (not loaded)", "n": 0}

    # F5: vol regime gate
    low_vol  = sorted([ev["vol_regime"] for ev in events if ev["vol_regime"] > 0])
    p33_vol  = low_vol[len(low_vol)//3] if low_vol else 100
    p67_vol  = low_vol[2*len(low_vol)//3] if low_vol else 200
    for name, lo, hi in [("low",0,p33_vol),("mid",p33_vol,p67_vol),("high",p67_vol,1e9)]:
        vals = [ev["long_4h"] for ev in events
                if base_long(ev) and lo <= ev["vol_regime"] < hi and ev.get("long_4h") is not None]
        R["F5_vol_%s" % name] = stats(vals, "F5: LONG vol_regime %s (%.0f-%.0f)" % (name, lo, hi))

    # F6: round-number timing
    vals_round    = [ev["long_4h"] for ev in events
                     if base_long(ev) and ev["near_round"] and ev.get("long_4h") is not None]
    vals_offround = [ev["long_4h"] for ev in events
                     if base_long(ev) and not ev["near_round"] and ev.get("long_4h") is not None]
    R["F6_near_round_time"] = stats(vals_round, "F6: LONG near :00 or :30 (+/-5min)")
    R["F6_off_round_time"]  = stats(vals_offround,"F6: LONG off round time")

    # F7: BTC cascade router (BTC >=1M present -> route SHORT, else LONG)
    router_vals = []
    for ev in events:
        if not ev["long_elig"]: continue
        if ev["btc_5d_30m"] >= 1_000_000 and base_short(ev):
            # Route to SHORT
            v = ev.get("short_2h")
        else:
            # Route to LONG
            v = ev.get("long_4h")
        if v is not None:
            router_vals.append(v)
    R["F7_btc_router"] = stats(router_vals, "F7: BTC cascade router (BTC1M->SHORT else LONG)")

    # F8: cross-TF state: last 4h return negative + cascade now
    vals_4h_neg = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev["btc4h"] < -50 and ev.get("long_4h") is not None]
    vals_4h_pos = [ev["long_4h"] for ev in events
                   if base_long(ev) and ev["btc4h"] > 50 and ev.get("long_4h") is not None]
    R["F8_4h_bear_cascade"]  = stats(vals_4h_neg, "F8: LONG cascade in 4h downtrend (btc4h<-50)")
    R["F8_4h_bull_pullback_cascade"] = stats(vals_4h_pos, "F8: LONG cascade in 4h uptrend (btc4h>50)")

    for k, v in R.items():
        if isinstance(v, dict) and "n" in v:
            print("  %s: N=%d WR=%s avg=%s MC_p=%s" % (v["label"], v["n"], v.get("wr"), v.get("avg"), v.get("mc_p")))
    return R

# ── ETH BUY cascade load ──────────────────────────────────────────────────────

def build_buy_events(conn, marks_eth):
    print("Loading ETH BUY cascade anchors ...")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    lo = now_ms - LOOKBACK_MS
    liqs_buy = load_liquidations(conn, "ETHUSDT", "BUY", lo, now_ms)
    if not liqs_buy:
        print("  No ETH BUY liq data")
        return []
    anchors_buy = reconstruct_anchors(liqs_buy, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
    print("  %d ETH BUY anchors" % len(anchors_buy))
    events_buy = []
    for anc in anchors_buy:
        ts = int(anc.anchor_ts_ms)
        p0r = marks_eth.at_or_after(ts)
        if p0r is None: continue
        sync_k = liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)+liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts)
        n2h = liq_cnt(conn,"ETHUSDT","BUY",ts-2*3600_000,ts-1000,PROP_THRESH)
        score = compute_score(conn, ts, sync_k, n2h)
        btc4h = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
        eth1h = mark_bps(conn,"ETHUSDT",ts,3600_000)
        bull = eth1h > 20 and btc4h > 50
        ev = {
            "ts": ts, "rn": float(anc.running_notional),
            "score": score, "btc4h": btc4h, "bull": bull,
            "sess": session_name(ts), "dow": dow_of(ts),
        }
        # Outcomes: BUY cascade -> SHORT (price should fall)
        ev["short_2h"] = short_out(marks_eth, ts, 2*3600_000)
        ev["short_4h"] = short_out(marks_eth, ts, 4*3600_000)
        events_buy.append(ev)
    print("  %d ETH BUY events with outcomes" % len(events_buy))
    return events_buy

# ── Report ────────────────────────────────────────────────────────────────────

def to_md(all_results):
    cols = [("label","Label"),("n","N"),("wr","WR%"),("avg","Avg bps"),
            ("t3r","T3R"),("worst","Worst"),("tail_n","Tail"),
            ("per_month","N/mo"),("mc_p","MC p"),
            ("ho_n","HO N"),("ho_wr","HO WR%"),("ho_avg","HO avg")]
    lines = []
    meta = all_results.get("meta", {})
    lines.append("# S34 Mega Research V1 — %s" % meta.get("generated_utc","")[:10])
    lines.append("> %d ETH SELL anchors | %d events | %.1f months | MC=1000" % (
        meta.get("total_anchors",0), meta.get("total_events",0), meta.get("total_months",0)))
    lines.append("")
    for sec in ["A","B","C","D","E","F"]:
        data = all_results.get(sec, {})
        if not data: continue
        lines.append("## Section %s" % sec)
        rows = []
        for k, v in data.items():
            if isinstance(v, dict) and "n" in v and v.get("n",0) > 0:
                rows.append(v)
        if not rows:
            lines.append("(no results)")
            continue
        header = "| " + " | ".join(c[1] for c in cols) + " |"
        sep    = "|" + "|".join("---" for _ in cols) + "|"
        lines.append(header); lines.append(sep)
        for r in rows:
            def fmt(k):
                v = r.get(k)
                if v is None: return "-"
                if isinstance(v, float): return ("%.1f"%v if abs(v)<10000 else "%.0f"%v)
                return str(v)
            lines.append("| " + " | ".join(fmt(k) for k,_ in cols) + " |")
        lines.append("")
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global total_months
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("S34 Mega Research V1 — all ideas A-F")

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        lo = now_ms - LOOKBACK_MS
        print("Loading ETH SELL liquidations ...")
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", lo, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        ts_span = sorted([int(a.anchor_ts_ms) for a in anchors])
        span_days = (ts_span[-1] - ts_span[0]) / 86_400_000 if len(ts_span) > 1 else 1
        total_months = max(1.0, span_days / 30.0)
        print("  %d anchors | %.0f days = %.1f months" % (len(anchors), span_days, total_months))

        marks_eth = load_mark_index(conn, "ETHUSDT")
        marks_btc = load_mark_index(conn, "BTCUSDT")

        events = build_events(conn, anchors, marks_eth)
        add_outcomes(events, marks_eth, marks_btc)

        events_buy = build_buy_events(conn, marks_eth)

    print("\nRunning all sections ...")
    all_results = {
        "meta": {
            "total_anchors": len(anchors),
            "total_events": len(events),
            "total_months": round(total_months, 1),
            "span_days": round(span_days, 0),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "A": run_A(events),
        "B": run_B(events, None, marks_eth),
        "C": run_C(events),
        "D": run_D(events),
        "E": run_E(events),
        "F": run_F(events, events_buy),
    }

    md = to_md(all_results)
    OUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print("\nSaved: %s" % OUT_JSON)
    print("Saved: %s" % OUT_MD)
    print("Done.")

if __name__ == "__main__":
    main()
