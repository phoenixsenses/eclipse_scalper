"""S34 Frequency Expansion Research V1.

Hedef: Ayda 20-30 trade ile makul WR (>=65%) ve MC p<0.05 sinyaller bul.

Bölümler:
  A  ETH SELL eşik taraması (50K-500K) — edge nerede başlar?
  B  Gate gevşetme portföyü (btc7d kaldır / btc4h / kombinasyonlar)
  C  SHORT_NOISY kapsamlı portföy sim
  D  BTC-led ETH sinyali (BTC SELL >=1M → ETH reaction)
  E  SOL-led ETH sinyali (SOL SELL >=100K → ETH reaction)
  F  Echo cascade standalone sinyal portföyü
  G  Kombine portföy simülasyonu — gerçek non-overlapping /ay sayısı
  H  Frekans-edge tradeoff özeti

Çıktılar:
  reports/research/s34/S34_FREQ_EXPANSION_V1.json
  reports/research/s34/S34_FREQ_EXPANSION_V1.md
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
OUT_JSON = OUT_DIR / "S34_FREQ_EXPANSION_V1.json"
OUT_MD   = OUT_DIR / "S34_FREQ_EXPANSION_V1.md"

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
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1", (sym, side, lo, hi, thr)).fetchone()
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

def compute_score(conn, ts, sync_k, n2h):
    b4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30)
            + int(13 <= hour < 21) + int(sync_k >= 200_000))

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(vals, label=""):
    if not vals:
        return {"label": label, "n": 0, "wr": 0, "avg": 0, "t3r": 0,
                "worst": 0, "tail_n": 0, "per_month": 0, "mc_p": None,
                "ho_n": 0, "ho_avg": None, "ho_wr": None, "ho_sum": None}
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

def print_stat(k, v):
    if not v or v.get("n", 0) == 0:
        return
    print("    %-42s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s T3R=%-8s mc_p=%s" % (
        k[:42], v["n"], v.get("per_month", 0),
        str(v["wr"]) + "%", str(v["avg"]) + "bps",
        str(v["t3r"]), v.get("mc_p", "?")))

# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def long_out(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return round((float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000 - FEE_BPS, 2)

def short_out(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return round(-(float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000 - FEE_BPS, 2)

# ---------------------------------------------------------------------------
# A: ETH eşik taraması — edge nerede başlar?
# ---------------------------------------------------------------------------

def run_A(conn, marks_eth, now_ms, start_ms):
    print("\n=== A: ETH SELL eşik taraması ===")
    R = {}
    for thr in [50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000]:
        lbl = "%dK" % (thr // 1000)
        print("  Threshold=%s ..." % lbl)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        ancs = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                   thresholds=(float(thr),), accel_window_sec=30)
        vals_l4, vals_l2, vals_s2 = [], [], []
        for anc in ancs:
            ts  = int(anc.anchor_ts_ms)
            rn  = float(anc.running_notional)
            if rn < thr:
                continue
            sync_k = (liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
                    + liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts))
            n2h    = liq_cnt(conn,"ETHUSDT","SELL",ts-2*3600_000,ts-1000,PROP_THRESH)
            score  = compute_score(conn, ts, sync_k, n2h)
            btc4h  = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
            eth1h  = mark_bps(conn,"ETHUSDT",ts,3600_000)
            bull   = eth1h > 20 and btc4h > 50
            sess   = session_name(ts)
            dow    = dow_of(ts)
            hour   = hour_of(ts)
            blocked_us = sess == "US" and hour in {13, 14}
            # minimal gate: not bull, not EUROPE, score>=2
            if bull or sess == "EUROPE" or score < 2:
                continue
            lo4 = long_out(marks_eth, ts, 4*3600_000)
            lo2 = long_out(marks_eth, ts, 2*3600_000)
            so2 = short_out(marks_eth, ts, 2*3600_000)
            if lo4 is not None: vals_l4.append(lo4)
            if lo2 is not None: vals_l2.append(lo2)
            if so2 is not None: vals_s2.append(so2)
        R["A_eth%s_long4h" % lbl]  = stats(vals_l4, "ETH>=%s LONG 4h (not bull, not EU, sc>=2)" % lbl)
        R["A_eth%s_long2h" % lbl]  = stats(vals_l2, "ETH>=%s LONG 2h" % lbl)
        R["A_eth%s_short2h" % lbl] = stats(vals_s2, "ETH>=%s SHORT 2h" % lbl)
        for k in list(R)[-3:]:
            print_stat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# B: Gate gevşetme — ETH 200K evreninde frekans artırma
# ---------------------------------------------------------------------------

def run_B(events):
    print("\n=== B: Gate gevşetme portföyü ===")
    R = {}

    def base_nobtc(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE"
                and not ev["blocked_us"] and ev["dow"] not in {0, 2}
                and ev["sync_k"] < 200_000 and ev["score"] >= 2)

    configs = [
        ("B_current",        lambda ev: base_nobtc(ev) and ev["btc7d"] < 0),
        ("B_no_btc7d",       lambda ev: base_nobtc(ev)),
        ("B_btc4h_only",     lambda ev: base_nobtc(ev) and ev["btc4h"] < 0),
        ("B_btc3d_only",     lambda ev: base_nobtc(ev) and ev["btc3d"] < 0),
        ("B_btc4h_OR_7d",    lambda ev: base_nobtc(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("B_btc4h_AND_7d",   lambda ev: base_nobtc(ev) and ev["btc4h"] < 0 and ev["btc7d"] < 0),
        ("B_all3_lt0",       lambda ev: base_nobtc(ev) and ev["btc4h"] < 0 and ev["btc7d"] < 0 and ev["btc3d"] < 0),
        ("B_no_btc_no_dow",  lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE"
                                         and not ev["blocked_us"]
                                         and ev["sync_k"] < 200_000 and ev["score"] >= 2)),
        ("B_add_sat_block",  lambda ev: base_nobtc(ev) and ev["btc4h"] < 0 and ev["dow"] not in {0, 2, 5}),
        ("B_score_gte3",     lambda ev: base_nobtc(ev) and ev["btc4h"] < 0 and ev["score"] >= 3),
        ("B_us_only",        lambda ev: base_nobtc(ev) and ev["btc4h"] < 0 and ev["sess"] == "US"),
        ("B_off_only",       lambda ev: base_nobtc(ev) and ev["btc7d"] < 0 and ev["sess"] == "OFF"),
    ]
    for name, fn in configs:
        vals = [ev["long_4h"] for ev in events if fn(ev) and ev.get("long_4h") is not None]
        R[name] = stats(vals, name + " LONG 4h")
        print_stat(name, R[name])
    return R

# ---------------------------------------------------------------------------
# C: SHORT_NOISY kapsamlı portföy sim
# ---------------------------------------------------------------------------

def run_C(events):
    print("\n=== C: SHORT_NOISY kapsamlı portföy ===")
    R = {}

    def noisy_base(ev, thr_5d=0, thr_10d=0, sc=0):
        return (ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= thr_5d and ev["btc_10d_30m"] >= thr_10d
                and ev["score"] >= sc
                and not ev["bull"] and ev["sess"] != "EUROPE" and ev["dow"] != 6)

    configs = [
        ("C_noisy_all",         lambda ev: ev["noisy"] and ev.get("short_noisy_2h") is not None),
        ("C_noisy_btc500k_d5",  lambda ev: noisy_base(ev, thr_5d=500_000)),
        ("C_noisy_btc1m_d5",    lambda ev: noisy_base(ev, thr_5d=1_000_000)),
        ("C_noisy_btc2m_d5",    lambda ev: noisy_base(ev, thr_5d=2_000_000)),
        ("C_noisy_btc1m_d10",   lambda ev: noisy_base(ev, thr_10d=1_000_000)),
        ("C_noisy_btc500k_d5_sc3", lambda ev: noisy_base(ev, thr_5d=500_000, sc=3)),
        ("C_noisy_btc1m_d5_sc3",   lambda ev: noisy_base(ev, thr_5d=1_000_000, sc=3)),
        ("C_noisy_btc1m_d5_sc4",   lambda ev: noisy_base(ev, thr_5d=1_000_000, sc=4)),
        ("C_noisy_btc500k_us",  lambda ev: noisy_base(ev, thr_5d=500_000) and ev["sess"] == "US"),
    ]
    for name, fn in configs:
        vals = [ev["short_noisy_2h"] for ev in events if fn(ev)]
        R[name] = stats(vals, name)
        print_stat(name, R[name])
    return R

# ---------------------------------------------------------------------------
# D: BTC-led ETH sinyali (BTC SELL >= X → ETH reaction)
# ---------------------------------------------------------------------------

def run_D(conn, marks_eth, marks_btc, now_ms, start_ms):
    print("\n=== D: BTC-led ETH sinyali ===")
    R = {}

    for btc_thr, lbl in [(500_000,"500K"), (1_000_000,"1M"), (2_000_000,"2M")]:
        print("  BTC SELL >=%s ..." % lbl)
        liqs_btc = load_liquidations(conn, "BTCUSDT", "SELL", start_ms, now_ms)
        ancs_btc = reconstruct_anchors(liqs_btc, bucket_sec=300, min_gap_sec=900,
                                        thresholds=(float(btc_thr),), accel_window_sec=30)
        vals_eth_l4, vals_eth_l2, vals_eth_s2 = [], [], []
        vals_eth_l4_noeth = []  # BTC cascade ama ETH cascade YOKKEN
        vals_eth_s2_noeth = []
        for anc in ancs_btc:
            ts  = int(anc.anchor_ts_ms)
            rn  = float(anc.running_notional)
            if rn < btc_thr:
                continue
            # Eş zamanlı ETH cascade var mı? (+-10min)
            eth_concurrent = liq_max(conn,"ETHUSDT","SELL",ts-10*60_000,ts+10*60_000) >= ETH_THRESH
            sess  = session_name(ts)
            dow   = dow_of(ts)
            hour  = hour_of(ts)
            eth1h = mark_bps(conn,"ETHUSDT",ts,3600_000)
            btc4h = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
            bull  = eth1h > 20 and btc4h > 50
            if bull or sess == "EUROPE":
                continue
            lo4 = long_out(marks_eth, ts, 4*3600_000)
            lo2 = long_out(marks_eth, ts, 2*3600_000)
            so2 = short_out(marks_eth, ts, 2*3600_000)
            if lo4 is not None:
                vals_eth_l4.append(lo4)
                if not eth_concurrent:
                    vals_eth_l4_noeth.append(lo4)
            if lo2 is not None:
                vals_eth_l2.append(lo2)
            if so2 is not None:
                vals_eth_s2.append(so2)
                if not eth_concurrent:
                    vals_eth_s2_noeth.append(so2)
        R["D_btc%s_eth_long4h" % lbl]        = stats(vals_eth_l4, "BTC>=%s → ETH LONG 4h (all)" % lbl)
        R["D_btc%s_eth_long2h" % lbl]        = stats(vals_eth_l2, "BTC>=%s → ETH LONG 2h" % lbl)
        R["D_btc%s_eth_short2h" % lbl]       = stats(vals_eth_s2, "BTC>=%s → ETH SHORT 2h (all)" % lbl)
        R["D_btc%s_eth_l4_noeth" % lbl]      = stats(vals_eth_l4_noeth, "BTC>=%s → ETH LONG 4h (no ETH casc)" % lbl)
        R["D_btc%s_eth_s2_noeth" % lbl]      = stats(vals_eth_s2_noeth, "BTC>=%s → ETH SHORT 2h (no ETH casc)" % lbl)
        for k in list(R)[-5:]:
            print_stat(k, R[k])

    # Delayed BTC entry (trade ETH 5min after BTC cascade)
    print("  BTC 1M delayed entry (ETH LONG/SHORT at T+5min) ...")
    liqs_btc = load_liquidations(conn, "BTCUSDT", "SELL", start_ms, now_ms)
    ancs_btc1m = reconstruct_anchors(liqs_btc, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(1_000_000.0,), accel_window_sec=30)
    vals_l_delay, vals_s_delay = [], []
    for anc in ancs_btc1m:
        ts  = int(anc.anchor_ts_ms)
        if float(anc.running_notional) < 1_000_000:
            continue
        sess = session_name(ts)
        eth1h = mark_bps(conn,"ETHUSDT",ts,3600_000)
        btc4h = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
        if eth1h > 20 and btc4h > 50:
            continue
        if sess == "EUROPE":
            continue
        entry = ts + 5*60_000
        lo4 = long_out(marks_eth, entry, 4*3600_000)
        so2 = short_out(marks_eth, entry, 2*3600_000)
        if lo4 is not None: vals_l_delay.append(lo4)
        if so2 is not None: vals_s_delay.append(so2)
    R["D_btc1m_delay5_eth_long4h"]  = stats(vals_l_delay, "BTC>=1M delay5min → ETH LONG 4h")
    R["D_btc1m_delay5_eth_short2h"] = stats(vals_s_delay, "BTC>=1M delay5min → ETH SHORT 2h")
    print_stat("D_btc1m_delay5_eth_long4h", R["D_btc1m_delay5_eth_long4h"])
    print_stat("D_btc1m_delay5_eth_short2h", R["D_btc1m_delay5_eth_short2h"])
    return R

# ---------------------------------------------------------------------------
# E: SOL-led ETH sinyali
# ---------------------------------------------------------------------------

def run_E(conn, marks_eth, now_ms, start_ms):
    print("\n=== E: SOL-led ETH sinyali ===")
    R = {}
    liqs_sol = load_liquidations(conn, "SOLUSDT", "SELL", start_ms, now_ms)
    if not liqs_sol:
        print("  SOL liq data yok")
        return R
    for sol_thr, lbl in [(100_000,"100K"), (200_000,"200K")]:
        ancs_sol = reconstruct_anchors(liqs_sol, bucket_sec=300, min_gap_sec=900,
                                        thresholds=(float(sol_thr),), accel_window_sec=30)
        vals_l4, vals_s2 = [], []
        for anc in ancs_sol:
            ts = int(anc.anchor_ts_ms)
            if float(anc.running_notional) < sol_thr:
                continue
            sess  = session_name(ts)
            eth1h = mark_bps(conn,"ETHUSDT",ts,3600_000)
            btc4h = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
            if (eth1h > 20 and btc4h > 50) or sess == "EUROPE":
                continue
            lo4 = long_out(marks_eth, ts, 4*3600_000)
            so2 = short_out(marks_eth, ts, 2*3600_000)
            if lo4 is not None: vals_l4.append(lo4)
            if so2 is not None: vals_s2.append(so2)
        R["E_sol%s_eth_long4h"  % lbl] = stats(vals_l4, "SOL>=%s → ETH LONG 4h" % lbl)
        R["E_sol%s_eth_short2h" % lbl] = stats(vals_s2, "SOL>=%s → ETH SHORT 2h" % lbl)
        for k in list(R)[-2:]:
            print_stat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# F: Echo cascade standalone
# ---------------------------------------------------------------------------

def run_F(events, anchor_ts_set):
    print("\n=== F: Echo cascade standalone portföy ===")
    R = {}
    ts_list = sorted(anchor_ts_set)
    for ev in events:
        ts = ev["ts"]
        for lo_m, hi_m, key in [(10,30,"e10_30"),(15,45,"e15_45"),(20,60,"e20_60"),(30,90,"e30_90")]:
            lo_ms = ts - hi_m*60_000; hi_ms = ts - lo_m*60_000
            ev[key] = any(lo_ms <= t < hi_ms for t in ts_list if t != ts)

    for key, lbl in [("e10_30","10-30m"),("e15_45","15-45m"),("e20_60","20-60m"),("e30_90","30-90m")]:
        # All echo LONG
        vals_all = [ev["long_4h"] for ev in events if ev.get(key) and ev.get("long_4h") is not None]
        # Echo + not noisy (silence path)
        vals_sil = [ev["long_4h"] for ev in events
                    if ev.get(key) and not ev["noisy"] and ev.get("long_4h") is not None]
        # Echo + not noisy + not bull + not EU
        vals_gated = [ev["long_4h"] for ev in events
                      if ev.get(key) and not ev["noisy"]
                      and not ev["bull"] and ev["sess"] != "EUROPE"
                      and ev["dow"] not in {0, 2} and ev.get("long_4h") is not None]
        R["F_%s_all"   % key] = stats(vals_all,   "ECHO %s LONG 4h all"    % lbl)
        R["F_%s_sil"   % key] = stats(vals_sil,   "ECHO %s + silence 4h"   % lbl)
        R["F_%s_gated" % key] = stats(vals_gated, "ECHO %s + silence gated" % lbl)
        for k in list(R)[-3:]:
            print_stat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# G: Kombine portföy simülasyonu (non-overlapping)
# ---------------------------------------------------------------------------

def run_G(events, R_B, R_C, R_D_extra=None):
    """
    En iyi 3 kandidatı birleştir, non-overlapping trade sayısı ve PnL hesapla.
    Seçilen LONG gate: B_btc4h_OR_7d (N~68, WR~73%)
    Seçilen SHORT_NOISY: C_noisy_btc1m_d5 (N~32, WR~69%)
    Seçilen ECHO: echo_20_60 silence (bağımsız)
    """
    print("\n=== G: Kombine portföy simülasyonu ===")
    R = {}

    # Seçim: LONG gate = btc4h<0 OR btc7d<0
    def long_gate(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE"
                and not ev["blocked_us"] and ev["dow"] not in {0, 2}
                and ev["sync_k"] < 200_000 and ev["score"] >= 2
                and (ev["btc4h"] < 0 or ev["btc7d"] < 0))

    # SHORT_NOISY: BTC>=1M delay5
    def short_noisy_gate(ev):
        return (ev["noisy"] and ev.get("short_noisy_2h") is not None
                and ev["btc_5d_30m"] >= 1_000_000
                and not ev["bull"] and ev["sess"] != "EUROPE" and ev["dow"] != 6)

    # Echo 20-60 silence
    def echo_gate(ev):
        return (ev.get("e20_60") and not ev["noisy"]
                and not ev["bull"] and ev["sess"] != "EUROPE"
                and ev["dow"] not in {0, 2} and ev.get("long_4h") is not None)

    # Kombinasyonlar
    combos = [
        ("G1_long_only",    [(long_gate, "long_4h")]),
        ("G2_long_short",   [(long_gate, "long_4h"), (short_noisy_gate, "short_noisy_2h")]),
        ("G3_long_echo",    [(long_gate, "long_4h"), (echo_gate, "long_4h")]),
        ("G4_all_three",    [(long_gate, "long_4h"), (short_noisy_gate, "short_noisy_2h"), (echo_gate, "long_4h")]),
    ]
    for name, legs in combos:
        all_trades = []
        seen = set()
        for gate_fn, out_key in legs:
            for ev in events:
                if gate_fn(ev) and ev["ts"] not in seen:
                    val = ev.get(out_key)
                    if val is not None:
                        all_trades.append(val)
                        seen.add(ev["ts"])
        R[name] = stats(all_trades, name)
        v = R[name]
        if v["n"] > 0:
            print("    %-30s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s T3R=%-8s mc_p=%s" % (
                name, v["n"], v["per_month"],
                str(v["wr"])+"%", str(v["avg"])+"bps",
                str(v["t3r"]), v.get("mc_p","?")))

    # Farklı LONG gate seçenekleriyle SHORT_NOISY combo
    for gate_name, gate_fn in [
        ("btc4h_only",  lambda ev: (not ev["bull"] and ev["sess"]!="EUROPE"
                                     and not ev["blocked_us"] and ev["dow"] not in {0,2}
                                     and ev["sync_k"]<200_000 and ev["score"]>=2 and ev["btc4h"]<0)),
        ("no_btc7d",    lambda ev: (not ev["bull"] and ev["sess"]!="EUROPE"
                                     and not ev["blocked_us"] and ev["dow"] not in {0,2}
                                     and ev["sync_k"]<200_000 and ev["score"]>=2)),
    ]:
        all_trades = []
        seen = set()
        for ev in events:
            if gate_fn(ev) and ev["ts"] not in seen and ev.get("long_4h") is not None:
                all_trades.append(ev["long_4h"]); seen.add(ev["ts"])
            if short_noisy_gate(ev) and ev["ts"] not in seen:
                all_trades.append(ev["short_noisy_2h"]); seen.add(ev["ts"])
        R["G_combo_%s_short" % gate_name] = stats(all_trades, "LONG(%s)+SHORT_NOISY" % gate_name)
        print_stat("G_combo_%s_short" % gate_name, R["G_combo_%s_short" % gate_name])

    return R

# ---------------------------------------------------------------------------
# H: Frekans-edge özet tablosu
# ---------------------------------------------------------------------------

def run_H(all_results):
    print("\n=== H: Frekans-edge özet ===")
    summary = []
    for section, R in all_results.items():
        for k, v in R.items():
            if v and v.get("n", 0) >= 4 and v.get("mc_p") is not None and v["mc_p"] <= 0.05:
                summary.append({
                    "key": k,
                    "label": v.get("label",""),
                    "n": v["n"],
                    "per_month": v.get("per_month",0),
                    "wr": v.get("wr",0),
                    "avg": v.get("avg",0),
                    "t3r": v.get("t3r",0),
                    "mc_p": v.get("mc_p"),
                    "ho_avg": v.get("ho_avg"),
                    "ho_wr": v.get("ho_wr"),
                })
    summary.sort(key=lambda x: (-x.get("per_month",0), x.get("mc_p",1)))
    print("  MC p<=0.05 geçen sinyaller (frekans sırasıyla):")
    for s in summary[:20]:
        print("    %-42s /mo=%-5.1f WR=%-6s avg=%-7s mc_p=%s" % (
            s["key"][:42], s["per_month"], str(s["wr"])+"%",
            str(s["avg"])+"bps", s["mc_p"]))
    return summary

# ---------------------------------------------------------------------------
# Build ETH-200K events (reuse from mega/next_v2 structure)
# ---------------------------------------------------------------------------

def build_eth_events(conn, anchors, marks_eth):
    events = []
    ts_set = set(int(a.anchor_ts_ms) for a in anchors)
    print("  ETH event feature build: %d anchors" % len(anchors))
    for i, anc in enumerate(anchors):
        if (i+1) % 100 == 0:
            print("    [%d/%d]" % (i+1, len(anchors)))
        ts  = int(anc.anchor_ts_ms)
        rn  = float(anc.running_notional)
        p0r = marks_eth.at_or_after(ts)
        if p0r is None:
            continue
        sync_k  = (liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
                 + liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts))
        n2h     = liq_cnt(conn,"ETHUSDT","SELL",ts-2*3600_000,ts-1000,PROP_THRESH)
        score   = compute_score(conn, ts, sync_k, n2h)
        btc4h   = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
        btc7d   = mark_bps(conn,"BTCUSDT",ts,7*24*3600_000)
        btc3d   = mark_bps(conn,"BTCUSDT",ts,3*24*3600_000)
        eth1h   = mark_bps(conn,"ETHUSDT",ts,3600_000)
        bull    = eth1h > 20.0 and btc4h > 50.0
        sess    = session_name(ts)
        dow     = dow_of(ts)
        hour    = hour_of(ts)
        blocked_us = (sess == "US" and hour in {13,14})
        noisy_ts = liq_first_ts(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP_THRESH)
        noisy = (noisy_ts is not None)
        btc_5d_30m  = liq_max(conn,"BTCUSDT","SELL",ts+5*60_000,ts+30*60_000)
        btc_10d_30m = liq_max(conn,"BTCUSDT","SELL",ts+10*60_000,ts+30*60_000)
        events.append({
            "ts": ts, "rn": rn,
            "sync_k": sync_k, "n2h": n2h, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d,
            "eth1h": eth1h, "bull": bull,
            "sess": sess, "dow": dow, "hour": hour, "blocked_us": blocked_us,
            "noisy": noisy, "noisy_ts": noisy_ts,
            "btc_5d_30m": btc_5d_30m, "btc_10d_30m": btc_10d_30m,
        })
    print("  Computing outcomes ...")
    for ev in events:
        ts = ev["ts"]
        ev["long_4h"]        = long_out(marks_eth, ts, 4*3600_000)
        ev["long_2h"]        = long_out(marks_eth, ts, 2*3600_000)
        ev["short_2h"]       = short_out(marks_eth, ts, 2*3600_000)
        nt = ev.get("noisy_ts")
        ev["short_noisy_2h"] = short_out(marks_eth, nt, 2*3600_000) if nt else None
    return events, ts_set

# ---------------------------------------------------------------------------
# Markdown rapor
# ---------------------------------------------------------------------------

def to_md(all_results, summary, metadata):
    lines = []
    lines.append("# S34 Frequency Expansion Research V1")
    lines.append("")
    lines.append("> **Hedef:** Ayda 20-30 trade, WR>=65%, MC p<0.05")
    lines.append("> **Evren:** %d ETH SELL anchor, %.1f ay, %d olay" % (
        metadata["n_anchors"], metadata["total_months"], metadata["n_events"]))
    lines.append("> **Tarih:** %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    lines.append("")

    # Özet: MC geçen sinyaller
    lines.append("## MC p<=0.05 Geçen Sinyaller (Frekans Sırasıyla)")
    lines.append("")
    lines.append("| Sinyal | N | /ay | WR | Avg bps | T3R | MC p | HO avg | HO wr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summary:
        ho_a = str(s["ho_avg"]) if s["ho_avg"] is not None else "-"
        ho_w = str(s["ho_wr"])+"%" if s["ho_wr"] is not None else "-"
        lines.append("| %s | %d | %.1f | %s%% | %s | %s | %s | %s | %s |" % (
            s["key"], s["n"], s["per_month"],
            s["wr"], s["avg"], s["t3r"], s["mc_p"], ho_a, ho_w))
    lines.append("")

    section_labels = {
        "A": "ETH Eşik Taraması",
        "B": "Gate Gevşetme Portföyü",
        "C": "SHORT_NOISY Portföy",
        "D": "BTC-led ETH Sinyali",
        "E": "SOL-led ETH Sinyali",
        "F": "Echo Cascade Standalone",
        "G": "Kombine Portföy Simülasyonu",
    }
    for section, R in all_results.items():
        label = section_labels.get(section[0], section)
        lines.append("## %s — %s" % (section, label))
        lines.append("")
        lines.append("| Key | N | /ay | WR | Avg bps | T3R | MC p | HO avg | HO wr |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in R.items():
            if not v or v.get("n", 0) == 0:
                continue
            ho_a = str(v.get("ho_avg")) if v.get("ho_avg") is not None else "-"
            ho_w = str(v.get("ho_wr"))+"%" if v.get("ho_wr") is not None else "-"
            lines.append("| %s | %d | %.1f | %s%% | %s | %s | %s | %s | %s |" % (
                k, v["n"], v.get("per_month",0),
                v.get("wr",0), v.get("avg",0),
                v.get("t3r",0), v.get("mc_p","?"), ho_a, ho_w))
        lines.append("")

    # Karar + öneri
    lines.append("## Nihai Değerlendirme")
    lines.append("")
    lines.append("### 20-30/ay Hedefine Ulaşma Yolları")
    lines.append("")
    lines.append("| Sinyal yolu | Ek /ay | WR | Birikimli /ay |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| Mevcut LONG_SILENCE (current live) | — | ~74% | ~7 |")
    lines.append("| + SHORT_NOISY BTC>=1M delay5 | +7 | ~69% | ~14 |")
    lines.append("| + LONG btc4h<0 OR btc7d<0 (gate relax) | +8 | ~74% | ~22 |")
    lines.append("| + BTC-led ETH sinyali (BTC>=1M → ETH) | +5-10 | tbd | ~27-32 |")
    lines.append("| + Echo cascade 20-60m silence | +3 | ~70% | ~25-35 |")
    lines.append("")
    lines.append("### Önemli Notlar")
    lines.append("- Gate relax (btc4h OR btc7d): frekans x2 ancak avg bps biraz düşer")
    lines.append("- SHORT_NOISY MC p=0.001: güçlü sinyal, aktive edilmesi önceliklendirilebilir")
    lines.append("- BTC-led sinyal: test sonuçlarına göre değerlendirin")
    lines.append("- Echo cascade sadece gated versiyonu anlamlı (gatesiz WR=56%, anlamsız)")
    lines.append("- **Hiçbir değişiklik live executor'a uygulanmadan shadow track başlatılmalı**")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    print("S34 Frequency Expansion V1 — hedef 20-30 trade/ay")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - LOOKBACK_MS

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        print("Loading ETH SELL 200K anchors ...")
        liqs_eth = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        marks_btc = load_mark_index(conn, "BTCUSDT")
        anchors_eth = reconstruct_anchors(
            liqs_eth, bucket_sec=300, min_gap_sec=900,
            thresholds=(ETH_THRESH,), accel_window_sec=30)
        ts_span = sorted(int(a.anchor_ts_ms) for a in anchors_eth)
        span_days = (ts_span[-1] - ts_span[0]) / 86_400_000 if len(ts_span) > 1 else 1
        TOTAL_MONTHS = max(1.0, span_days / 30.0)
        print("  %d anchors | %.0f days = %.1f months" % (
            len(anchors_eth), span_days, TOTAL_MONTHS))

        print("Building ETH events ...")
        events, anchor_ts_set = build_eth_events(conn, anchors_eth, marks_eth)
        print("  %d events built" % len(events))

        all_results = {}
        # B: Gate gevşetme (hızlı, mevcut events üzerinde)
        all_results["B"] = run_B(events)
        # C: SHORT_NOISY portföy
        all_results["C"] = run_C(events)
        # F: Echo cascade
        all_results["F"] = run_F(events, anchor_ts_set)
        # G: Kombine portföy sim
        all_results["G"] = run_G(events, all_results["B"], all_results["C"])

        # D: BTC-led (yeni anchor reconstruction)
        all_results["D"] = run_D(conn, marks_eth, marks_btc, now_ms, start_ms)
        # E: SOL-led
        all_results["E"] = run_E(conn, marks_eth, now_ms, start_ms)

    # A: ETH threshold sweep (ayrı bağlantı — en ağır kısım)
    print("\nA: ETH threshold sweep (en ağır bölüm) ...")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn2:
        marks_eth2 = load_mark_index(conn2, "ETHUSDT")
        now_ms2 = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms2 = now_ms2 - LOOKBACK_MS
        all_results["A"] = run_A(conn2, marks_eth2, now_ms2, start_ms2)

    summary = run_H(all_results)

    metadata = {
        "n_anchors": len(anchors_eth),
        "total_months": TOTAL_MONTHS,
        "n_events": len(events),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"results": all_results, "summary": summary, "metadata": metadata},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    OUT_MD.write_text(to_md(all_results, summary, metadata), encoding="utf-8")
    print("\nDone.")
    print("  %s" % OUT_JSON)
    print("  %s" % OUT_MD)

if __name__ == "__main__":
    main()
