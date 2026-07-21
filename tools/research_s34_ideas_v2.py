"""S34 Research Ideas V2 — Comprehensive test suite.

Sections:
  VOL  Vol regime at cascade (vol_state join)
  OFI  Order flow imbalance 5m pre-cascade (agg_trades)
  ST   State transition 4-partition (silence/noisy x btc)
  LAG  BTC-ETH timing delta
  DEN  Cascade density quartile
  EX   Cascade exhaustion (prebuildup)
  SW   Second wave / echo window sweep
  SH   SHORT_NOISY hold time sweep
  GR   LONG gate: btc4h vs btc7d driver
  FB   Failed bounce SHORT signal
  VAC  Liquidity vacuum depth
  SYM  SHORT symmetry DOW x score

Outputs:
  reports/research/s34/S34_IDEAS_V2.json
  reports/research/s34/S34_IDEAS_V2.md
"""
from __future__ import annotations

import bisect
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH      = ROOT / "data" / "microstructure.db"
OUT_DIR      = ROOT / "reports" / "research" / "s34"
OUT_JSON     = OUT_DIR / "S34_IDEAS_V2.json"
OUT_MD       = OUT_DIR / "S34_IDEAS_V2.md"

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
# Vol state helpers
# ---------------------------------------------------------------------------

def load_vol_state(conn):
    rows = conn.execute(
        "SELECT ts_ms, rv_5m, vol_decile, high_vol_alert FROM vol_state "
        "WHERE symbol='ETHUSDT' ORDER BY ts_ms"
    ).fetchall()
    print("  vol_state rows: %d" % len(rows))
    return rows

def vol_at(vol_rows, ts_ms):
    if not vol_rows:
        return None
    ts_arr = [r[0] for r in vol_rows]
    idx = bisect.bisect_right(ts_arr, ts_ms) - 1
    if idx < 0:
        return None
    return vol_rows[idx]

# ---------------------------------------------------------------------------
# OFI helpers (agg_trades)
# ---------------------------------------------------------------------------

def ofi_window(conn, sym, lo_ms, hi_ms):
    row = conn.execute(
        "SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0.0 END),"
        "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0.0 END) "
        "FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
        (sym, lo_ms, hi_ms)
    ).fetchone()
    if not row or row[0] is None:
        return None
    buy_n  = float(row[0])
    sell_n = float(row[1])
    total  = buy_n + sell_n
    return {
        "ofi":   buy_n - sell_n,
        "ratio": (buy_n - sell_n) / total if total > 0 else 0.0,
    }

# ---------------------------------------------------------------------------
# BTC timing helpers
# ---------------------------------------------------------------------------

def load_btc_big_sells(conn, start_ms, end_ms, thr=500_000):
    rows = conn.execute(
        "SELECT ts_ms FROM liquidations "
        "WHERE symbol='BTCUSDT' AND side='SELL' AND notional>=? "
        "AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (thr, start_ms, end_ms)
    ).fetchall()
    return [int(r[0]) for r in rows]

def btc_nearest_delta_ms(btc_ts_list, anchor_ts, window_ms=30 * 60_000):
    if not btc_ts_list:
        return None
    lo = anchor_ts - window_ms
    hi = anchor_ts + window_ms
    lo_i = bisect.bisect_left(btc_ts_list, lo)
    hi_i = bisect.bisect_right(btc_ts_list, hi)
    cands = btc_ts_list[lo_i:hi_i]
    if not cands:
        return None
    nearest = min(cands, key=lambda t: abs(t - anchor_ts))
    return nearest - anchor_ts

def echo_check(ts_list, ts, lo_min, hi_min):
    lo_ms = ts - hi_min * 60_000
    hi_ms = ts - lo_min * 60_000
    lo_i = bisect.bisect_left(ts_list, lo_ms)
    hi_i = bisect.bisect_left(ts_list, hi_ms)
    for i in range(lo_i, hi_i):
        if ts_list[i] != ts:
            return True
    return False

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(vals, label=""):
    if not vals:
        return {"label": label, "n": 0, "wr": 0, "avg": 0, "t3r": 0,
                "worst": 0, "tail_n": 0, "per_month": 0, "mc_p": None,
                "ho_n": 0, "ho_avg": None, "ho_wr": None, "ho_sum": None}
    n    = len(vals)
    wins = sum(1 for v in vals if v > 0)
    sv   = sorted(vals)
    t3   = sorted(vals, reverse=True)
    t3r  = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    avg  = sum(vals) / n
    rng  = random.Random(0)
    mc_p = None
    if n >= 4:
        ct   = sum(1 for _ in range(MC_ITER)
                   if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / n >= avg)
        mc_p = round(ct / MC_ITER, 3)
    cut  = max(1, int(n * 0.7))
    ho   = vals[cut:]
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
        "ho_wr":  round(100 * sum(1 for v in ho if v > 0) / len(ho), 1) if ho else None,
        "ho_sum": round(sum(ho), 1) if ho else None,
    }

def pstat(k, v):
    if not v or v.get("n", 0) == 0:
        return
    print("    %-44s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s T3R=%-8s mc_p=%s" % (
        k[:44], v["n"], v.get("per_month", 0),
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
# Build events
# ---------------------------------------------------------------------------

def build_events(conn, anchors, marks_eth, vol_rows, btc_sell_ts):
    events = []
    total  = len(anchors)
    ts_list = sorted(int(a.anchor_ts_ms) for a in anchors)
    print("  Building %d events ..." % total)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print("    [%d/%d]" % (i + 1, total))
        ts  = int(anc.anchor_ts_ms)
        rn  = float(anc.running_notional)
        p0r = marks_eth.at_or_after(ts)
        if p0r is None:
            continue

        sync_k  = (liq_sum(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts)
                 + liq_sum(conn, "SOLUSDT", "SELL", ts - 10*60_000, ts))
        n2h     = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
        score   = compute_score(conn, ts, sync_k, n2h)
        btc4h   = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        btc7d   = mark_bps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000)
        btc3d   = mark_bps(conn, "BTCUSDT", ts, 3 * 24 * 3600_000)
        btc5m   = mark_bps(conn, "BTCUSDT", ts, 5 * 60_000)
        eth1h   = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        bull    = eth1h > 20.0 and btc4h > 50.0
        sess    = session_name(ts)
        dow     = dow_of(ts)
        hour    = hour_of(ts)
        blocked = (sess == "US" and hour in {13, 14})

        density_24h = liq_cnt(conn, "ETHUSDT", "SELL",
                               ts - 24*3600_000, ts - 300_000, ETH_THRESH)
        prebuildup  = liq_cnt(conn, "ETHUSDT", "SELL",
                               ts - 30*60_000,   ts - 1000, PROP_THRESH)

        p5m = marks_eth.at_or_before(ts + 5 * 60_000)
        failed_cascade = False
        fc_bps = None
        if p5m and float(p0r[1]) > 0:
            fc_bps = (float(p5m[1]) - float(p0r[1])) / float(p0r[1]) * 10_000
            failed_cascade = fc_bps > 0

        noisy_ts    = liq_first_ts(conn, "ETHUSDT", "SELL",
                                    ts + 60_000, ts + 30*60_000, PROP_THRESH)
        noisy       = (noisy_ts is not None)
        btc_5d_30m  = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000,  ts + 30*60_000)
        btc_5d_60m  = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000,  ts + 60*60_000)
        btc_10d_60m = liq_max(conn, "BTCUSDT", "SELL", ts + 10*60_000, ts + 60*60_000)

        # second wave after anchor
        sw_90_180  = liq_max(conn, "ETHUSDT", "SELL", ts + 90*60_000,  ts + 180*60_000)
        sw_180_360 = liq_max(conn, "ETHUSDT", "SELL", ts + 180*60_000, ts + 360*60_000)

        # vacuum: total sell notional 0-15m and clean flags
        vac_15m     = liq_sum(conn, "ETHUSDT", "SELL", ts + 1000, ts + 15*60_000)
        vac_clean   = liq_cnt(conn, "ETHUSDT", "SELL", ts + 1000, ts + 30*60_000, 10_000) == 0

        # echo flags (binary search)
        echo_10_30  = echo_check(ts_list, ts, 10, 30)
        echo_20_60  = echo_check(ts_list, ts, 20, 60)
        echo_30_90  = echo_check(ts_list, ts, 30, 90)
        echo_45_120 = echo_check(ts_list, ts, 45, 120)
        echo_60_180 = echo_check(ts_list, ts, 60, 180)

        # vol state
        vs_now  = vol_at(vol_rows, ts)
        vs_pre  = vol_at(vol_rows, ts - 30 * 60_000)
        vd_now  = int(vs_now[2]) if vs_now and vs_now[2] is not None else None
        vd_pre  = int(vs_pre[2]) if vs_pre and vs_pre[2] is not None else None
        rv5m    = float(vs_now[1]) if vs_now and vs_now[1] is not None else None
        hi_vol  = bool(vs_now[3]) if vs_now and vs_now[3] is not None else None
        vol_exp = (vd_now - vd_pre) if (vd_now is not None and vd_pre is not None) else None

        # BTC timing delta
        btc_delta = btc_nearest_delta_ms(btc_sell_ts, ts, 30 * 60_000)

        events.append({
            "ts": ts, "rn": rn,
            "sync_k": sync_k, "n2h": n2h, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d, "btc5m": btc5m,
            "eth1h": eth1h, "bull": bull, "sess": sess, "dow": dow,
            "hour": hour, "blocked": blocked,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "failed_cascade": failed_cascade, "fc_bps": fc_bps,
            "noisy": noisy, "noisy_ts": noisy_ts,
            "btc_5d_30m": btc_5d_30m, "btc_5d_60m": btc_5d_60m,
            "btc_10d_60m": btc_10d_60m,
            "sw_90_180": sw_90_180, "sw_180_360": sw_180_360,
            "vac_15m": vac_15m, "vac_clean": vac_clean,
            "echo_10_30": echo_10_30, "echo_20_60": echo_20_60,
            "echo_30_90": echo_30_90, "echo_45_120": echo_45_120,
            "echo_60_180": echo_60_180,
            "vd_now": vd_now, "vd_pre": vd_pre,
            "rv5m": rv5m, "hi_vol": hi_vol, "vol_exp": vol_exp,
            "btc_delta": btc_delta,
        })
    return events

def add_outcomes(events, marks_eth):
    print("  Computing outcomes ...")
    for ev in events:
        ts = ev["ts"]
        ev["l4h"]  = long_out(marks_eth, ts, 4 * 3600_000)
        ev["l3h"]  = long_out(marks_eth, ts, 3 * 3600_000)
        ev["l2h"]  = long_out(marks_eth, ts, 2 * 3600_000)
        ev["s2h"]  = short_out(marks_eth, ts, 2 * 3600_000)
        nt = ev.get("noisy_ts")
        ev["sn_90m"] = short_out(marks_eth, nt, 90 * 60_000) if nt else None
        ev["sn_2h"]  = short_out(marks_eth, nt, 2 * 3600_000) if nt else None
        ev["sn_3h"]  = short_out(marks_eth, nt, 3 * 3600_000) if nt else None
        ev["sn_4h"]  = short_out(marks_eth, nt, 4 * 3600_000) if nt else None

def add_ofi(conn, events):
    print("  OFI from agg_trades (%d anchors) ..." % len(events))
    for i, ev in enumerate(events):
        if (i + 1) % 150 == 0:
            print("    OFI [%d/%d]" % (i + 1, len(events)))
        ts = ev["ts"]
        pre  = ofi_window(conn, "ETHUSDT", ts - 5*60_000, ts)
        post = ofi_window(conn, "ETHUSDT", ts, ts + 10*60_000)
        ev["ofi_pre_ratio"]  = pre["ratio"]  if pre  else None
        ev["ofi_pre"]        = pre["ofi"]    if pre  else None
        ev["ofi_post_ratio"] = post["ratio"] if post else None

# ---------------------------------------------------------------------------
# Base gates
# ---------------------------------------------------------------------------

def bl_cur(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked"]
            and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
            and ev["btc7d"] < 0 and ev["score"] >= 2)

def bl_or(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked"]
            and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000
            and (ev["btc4h"] < 0 or ev["btc7d"] < 0) and ev["score"] >= 2)

def bl_nobtc(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked"]
            and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000 and ev["score"] >= 2)

# ---------------------------------------------------------------------------
# VOL
# ---------------------------------------------------------------------------

def run_VOL(events):
    print("\n=== VOL ===")
    R = {}
    ev_v = [ev for ev in events if ev.get("vd_now") is not None]
    print("  With vol data: %d" % len(ev_v))

    bl = [ev["l4h"] for ev in ev_v if bl_cur(ev) and ev.get("l4h") is not None]
    R["VOL_baseline"] = stats(bl, "VOL baseline")
    pstat("VOL_baseline", R["VOL_baseline"])

    for lo, hi, lbl in [(1,3,"low"), (4,6,"mid"), (7,10,"high")]:
        v = [ev["l4h"] for ev in ev_v if bl_cur(ev) and lo <= (ev["vd_now"] or 0) <= hi
             and ev.get("l4h") is not None]
        R["VOL_dec_%s" % lbl] = stats(v, "vol_dec %d-%d LONG" % (lo, hi))
        pstat("VOL_dec_%s" % lbl, R["VOL_dec_%s" % lbl])

    for flag in [True, False]:
        v = [ev["l4h"] for ev in ev_v if bl_cur(ev) and ev.get("hi_vol") == flag
             and ev.get("l4h") is not None]
        R["VOL_alert_%s" % flag] = stats(v, "hi_vol_alert=%s LONG" % flag)
        pstat("VOL_alert_%s" % flag, R["VOL_alert_%s" % flag])

    v_comp = [ev["l4h"] for ev in ev_v
              if bl_cur(ev) and ev.get("vd_pre") is not None and ev.get("vd_now") is not None
              and ev["vd_pre"] <= 3 and ev["vd_now"] >= 7 and ev.get("l4h") is not None]
    R["VOL_compress_lo_to_hi"] = stats(v_comp, "vol pre<=3 then now>=7 LONG")
    pstat("VOL_compress_lo_to_hi", R["VOL_compress_lo_to_hi"])

    ev_exp = [ev for ev in ev_v if bl_cur(ev) and ev.get("vol_exp") is not None
              and ev.get("l4h") is not None]
    if ev_exp:
        exps = sorted(ev["vol_exp"] for ev in ev_exp)
        med  = exps[len(exps) // 2]
        R["VOL_high_exp"] = stats([ev["l4h"] for ev in ev_exp if ev["vol_exp"] >= med],
                                   "vol_expansion>=med(%+d) LONG" % int(med))
        R["VOL_low_exp"]  = stats([ev["l4h"] for ev in ev_exp if ev["vol_exp"] < med],
                                   "vol_expansion<med(%+d) LONG" % int(med))
        pstat("VOL_high_exp", R["VOL_high_exp"])
        pstat("VOL_low_exp",  R["VOL_low_exp"])

    rv_list = [(ev["rv5m"], ev["l4h"]) for ev in ev_v
               if bl_cur(ev) and ev.get("rv5m") is not None and ev.get("l4h") is not None]
    if rv_list:
        rv_list.sort()
        med_rv = rv_list[len(rv_list) // 2][0]
        R["VOL_rv_low"]  = stats([v for r, v in rv_list if r < med_rv],
                                  "rv5m<med(%.4f) LONG" % med_rv)
        R["VOL_rv_high"] = stats([v for r, v in rv_list if r >= med_rv],
                                  "rv5m>=med(%.4f) LONG" % med_rv)
        pstat("VOL_rv_low",  R["VOL_rv_low"])
        pstat("VOL_rv_high", R["VOL_rv_high"])

    return R

# ---------------------------------------------------------------------------
# OFI
# ---------------------------------------------------------------------------

def run_OFI(events):
    print("\n=== OFI ===")
    R = {}
    ev_o = [ev for ev in events if ev.get("ofi_pre_ratio") is not None]
    print("  With OFI data: %d" % len(ev_o))

    bl = [ev["l4h"] for ev in ev_o if bl_cur(ev) and ev.get("l4h") is not None]
    R["OFI_baseline"] = stats(bl, "OFI baseline")
    pstat("OFI_baseline", R["OFI_baseline"])

    vb = [ev["l4h"] for ev in ev_o if bl_cur(ev) and (ev["ofi_pre"] or 0) > 0
          and ev.get("l4h") is not None]
    vs = [ev["l4h"] for ev in ev_o if bl_cur(ev) and (ev["ofi_pre"] or 0) <= 0
          and ev.get("l4h") is not None]
    R["OFI_net_buy_pre"]  = stats(vb, "OFI pre-5m net buyers LONG")
    R["OFI_net_sell_pre"] = stats(vs, "OFI pre-5m net sellers LONG")
    pstat("OFI_net_buy_pre",  R["OFI_net_buy_pre"])
    pstat("OFI_net_sell_pre", R["OFI_net_sell_pre"])

    ratios = sorted((ev["ofi_pre_ratio"], ev["l4h"]) for ev in ev_o
                    if bl_cur(ev) and ev.get("ofi_pre_ratio") is not None
                    and ev.get("l4h") is not None)
    if ratios:
        n = len(ratios)
        q25 = ratios[n // 4][0]
        q75 = ratios[3 * n // 4][0]
        R["OFI_Q1_sell"] = stats([v for r, v in ratios if r < q25],
                                  "OFI ratio Q1 (most sellers) LONG")
        R["OFI_Q4_buy"]  = stats([v for r, v in ratios if r >= q75],
                                  "OFI ratio Q4 (most buyers) LONG")
        pstat("OFI_Q1_sell", R["OFI_Q1_sell"])
        pstat("OFI_Q4_buy",  R["OFI_Q4_buy"])

    v_buy_sil = [ev["l4h"] for ev in ev_o
                 if bl_cur(ev) and not ev["noisy"]
                 and (ev.get("ofi_post_ratio") or 0) > 0 and ev.get("l4h") is not None]
    R["OFI_silence_post_buy"] = stats(v_buy_sil, "silence + OFI post>0 (buyers filling) LONG")
    pstat("OFI_silence_post_buy", R["OFI_silence_post_buy"])

    return R

# ---------------------------------------------------------------------------
# State Transition 4-partition
# ---------------------------------------------------------------------------

def run_ST(events):
    print("\n=== ST ===")
    R = {}
    for noisy_v, btc_v, lbl in [
        (False, False, "sil_nobtc"),
        (False, True,  "sil_btc"),
        (True,  False, "noisy_nobtc"),
        (True,  True,  "noisy_btc"),
    ]:
        va = [ev["l4h"] for ev in events
              if ev["noisy"] == noisy_v and (ev["btc_5d_30m"] >= 1_000_000) == btc_v
              and ev.get("l4h") is not None]
        vg = [ev["l4h"] for ev in events
              if bl_cur(ev) and ev["noisy"] == noisy_v
              and (ev["btc_5d_30m"] >= 1_000_000) == btc_v and ev.get("l4h") is not None]
        R["ST_%s_all"  % lbl] = stats(va, "ST all   %s LONG" % lbl)
        R["ST_%s_gate" % lbl] = stats(vg, "ST gated %s LONG" % lbl)
        pstat("ST_%s_all"  % lbl, R["ST_%s_all"  % lbl])
        pstat("ST_%s_gate" % lbl, R["ST_%s_gate" % lbl])

    vs = [ev["sn_2h"] for ev in events
          if ev["noisy"] and ev["btc_5d_30m"] >= 1_000_000
          and ev.get("sn_2h") is not None]
    R["ST_noisy_btc_SHORT"] = stats(vs, "ST noisy+BTC1M → SHORT noisy-entry 2h")
    pstat("ST_noisy_btc_SHORT", R["ST_noisy_btc_SHORT"])

    for lo_m, hi_m, lbl in [(1,5,"1_5m"), (5,15,"5_15m"), (15,30,"15_30m")]:
        lo_ms = lo_m * 60_000
        hi_ms = hi_m * 60_000
        vt = [ev["l4h"] for ev in events
              if ev["noisy"] and ev.get("noisy_ts") is not None
              and lo_ms <= (ev["noisy_ts"] - ev["ts"]) < hi_ms
              and ev.get("l4h") is not None]
        R["ST_noisy_delay_%s" % lbl] = stats(vt, "noisy delay %s LONG 4h" % lbl)
        pstat("ST_noisy_delay_%s" % lbl, R["ST_noisy_delay_%s" % lbl])

    return R

# ---------------------------------------------------------------------------
# BTC Lead-Lag
# ---------------------------------------------------------------------------

def run_LAG(events):
    print("\n=== LAG ===")
    R = {}
    ev_lag  = [ev for ev in events if ev.get("btc_delta") is not None]
    ev_none = [ev for ev in events if ev.get("btc_delta") is None]

    v_no = [ev["l4h"] for ev in ev_none if bl_cur(ev) and ev.get("l4h") is not None]
    R["LAG_no_btc"] = stats(v_no, "No BTC ≥500K in ±30m LONG")
    pstat("LAG_no_btc", R["LAG_no_btc"])

    v_led = [ev["l4h"] for ev in ev_lag if bl_cur(ev) and ev["btc_delta"] < 0
             and ev.get("l4h") is not None]
    v_eth = [ev["l4h"] for ev in ev_lag if bl_cur(ev) and ev["btc_delta"] > 0
             and ev.get("l4h") is not None]
    R["LAG_btc_led"] = stats(v_led, "BTC led (delta<0, BTC before ETH) LONG")
    R["LAG_eth_led"] = stats(v_eth, "ETH led (delta>0, BTC after ETH) LONG")
    pstat("LAG_btc_led", R["LAG_btc_led"])
    pstat("LAG_eth_led", R["LAG_eth_led"])

    for lo_m, hi_m in [(-5,0), (-15,-5), (-30,-15)]:
        lms, hms = lo_m * 60_000, hi_m * 60_000
        vb = [ev["l4h"] for ev in ev_lag if bl_cur(ev)
              and lms <= ev["btc_delta"] < hms and ev.get("l4h") is not None]
        lbl = "LAG_btc_%dto%dm" % (-hi_m, -lo_m)
        R[lbl] = stats(vb, "BTC %d-%dm before ETH LONG" % (-hi_m, -lo_m))
        pstat(lbl, R[lbl])

    bps_list = sorted((ev["btc5m"], ev["l4h"]) for ev in events
                      if bl_cur(ev) and ev.get("l4h") is not None)
    if bps_list:
        med = bps_list[len(bps_list) // 2][0]
        R["LAG_btc5m_down"] = stats([v for b, v in bps_list if b < med],
                                     "btc5m<med(%.1f bps) LONG" % med)
        R["LAG_btc5m_up"]   = stats([v for b, v in bps_list if b >= med],
                                     "btc5m>=med(%.1f bps) LONG" % med)
        pstat("LAG_btc5m_down", R["LAG_btc5m_down"])
        pstat("LAG_btc5m_up",   R["LAG_btc5m_up"])

    return R

# ---------------------------------------------------------------------------
# Cascade Density
# ---------------------------------------------------------------------------

def run_DEN(events):
    print("\n=== DEN ===")
    R = {}
    for d in [0, 1, 2]:
        v = [ev["l4h"] for ev in events if bl_cur(ev) and ev["density_24h"] == d
             and ev.get("l4h") is not None]
        R["DEN_%d" % d] = stats(v, "density_24h=%d cur gate LONG" % d)
        pstat("DEN_%d" % d, R["DEN_%d" % d])

    v3 = [ev["l4h"] for ev in events if bl_cur(ev) and ev["density_24h"] >= 3
          and ev.get("l4h") is not None]
    R["DEN_3plus"] = stats(v3, "density_24h>=3 LONG")
    pstat("DEN_3plus", R["DEN_3plus"])

    v_clean = [ev["l4h"] for ev in events if bl_cur(ev) and ev["density_24h"] == 0
               and ev.get("l4h") is not None]
    v_crowd = [ev["l4h"] for ev in events if bl_cur(ev) and ev["density_24h"] >= 2
               and ev.get("l4h") is not None]
    R["DEN_clean_or"] = stats([ev["l4h"] for ev in events
                                if bl_or(ev) and ev["density_24h"] == 0
                                and ev.get("l4h") is not None], "density=0 OR gate LONG")
    R["DEN_veto_2up"] = stats([ev["l4h"] for ev in events
                                if bl_cur(ev) and ev["density_24h"] < 2
                                and ev.get("l4h") is not None], "density<2 veto LONG")
    pstat("DEN_clean_or",  R["DEN_clean_or"])
    pstat("DEN_veto_2up",  R["DEN_veto_2up"])

    return R

# ---------------------------------------------------------------------------
# Cascade Exhaustion
# ---------------------------------------------------------------------------

def run_EX(events):
    print("\n=== EX ===")
    R = {}
    for p in [0, 1, 2]:
        v = [ev["l4h"] for ev in events if bl_cur(ev) and ev["prebuildup"] == p
             and ev.get("l4h") is not None]
        R["EX_pre%d" % p] = stats(v, "prebuildup=%d LONG" % p)
        pstat("EX_pre%d" % p, R["EX_pre%d" % p])

    v3 = [ev["l4h"] for ev in events if bl_cur(ev) and ev["prebuildup"] >= 3
          and ev.get("l4h") is not None]
    R["EX_pre3up"] = stats(v3, "prebuildup>=3 LONG")
    pstat("EX_pre3up", R["EX_pre3up"])

    R["EX_any_pre"] = stats([ev["l4h"] for ev in events if bl_cur(ev) and ev["prebuildup"] > 0
                              and ev.get("l4h") is not None], "prebuildup>0 LONG")
    R["EX_no_pre"]  = stats([ev["l4h"] for ev in events if bl_cur(ev) and ev["prebuildup"] == 0
                              and ev.get("l4h") is not None], "prebuildup=0 LONG")
    pstat("EX_any_pre", R["EX_any_pre"])
    pstat("EX_no_pre",  R["EX_no_pre"])

    ev_pb = [ev for ev in events if bl_cur(ev) and ev["prebuildup"] > 0
             and ev.get("l4h") is not None]
    if ev_pb:
        rn_list = sorted(ev["rn"] for ev in ev_pb)
        med_rn  = rn_list[len(rn_list) // 2]
        R["EX_climax"]  = stats([ev["l4h"] for ev in ev_pb if ev["rn"] >= med_rn],
                                  "prebuildup>0 high rn (climax) LONG")
        R["EX_cont"]    = stats([ev["l4h"] for ev in ev_pb if ev["rn"] < med_rn],
                                  "prebuildup>0 low rn (continuation) LONG")
        pstat("EX_climax", R["EX_climax"])
        pstat("EX_cont",   R["EX_cont"])

    return R

# ---------------------------------------------------------------------------
# Second Wave + Echo Window
# ---------------------------------------------------------------------------

def run_SW(events):
    print("\n=== SW + EW ===")
    R = {}
    bl = [ev["l4h"] for ev in events if bl_or(ev) and ev.get("l4h") is not None]
    R["SW_baseline_or"] = stats(bl, "Baseline OR gate LONG")
    pstat("SW_baseline_or", R["SW_baseline_or"])

    for echo_k, lo_m, hi_m in [
        ("echo_10_30",  10, 30),
        ("echo_20_60",  20, 60),
        ("echo_30_90",  30, 90),
        ("echo_45_120", 45, 120),
        ("echo_60_180", 60, 180),
    ]:
        lbl = "%d_%dm" % (lo_m, hi_m)
        v_all  = [ev["l4h"] for ev in events if bl_nobtc(ev) and ev.get(echo_k)
                  and ev.get("l4h") is not None]
        v_sil  = [ev["l4h"] for ev in events if bl_nobtc(ev) and ev.get(echo_k)
                  and not ev["noisy"] and ev.get("l4h") is not None]
        v_or   = [ev["l4h"] for ev in events if bl_or(ev) and ev.get(echo_k)
                  and not ev["noisy"] and ev.get("l4h") is not None]
        R["EW_%s_all" % echo_k]    = stats(v_all, "Echo %s all LONG" % lbl)
        R["EW_%s_sil" % echo_k]    = stats(v_sil, "Echo %s + sil LONG" % lbl)
        R["EW_%s_or_sil" % echo_k] = stats(v_or,  "Echo %s OR+sil LONG" % lbl)
        pstat("EW_%s_sil"    % echo_k, R["EW_%s_sil"    % echo_k])
        pstat("EW_%s_or_sil" % echo_k, R["EW_%s_or_sil" % echo_k])

    for key, lbl in [("sw_90_180","90-180m"), ("sw_180_360","180-360m")]:
        v_has = [ev["l4h"] for ev in events if bl_cur(ev) and ev.get(key, 0) >= ETH_THRESH
                 and ev.get("l4h") is not None]
        v_no  = [ev["l4h"] for ev in events if bl_cur(ev) and ev.get(key, 0) < ETH_THRESH
                 and ev.get("l4h") is not None]
        R["SW_follow_%s" % lbl]    = stats(v_has, "Has follow %s LONG" % lbl)
        R["SW_no_follow_%s" % lbl] = stats(v_no,  "No follow %s LONG"  % lbl)
        pstat("SW_follow_%s"    % lbl, R["SW_follow_%s" % lbl])
        pstat("SW_no_follow_%s" % lbl, R["SW_no_follow_%s" % lbl])

    return R

# ---------------------------------------------------------------------------
# SHORT_NOISY hold sweep
# ---------------------------------------------------------------------------

def run_SH(events):
    print("\n=== SH: SHORT_NOISY hold sweep ===")
    R = {}
    for thr, tl in [(500_000,"500K"), (1_000_000,"1M"), (2_000_000,"2M")]:
        for hk, hl in [("sn_90m","90m"), ("sn_2h","2h"), ("sn_3h","3h"), ("sn_4h","4h")]:
            v = [ev[hk] for ev in events
                 if ev["noisy"] and ev.get("btc_5d_30m", 0) >= thr and ev.get(hk) is not None]
            k = "SH_%s_%s" % (tl, hl)
            R[k] = stats(v, "SHORT BTC>=%s %s" % (tl, hl))
            pstat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# LONG Gate comparison
# ---------------------------------------------------------------------------

def run_GR(events):
    print("\n=== GR: LONG gate btc4h vs btc7d ===")
    R = {}

    def base(ev):
        return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["blocked"]
                and ev["dow"] not in {0, 2} and ev["sync_k"] < 200_000 and ev["score"] >= 2)

    configs = [
        ("GR_btc7d",        lambda ev: base(ev) and ev["btc7d"] < 0),
        ("GR_btc4h",        lambda ev: base(ev) and ev["btc4h"] < 0),
        ("GR_btc3d",        lambda ev: base(ev) and ev["btc3d"] < 0),
        ("GR_or_4h_7d",     lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("GR_and_4h_7d",    lambda ev: base(ev) and ev["btc4h"] < 0 and ev["btc7d"] < 0),
        ("GR_4h_only_adds", lambda ev: base(ev) and ev["btc4h"] < 0 and ev["btc7d"] >= 0),
        ("GR_7d_only_adds", lambda ev: base(ev) and ev["btc7d"] < 0 and ev["btc4h"] >= 0),
        ("GR_no_btc",       lambda ev: base(ev)),
    ]
    for k, fn in configs:
        v = [ev["l4h"] for ev in events if fn(ev) and ev.get("l4h") is not None]
        R[k] = stats(v, k)
        pstat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# Failed Bounce SHORT
# ---------------------------------------------------------------------------

def run_FB(events):
    print("\n=== FB: Failed bounce SHORT ===")
    R = {}
    v_fc   = [ev["l4h"] for ev in events if bl_cur(ev) and ev["failed_cascade"]
              and ev.get("l4h") is not None]
    v_real = [ev["l4h"] for ev in events if bl_cur(ev) and not ev["failed_cascade"]
              and ev.get("l4h") is not None]
    R["FB_fc_long"]   = stats(v_fc,   "failed_cascade=True LONG 4h")
    R["FB_real_long"] = stats(v_real, "failed_cascade=False LONG 4h")
    pstat("FB_fc_long",   R["FB_fc_long"])
    pstat("FB_real_long", R["FB_real_long"])

    v_fcs = [ev["s2h"] for ev in events if ev["failed_cascade"] and ev.get("s2h") is not None]
    v_fcns = [ev["sn_2h"] for ev in events if ev["failed_cascade"] and ev["noisy"]
              and ev.get("sn_2h") is not None]
    v_fcnb = [ev["sn_2h"] for ev in events
              if ev["failed_cascade"] and ev["noisy"] and ev.get("btc_5d_30m", 0) >= 1_000_000
              and ev.get("sn_2h") is not None]
    R["FB_short_anchor"]  = stats(v_fcs,  "failed_cascade → SHORT anchor 2h")
    R["FB_short_noisy"]   = stats(v_fcns, "failed_cascade + noisy → SHORT 2h")
    R["FB_short_btc_con"] = stats(v_fcnb, "failed_cascade + noisy + BTC1M → SHORT 2h")
    pstat("FB_short_anchor",  R["FB_short_anchor"])
    pstat("FB_short_noisy",   R["FB_short_noisy"])
    pstat("FB_short_btc_con", R["FB_short_btc_con"])

    ev_fc = [ev for ev in events if ev["failed_cascade"] and ev.get("fc_bps") is not None
             and ev.get("s2h") is not None]
    if len(ev_fc) >= 4:
        fc_sorted = sorted(ev["fc_bps"] for ev in ev_fc)
        med_fc    = fc_sorted[len(fc_sorted) // 2]
        R["FB_big_bounce"] = stats([ev["s2h"] for ev in ev_fc if ev["fc_bps"] >= med_fc],
                                    "fc big bounce >=%.0fbps → SHORT" % med_fc)
        R["FB_sm_bounce"]  = stats([ev["s2h"] for ev in ev_fc if ev["fc_bps"] < med_fc],
                                    "fc small bounce <%.0fbps → SHORT" % med_fc)
        pstat("FB_big_bounce", R["FB_big_bounce"])
        pstat("FB_sm_bounce",  R["FB_sm_bounce"])

    return R

# ---------------------------------------------------------------------------
# Vacuum
# ---------------------------------------------------------------------------

def run_VAC(events):
    print("\n=== VAC ===")
    R = {}
    bl = [ev["l4h"] for ev in events if bl_cur(ev) and ev.get("l4h") is not None]
    R["VAC_baseline"] = stats(bl, "VAC baseline")
    pstat("VAC_baseline", R["VAC_baseline"])

    v_cl = [ev["l4h"] for ev in events if bl_cur(ev) and ev.get("vac_clean")
            and ev.get("l4h") is not None]
    v_nc = [ev["l4h"] for ev in events if bl_cur(ev) and not ev.get("vac_clean")
            and ev.get("l4h") is not None]
    R["VAC_clean_30m"]    = stats(v_cl, "No sell ≥10K in 0-30m LONG")
    R["VAC_not_clean_30m"] = stats(v_nc, "Some sell ≥10K in 0-30m LONG")
    pstat("VAC_clean_30m",     R["VAC_clean_30m"])
    pstat("VAC_not_clean_30m", R["VAC_not_clean_30m"])

    vac15_list = sorted((ev["vac_15m"], ev["l4h"]) for ev in events
                        if bl_cur(ev) and ev.get("l4h") is not None)
    if vac15_list:
        med_v = vac15_list[len(vac15_list) // 2][0]
        R["VAC_quiet_15m"] = stats([v for s, v in vac15_list if s < med_v],
                                    "vac_15m<med quiet LONG")
        R["VAC_loud_15m"]  = stats([v for s, v in vac15_list if s >= med_v],
                                    "vac_15m>=med loud LONG")
        pstat("VAC_quiet_15m", R["VAC_quiet_15m"])
        pstat("VAC_loud_15m",  R["VAC_loud_15m"])

    v_ofi_vac = [ev["l4h"] for ev in events
                 if bl_cur(ev) and ev.get("vac_clean") and ev.get("ofi_post_ratio") is not None
                 and ev["ofi_post_ratio"] > 0 and ev.get("l4h") is not None]
    R["VAC_ofi_clean"] = stats(v_ofi_vac, "clean vac + OFI post>0 LONG")
    pstat("VAC_ofi_clean", R["VAC_ofi_clean"])

    return R

# ---------------------------------------------------------------------------
# Symmetry SHORT
# ---------------------------------------------------------------------------

def run_SYM(events):
    print("\n=== SYM ===")
    R = {}
    bl_s = [ev["sn_2h"] for ev in events
            if ev["noisy"] and ev.get("btc_5d_30m", 0) >= 1_000_000
            and ev.get("sn_2h") is not None]
    R["SYM_short_base"] = stats(bl_s, "SHORT BTC1M baseline 2h")
    pstat("SYM_short_base", R["SYM_short_base"])

    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    for d, dn in dow_names.items():
        v = [ev["sn_2h"] for ev in events
             if ev["noisy"] and ev.get("btc_5d_30m", 0) >= 1_000_000
             and ev["dow"] == d and ev.get("sn_2h") is not None]
        if v:
            R["SYM_dow_%s" % dn] = stats(v, "SHORT BTC1M dow=%s" % dn)
            pstat("SYM_dow_%s" % dn, R["SYM_dow_%s" % dn])

    for sc in [3, 4, 5]:
        v = [ev["sn_2h"] for ev in events
             if ev["noisy"] and ev.get("btc_5d_30m", 0) >= 1_000_000
             and ev["score"] >= sc and ev.get("sn_2h") is not None]
        R["SYM_score%d" % sc] = stats(v, "SHORT BTC1M score>=%d" % sc)
        pstat("SYM_score%d" % sc, R["SYM_score%d" % sc])

    # wrong direction control
    v_l = [ev["l4h"] for ev in events
           if ev["noisy"] and ev.get("btc_5d_30m", 0) >= 1_000_000
           and ev.get("l4h") is not None]
    R["SYM_noisy_btc_LONG"] = stats(v_l, "noisy+BTC1M → LONG (wrong dir ctrl)")
    pstat("SYM_noisy_btc_LONG", R["SYM_noisy_btc_LONG"])

    return R

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def make_md(R):
    lines = [
        "# S34 Research Ideas V2 — Sonuçlar",
        "**Tarih:** 2026-07-01  **Evren:** 596 anchor, 4.5 ay, FEE=5bps", "",
    ]
    sections = [
        ("VOL", "Vol Rejim Filtresi"),
        ("OFI", "Order Flow Imbalance (agg_trades 5m pre)"),
        ("ST",  "State Transition 4-Partition"),
        ("LAG", "BTC Lead-Lag Timing Delta"),
        ("DEN", "Cascade Density Quartile"),
        ("EX",  "Cascade Exhaustion (prebuildup)"),
        ("EW",  "Echo Window Sweep"),
        ("SW",  "Second Wave (follow-on)"),
        ("SH",  "SHORT_NOISY Hold Time Sweep"),
        ("GR",  "LONG Gate: btc4h vs btc7d"),
        ("FB",  "Failed Bounce → SHORT"),
        ("VAC", "Liquidity Vacuum Depth"),
        ("SYM", "SHORT Simetri DOW × Score"),
    ]
    hdr = "| Test | N | /ay | WR | Avg bps | T3R | mc_p | HO avg |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    for pfx, title in sections:
        keys = [k for k in R if k.startswith(pfx + "_")]
        if not keys:
            continue
        lines += ["", "## %s" % title, "", hdr, sep]
        for k in keys:
            v = R[k]
            if not v or v.get("n", 0) == 0:
                continue
            ho = v.get("ho_avg")
            ho_s = ("%+.1f" % ho) if ho is not None else "-"
            lines.append("| %-44s | %4d | %5.1f | %5.1f%% | %+7.1f | %+8.1f | %s | %s |" % (
                k[:44], v["n"], v.get("per_month", 0), v["wr"],
                v["avg"], v["t3r"], v.get("mc_p", "?"), ho_s))
    lines += ["", "---", "*Script: tools/research_s34_ideas_v2.py*"]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== S34 Research Ideas V2 ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA temp_store=MEMORY")

        now_ms   = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - LOOKBACK_MS

        print("Loading ETH SELL liqs ...")
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        print("  liqs: %d" % len(liqs))

        print("Reconstructing anchors ...")
        anchors = reconstruct_anchors(
            liqs, bucket_sec=300, min_gap_sec=900,
            thresholds=(ETH_THRESH,), accel_window_sec=30
        )
        print("  anchors: %d" % len(anchors))

        print("Loading mark prices ...")
        marks_eth = load_mark_index(conn, "ETHUSDT")

        print("Loading vol_state ...")
        vol_rows = load_vol_state(conn)

        print("Loading BTC big sells ...")
        btc_sell_ts = load_btc_big_sells(conn, start_ms, now_ms, thr=500_000)
        print("  BTC SELL >=500K: %d" % len(btc_sell_ts))

        print("Building events ...")
        events = build_events(conn, anchors, marks_eth, vol_rows, btc_sell_ts)
        print("  events: %d" % len(events))

        add_outcomes(events, marks_eth)
        add_ofi(conn, events)

        R = {}
        R.update(run_VOL(events))
        R.update(run_OFI(events))
        R.update(run_ST(events))
        R.update(run_LAG(events))
        R.update(run_DEN(events))
        R.update(run_EX(events))
        R.update(run_SW(events))
        R.update(run_SH(events))
        R.update(run_GR(events))
        R.update(run_FB(events))
        R.update(run_VAC(events))
        R.update(run_SYM(events))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    print("JSON: %s" % OUT_JSON)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(R))
    print("MD:   %s" % OUT_MD)

    print("\n=== TOP SIGNALS (N>=8, mc_p<=0.05) ===")
    good = [(k, v) for k, v in R.items()
            if v and v.get("n", 0) >= 8 and v.get("mc_p") is not None and v["mc_p"] <= 0.05]
    good.sort(key=lambda x: x[1].get("avg", 0), reverse=True)
    for k, v in good[:25]:
        pstat(k, v)
    print("Done.")


if __name__ == "__main__":
    main()
