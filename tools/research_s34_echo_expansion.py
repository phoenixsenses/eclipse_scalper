"""S34 Echo Expansion Research — "Echo geniş" (17/ay, WR 68%) üstüne inşa.

Baz sinyal: echo_30_90 + silence gated LONG 4h
  (önceki 30-90dk içinde ETH SELL 200K cascade + current silence + not bull
   + not EUROPE + dow not in {Mon,Wed}) → LONG 4h hold.
  Referans: N~76, 16.8/ay, WR 68.4% (S34_FREQ_EXPANSION_V1).

İki hedef:
  1) FREKANS ARTIR  — echo penceresini genişlet, dow bloğunu gevşet, union band,
     quasi-silence, echo-count gevşetme.
  2) TAIL KORU / WR ARTIR — rejim (btc4h/7d/3d), vol_decile low, prebuildup,
     sync, score, btc5m, vacuum, OFI post-buyers, btc/eth ratio veto,
     ve STOP simülasyonu (path-min ile gerçek stop-out).

Robustness: holdout (70/30), 5-fold walk-forward, no-overlap execution,
max-stat permutation (MC).

Çıktı:
  reports/research/s34/S34_ECHO_EXPANSION.json
  reports/research/s34/S34_ECHO_EXPANSION.md
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
OUT_JSON     = OUT_DIR / "S34_ECHO_EXPANSION.json"
OUT_MD       = OUT_DIR / "S34_ECHO_EXPANSION.md"

ETH_THRESH   = 200_000.0
PROP_THRESH  =  50_000.0
LOOKBACK_MS  = 400 * 24 * 3600_000
FEE_BPS      = 5.0
MC_ITER      = 2000
HOLD_MS      = 4 * 3600_000
TOTAL_MONTHS = 4.5

# Echo pencereleri (dk) — geniş tarama
ECHO_WINDOWS = [
    (10, 30), (15, 45), (20, 60), (30, 90),
    (20, 90), (20, 120), (30, 120), (30, 150),
    (15, 120), (20, 150), (45, 120), (45, 150), (60, 180),
]

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

def ofi_post_ratio(conn, sym, lo_ms, hi_ms):
    row = conn.execute(
        "SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0.0 END),"
        "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0.0 END) "
        "FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
        (sym, lo_ms, hi_ms)).fetchone()
    if not row or row[0] is None:
        return None
    buy_n, sell_n = float(row[0]), float(row[1])
    total = buy_n + sell_n
    return (buy_n - sell_n) / total if total > 0 else 0.0

def compute_score(conn, ts, sync_k, n2h):
    b4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30)
            + int(13 <= hour < 21) + int(sync_k >= 200_000))

def load_vol_state(conn):
    return conn.execute(
        "SELECT ts_ms, rv_5m, vol_decile, high_vol_alert FROM vol_state "
        "WHERE symbol='ETHUSDT' ORDER BY ts_ms").fetchall()

def vol_at(vol_rows, vol_ts, ts_ms):
    if not vol_rows:
        return None
    idx = bisect.bisect_right(vol_ts, ts_ms) - 1
    return vol_rows[idx] if idx >= 0 else None

def echo_check(ts_list, ts, lo_min, hi_min):
    lo_ms = ts - hi_min * 60_000
    hi_ms = ts - lo_min * 60_000
    lo_i = bisect.bisect_left(ts_list, lo_ms)
    hi_i = bisect.bisect_left(ts_list, hi_ms)
    for i in range(lo_i, hi_i):
        if ts_list[i] != ts:
            return True
    return False

def echo_count(ts_list, ts, lo_min, hi_min):
    lo_ms = ts - hi_min * 60_000
    hi_ms = ts - lo_min * 60_000
    lo_i = bisect.bisect_left(ts_list, lo_ms)
    hi_i = bisect.bisect_left(ts_list, hi_ms)
    return sum(1 for i in range(lo_i, hi_i) if ts_list[i] != ts)

# ---------------------------------------------------------------------------
# Outcomes (with optional path-min stop)
# ---------------------------------------------------------------------------

def long_out(marks, ts, hold_ms=HOLD_MS, stop_bps=None):
    r0 = marks.at_or_after(ts)
    if not r0 or float(r0[1]) <= 0:
        return None
    entry = float(r0[1])
    if stop_bps is not None:
        stop_px = entry * (1.0 - stop_bps / 10_000.0)
        for _, px in marks.slice_range(r0[0], ts + hold_ms):
            if float(px) <= stop_px:
                return round(-stop_bps - FEE_BPS, 2)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r1:
        return None
    return round((float(r1[1]) - entry) / entry * 10_000 - FEE_BPS, 2)

# ---------------------------------------------------------------------------
# Stats + robustness
# ---------------------------------------------------------------------------

def _mc_p(vals, avg):
    if len(vals) < 4:
        return None
    rng = random.Random(0)
    ct = sum(1 for _ in range(MC_ITER)
             if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / len(vals) >= avg)
    return round(ct / MC_ITER, 3)

def wf_folds(vals, k=5):
    """vals time-ordered → kaç fold pozitif sum."""
    n = len(vals)
    if n < k:
        return None
    pos = 0
    for i in range(k):
        seg = vals[i * n // k:(i + 1) * n // k]
        if seg and sum(seg) > 0:
            pos += 1
    return "%d/%d" % (pos, k)

def stats(vals, label="", months=None):
    m = months or TOTAL_MONTHS
    if not vals:
        return {"label": label, "n": 0}
    n = len(vals)
    wins = sum(1 for v in vals if v > 0)
    sv = sorted(vals)
    t3 = sorted(vals, reverse=True)
    t3r = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    avg = sum(vals) / n
    cut = max(1, int(n * 0.7))
    ho = vals[cut:]
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "sum": round(sum(vals), 1),
        "t3r": round(t3r, 1),
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in vals if v < -100),
        "per_month": round(n / m, 1),
        "mc_p": _mc_p(vals, avg),
        "wf": wf_folds(vals),
        "ho_n": len(ho),
        "ho_avg": round(sum(ho) / len(ho), 1) if ho else None,
        "ho_wr": round(100 * sum(1 for v in ho if v > 0) / len(ho), 1) if ho else None,
        "ho_sum": round(sum(ho), 1) if ho else None,
    }

def no_overlap_vals(pairs, hold_ms=HOLD_MS):
    """pairs: [(ts, val)] → tek pozisyon (non-overlapping) val listesi, zaman sıralı."""
    busy_until = -1
    out = []
    for ts, val in sorted(pairs):
        if ts >= busy_until:
            out.append(val)
            busy_until = ts + hold_ms
    return out

def pstat(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-40s N=0" % k[:40])
        return
    print("    %-40s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s worst=%-7s tail=%-2d mc_p=%s wf=%s" % (
        k[:40], v["n"], v.get("per_month", 0),
        str(v["wr"]) + "%", str(v["avg"]) + "bps",
        str(v.get("worst")), v.get("tail_n", 0), v.get("mc_p", "?"), v.get("wf")))

# ---------------------------------------------------------------------------
# Build events
# ---------------------------------------------------------------------------

def build_events(conn, anchors, marks_eth, vol_rows):
    events = []
    ts_list = sorted(int(a.anchor_ts_ms) for a in anchors)
    vol_ts = [r[0] for r in vol_rows]
    total = len(anchors)
    print("  Building %d events ..." % total)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print("    [%d/%d]" % (i + 1, total))
        ts = int(anc.anchor_ts_ms)
        rn = float(anc.running_notional)
        p0 = marks_eth.at_or_after(ts)
        if p0 is None:
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

        prebuildup = liq_cnt(conn, "ETHUSDT", "SELL", ts - 30*60_000, ts - 1000, PROP_THRESH)

        # noisy: ilk follow-on 1-30dk; quiet_15: ilk 15dk follow-on YOK
        noisy_ts   = liq_first_ts(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 30*60_000, PROP_THRESH)
        noisy      = (noisy_ts is not None)
        noisy15_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 15*60_000, PROP_THRESH)
        quiet_15   = (noisy15_ts is None)

        vac_clean  = liq_cnt(conn, "ETHUSDT", "SELL", ts + 1000, ts + 30*60_000, 10_000) == 0

        # veto: eş zamanlı BTC cascade / ETH cascade oranı
        btc_conc = liq_max(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts + 10*60_000)
        be_ratio = (btc_conc / rn) if rn > 0 else 0.0

        # OFI post 0-10m buyers
        ofi_post = ofi_post_ratio(conn, "ETHUSDT", ts, ts + 10*60_000)

        # vol_decile
        vs = vol_at(vol_rows, vol_ts, ts)
        vd_now = int(vs[2]) if vs and vs[2] is not None else None

        # echo flags + counts
        echo = {}
        ecnt = {}
        for lo_m, hi_m in ECHO_WINDOWS:
            key = "e%d_%d" % (lo_m, hi_m)
            echo[key] = echo_check(ts_list, ts, lo_m, hi_m)
            ecnt[key] = echo_count(ts_list, ts, lo_m, hi_m)

        events.append({
            "ts": ts, "rn": rn,
            "sync_k": sync_k, "n2h": n2h, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d, "btc5m": btc5m,
            "eth1h": eth1h, "bull": bull, "sess": sess, "dow": dow,
            "hour": hour, "blocked": blocked,
            "prebuildup": prebuildup,
            "noisy": noisy, "quiet_15": quiet_15,
            "vac_clean": vac_clean, "be_ratio": be_ratio,
            "ofi_post": ofi_post, "vd_now": vd_now,
            "echo": echo, "ecnt": ecnt,
        })
    events.sort(key=lambda e: e["ts"])
    print("  Computing outcomes (+stops) ...")
    for ev in events:
        ts = ev["ts"]
        ev["l4h"]       = long_out(marks_eth, ts)
        ev["l4h_s150"]  = long_out(marks_eth, ts, stop_bps=150.0)
        ev["l4h_s100"]  = long_out(marks_eth, ts, stop_bps=100.0)
        ev["l4h_s75"]   = long_out(marks_eth, ts, stop_bps=75.0)
        ev["l4h_s50"]   = long_out(marks_eth, ts, stop_bps=50.0)
        ev["l4h_s200"]  = long_out(marks_eth, ts, stop_bps=200.0)
    return events

# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def echo_base(ev, echo_key, dow_block=True):
    """Echo geniş baz: echo + silence + not bull + not EU (+ Mon/Wed blok)."""
    ok = (not ev["bull"] and ev["sess"] != "EUROPE"
          and not ev["noisy"] and ev["echo"].get(echo_key))
    if dow_block:
        ok = ok and ev["dow"] not in {0, 2}
    return bool(ok)

def vals_for(events, gate_fn, out_key="l4h"):
    return [ev[out_key] for ev in events if gate_fn(ev) and ev.get(out_key) is not None]

def pairs_for(events, gate_fn, out_key="l4h"):
    return [(ev["ts"], ev[out_key]) for ev in events
            if gate_fn(ev) and ev.get(out_key) is not None]

# ---------------------------------------------------------------------------
# FREQ: frekans artırma
# ---------------------------------------------------------------------------

def run_FREQ(events, months):
    print("\n=== FREQ: frekans artırma (echo genişletme) ===")
    R = {}

    # F1: echo penceresi taraması (silence gated) — baz = e30_90
    print("  F1: echo penceresi taraması")
    for lo_m, hi_m in ECHO_WINDOWS:
        key = "e%d_%d" % (lo_m, hi_m)
        v = vals_for(events, lambda ev, k=key: echo_base(ev, k))
        R["F1_%s" % key] = stats(v, "echo %d-%dm silence gated" % (lo_m, hi_m), months)
        pstat("F1_%s" % key, R["F1_%s" % key])

    # F2: dow bloğu kaldır (en geniş iyi pencere = e30_90 üstünde)
    print("  F2: dow (Mon/Wed) bloğu kaldır")
    for key in ["e30_90", "e30_120", "e30_150", "e20_120"]:
        v = vals_for(events, lambda ev, k=key: echo_base(ev, k, dow_block=False))
        R["F2_%s_nodow" % key] = stats(v, "echo %s silence, dow blok YOK" % key, months)
        pstat("F2_%s_nodow" % key, R["F2_%s_nodow" % key])

    # F3: union band — 20-120m herhangi bir prior cascade
    print("  F3: union band (herhangi prior cascade)")
    def union_gate(ev, keys, dow_block=True):
        ok = (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
              and any(ev["echo"].get(k) for k in keys))
        if dow_block:
            ok = ok and ev["dow"] not in {0, 2}
        return bool(ok)
    for name, keys in [
        ("union_20_120", ["e20_60", "e30_90", "e45_120"]),
        ("union_15_150", ["e15_45", "e30_90", "e45_150"]),
        ("union_wide",   ["e20_60", "e30_90", "e45_120", "e60_180"]),
    ]:
        v = vals_for(events, lambda ev, ks=keys: union_gate(ev, ks))
        R["F3_%s" % name] = stats(v, "union %s silence gated" % name, months)
        pstat("F3_%s" % name, R["F3_%s" % name])

    # F4: quasi-silence — ilk 15dk sessiz yeterli (15-30dk noisy tolere)
    print("  F4: quasi-silence (quiet_15)")
    def quasi_gate(ev, key):
        return (not ev["bull"] and ev["sess"] != "EUROPE"
                and ev["dow"] not in {0, 2} and ev["quiet_15"]
                and ev["echo"].get(key))
    for key in ["e30_90", "e30_120", "e20_120"]:
        v = vals_for(events, lambda ev, k=key: quasi_gate(ev, k))
        R["F4_%s_quiet15" % key] = stats(v, "echo %s quiet15 (quasi-silence)" % key, months)
        pstat("F4_%s_quiet15" % key, R["F4_%s_quiet15" % key])

    # F5: echo-count gevşetme — >=1 (baz) vs >=2 prior cascade
    print("  F5: echo count >=1 vs >=2")
    for key in ["e30_90", "e30_120"]:
        for c in [1, 2]:
            v = vals_for(events, lambda ev, k=key, cc=c:
                         echo_base(ev, k) and ev["ecnt"].get(k, 0) >= cc)
            R["F5_%s_ge%d" % (key, c)] = stats(v, "echo %s count>=%d" % (key, c), months)
            pstat("F5_%s_ge%d" % (key, c), R["F5_%s_ge%d" % (key, c)])

    return R

# ---------------------------------------------------------------------------
# TAILWR: tail koru / WR artır (echo geniş baz üstünde katmanlı filtre)
# ---------------------------------------------------------------------------

def run_TAILWR(events, months, base_key="e30_90"):
    print("\n=== TAILWR: tail koru / WR artır (baz=%s) ===" % base_key)
    R = {}

    base = lambda ev: echo_base(ev, base_key)
    R["T0_base"] = stats(vals_for(events, base), "echo %s baz" % base_key, months)
    pstat("T0_base", R["T0_base"])

    filters = [
        ("T_btc4h",      lambda ev: base(ev) and ev["btc4h"] < 0),
        ("T_btc7d",      lambda ev: base(ev) and ev["btc7d"] < 0),
        ("T_btc4h_or7d", lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("T_btc3d",      lambda ev: base(ev) and ev["btc3d"] < 0),
        ("T_btc5m_dn",   lambda ev: base(ev) and ev["btc5m"] < 0),
        ("T_sync200",    lambda ev: base(ev) and ev["sync_k"] < 200_000),
        ("T_score2",     lambda ev: base(ev) and ev["score"] >= 2),
        ("T_score3",     lambda ev: base(ev) and ev["score"] >= 3),
        ("T_prebuild",   lambda ev: base(ev) and ev["prebuildup"] > 0),
        ("T_vac_clean",  lambda ev: base(ev) and ev["vac_clean"]),
        ("T_not_blocked",lambda ev: base(ev) and not ev["blocked"]),
        ("T_no_sat",     lambda ev: base(ev) and ev["dow"] != 5),
        ("T_veto_ratio", lambda ev: base(ev) and ev["be_ratio"] < 2.0),
        ("T_vol_low",    lambda ev: base(ev) and ev.get("vd_now") is not None and ev["vd_now"] <= 3),
        ("T_ofi_buy",    lambda ev: base(ev) and (ev.get("ofi_post") or -1) > 0),
    ]
    for name, fn in filters:
        R[name] = stats(vals_for(events, fn), name, months)
        pstat(name, R[name])

    # Kümülatif WR yığını (freq'i çok düşürmeden tail kes)
    print("  Kümülatif WR stack:")
    stacks = [
        ("S1_regime",         lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("S2_regime_sync",    lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)
                                          and ev["sync_k"] < 200_000),
        ("S3_regime_sync_veto", lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)
                                          and ev["sync_k"] < 200_000 and ev["be_ratio"] < 2.0),
        ("S4_regime_prebuild",lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)
                                          and ev["prebuildup"] > 0),
    ]
    for name, fn in stacks:
        R[name] = stats(vals_for(events, fn), name, months)
        pstat(name, R[name])

    return R

# ---------------------------------------------------------------------------
# STOP: stop simülasyonu (tail koruma) — echo geniş baz + en iyi WR aday
# ---------------------------------------------------------------------------

def run_STOP(events, months, base_key="e30_90"):
    print("\n=== STOP: path-min stop simülasyonu ===")
    R = {}

    base = lambda ev: echo_base(ev, base_key)
    regime = lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)

    stop_keys = [("none", "l4h"), ("s200", "l4h_s200"), ("s150", "l4h_s150"),
                 ("s100", "l4h_s100"), ("s75", "l4h_s75"), ("s50", "l4h_s50")]

    print("  Echo geniş baz + stop:")
    for sname, okey in stop_keys:
        R["STOP_base_%s" % sname] = stats(vals_for(events, base, okey),
                                          "echo baz stop=%s" % sname, months)
        pstat("STOP_base_%s" % sname, R["STOP_base_%s" % sname])

    print("  Echo geniş + regime + stop:")
    for sname, okey in stop_keys:
        R["STOP_regime_%s" % sname] = stats(vals_for(events, regime, okey),
                                            "echo+regime stop=%s" % sname, months)
        pstat("STOP_regime_%s" % sname, R["STOP_regime_%s" % sname])

    return R

# ---------------------------------------------------------------------------
# MAXFREQ: 17/ay üstü frekans + rejim/stop ile tail koruma birlikte
# ---------------------------------------------------------------------------

def run_MAXFREQ(events, months):
    print("\n=== MAXFREQ: yüksek frekans + rejim/stop ===")
    R = {}

    # Kullanıcının gerçek bazı: echo_30_90 + silence, HİÇ gate yok (17/ay WR68 referans)
    user_base = lambda ev: (not ev["noisy"] and ev["echo"].get("e30_90"))
    R["M0_user_baseline"] = stats(vals_for(events, user_base),
                                  "REF: echo30_90 silence (gate yok)", months)
    pstat("M0_user_baseline", R["M0_user_baseline"])

    # nodow silence universe: en geniş frekans kaynakları
    def nodow_sil(ev, keys):
        return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
                and any(ev["echo"].get(k) for k in keys))
    def regime(ev):
        return ev["btc4h"] < 0 or ev["btc7d"] < 0

    universes = [
        ("e30_150", ["e30_150"]),
        ("e20_150", ["e20_150"]),
        ("e30_120", ["e30_120"]),
        ("union",   ["e20_60", "e30_90", "e45_120", "e60_180"]),
    ]
    for uname, keys in universes:
        raw   = lambda ev, ks=keys: nodow_sil(ev, ks)
        reg   = lambda ev, ks=keys: nodow_sil(ev, ks) and regime(ev)
        # raw (nodow, no regime)
        R["M_%s_raw" % uname] = stats(vals_for(events, raw), "nodow %s raw" % uname, months)
        pstat("M_%s_raw" % uname, R["M_%s_raw" % uname])
        # + regime
        R["M_%s_reg" % uname] = stats(vals_for(events, reg), "nodow %s +regime" % uname, months)
        pstat("M_%s_reg" % uname, R["M_%s_reg" % uname])
        # + regime + stop150 (tail cap)
        R["M_%s_reg_s150" % uname] = stats(vals_for(events, reg, "l4h_s150"),
                                           "nodow %s +regime +stop150" % uname, months)
        pstat("M_%s_reg_s150" % uname, R["M_%s_reg_s150" % uname])
        # + regime + prebuildup (kalite)
        reg_pb = lambda ev, ks=keys: nodow_sil(ev, ks) and regime(ev) and ev["prebuildup"] > 0
        R["M_%s_reg_pb" % uname] = stats(vals_for(events, reg_pb),
                                         "nodow %s +regime +prebuild" % uname, months)
        pstat("M_%s_reg_pb" % uname, R["M_%s_reg_pb" % uname])
        # no-overlap of the +regime version
        no = no_overlap_vals(pairs_for(events, reg))
        s_no = stats(no, "nodow %s +regime no-overlap" % uname, months)
        s_no["per_month"] = round(len(no) / months, 1)
        R["M_%s_reg_noov" % uname] = s_no
        pstat("M_%s_reg_noov" % uname, s_no)

    return R

# ---------------------------------------------------------------------------
# FINAL: aday kombinasyonlar + no-overlap
# ---------------------------------------------------------------------------

def run_FINAL(events, months):
    print("\n=== FINAL: aday kombinasyonlar (no-overlap dahil) ===")
    R = {}

    cands = {
        "A_echo30_90_base":      lambda ev: echo_base(ev, "e30_90"),
        "B_echo30_120_base":     lambda ev: echo_base(ev, "e30_120"),
        "C_echo30_120_regime":   lambda ev: echo_base(ev, "e30_120")
                                             and (ev["btc4h"] < 0 or ev["btc7d"] < 0),
        "D_echo30_120_reg_stop100": None,   # stop overlay aşağıda
        "E_union_regime":        lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE"
                                             and not ev["noisy"] and ev["dow"] not in {0, 2}
                                             and any(ev["echo"].get(k) for k in
                                                     ["e20_60", "e30_90", "e45_120"])
                                             and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
    }

    for name, fn in cands.items():
        if fn is None:
            continue
        raw = pairs_for(events, fn, "l4h")
        no  = no_overlap_vals(raw)
        s_raw = stats([v for _, v in raw], name + " raw", months)
        s_no  = stats(no, name + " no-overlap", months / max(1, len(raw)) * len(no) if raw else months)
        # no-overlap /ay: aynı ay tabanı ile ölç
        s_no["per_month"] = round(len(no) / months, 1)
        R[name] = s_raw
        R[name + "_noov"] = s_no
        pstat(name, s_raw)
        pstat(name + "_noov", s_no)

    # D: echo30_120 + regime + stop100
    fnD = lambda ev: (echo_base(ev, "e30_120")
                      and (ev["btc4h"] < 0 or ev["btc7d"] < 0))
    rawD = pairs_for(events, fnD, "l4h_s100")
    noD  = no_overlap_vals(rawD)
    R["D_echo30_120_reg_stop100"] = stats([v for _, v in rawD],
                                          "echo30_120 regime stop100", months)
    sD_no = stats(noD, "echo30_120 regime stop100 no-overlap", months)
    sD_no["per_month"] = round(len(noD) / months, 1)
    R["D_echo30_120_reg_stop100_noov"] = sD_no
    pstat("D_echo30_120_reg_stop100", R["D_echo30_120_reg_stop100"])
    pstat("D_echo30_120_reg_stop100_noov", sD_no)

    return R

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _row(k, v):
    if not v or v.get("n", 0) == 0:
        return "| %s | 0 | - | - | - | - | - | - | - | - |" % k
    return "| %s | %d | %.1f | %.1f%% | %+.1f | %+.1f | %+.1f | %d | %s | %s |" % (
        k, v["n"], v.get("per_month", 0), v["wr"], v["avg"],
        v.get("t3r", 0), v.get("worst", 0), v.get("tail_n", 0),
        v.get("mc_p", "?"), v.get("wf", "-"))

def make_md(sections, meta):
    L = ["# S34 Echo Expansion — Frekans + Tail/WR",
         "",
         "> Baz: **echo_30_90 silence gated** (\"Echo geniş\" ~17/ay, WR ~68%).",
         "> Evren: %d ETH SELL 200K anchor, %.1f ay, FEE=%dbps, hold=4h." % (
             meta["n_events"], meta["months"], int(FEE_BPS)),
         "> Tarih: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"),
         "",
         "Kolonlar: N, /ay, WR, Avg bps, T3R (top-3 removed sum), Worst, Tail (<-100bps), mc_p, WF (pozitif fold/5).",
         ""]
    titles = {
        "FREQ":   "1) Frekans Artırma — Echo Genişletme",
        "TAILWR": "2) Tail Koru / WR Artır — Katmanlı Filtre",
        "STOP":   "3) Stop Simülasyonu (path-min gerçek stop-out)",
        "MAXFREQ":"4) Yüksek Frekans + Rejim/Stop (tail koruma birlikte)",
        "FINAL":  "5) Aday Kombinasyonlar (no-overlap dahil)",
    }
    hdr = "| Test | N | /ay | WR | Avg | T3R | Worst | Tail | mc_p | WF |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for sec, R in sections.items():
        L += ["## %s" % titles.get(sec, sec), "", hdr, sep]
        for k, v in R.items():
            L.append(_row(k, v))
        L.append("")
    L += ["---", "*Script: tools/research_s34_echo_expansion.py*"]
    return "\n".join(L)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    print("=== S34 Echo Expansion ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")

        now_ms   = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - LOOKBACK_MS

        print("Loading ETH SELL liqs ...")
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        print("  liqs: %d" % len(liqs))

        print("Reconstructing 200K anchors ...")
        anchors = reconstruct_anchors(
            liqs, bucket_sec=300, min_gap_sec=900,
            thresholds=(ETH_THRESH,), accel_window_sec=30)
        print("  anchors: %d" % len(anchors))

        ts_span = sorted(int(a.anchor_ts_ms) for a in anchors)
        span_days = (ts_span[-1] - ts_span[0]) / 86_400_000 if len(ts_span) > 1 else 30
        TOTAL_MONTHS = max(1.0, span_days / 30.0)
        months = TOTAL_MONTHS
        print("  span %.0f gün = %.2f ay" % (span_days, months))

        print("Loading mark prices ...")
        marks_eth = load_mark_index(conn, "ETHUSDT")
        print("Loading vol_state ...")
        vol_rows = load_vol_state(conn)
        print("  vol rows: %d" % len(vol_rows))

        events = build_events(conn, anchors, marks_eth, vol_rows)
        print("  events: %d" % len(events))

        sections = {}
        sections["FREQ"]   = run_FREQ(events, months)
        sections["TAILWR"] = run_TAILWR(events, months, base_key="e30_90")
        sections["STOP"]   = run_STOP(events, months, base_key="e30_90")
        sections["MAXFREQ"] = run_MAXFREQ(events, months)
        sections["FINAL"]  = run_FINAL(events, months)

    meta = {"n_events": len(events), "months": round(months, 2)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sections": sections, "meta": meta}, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(sections, meta))
    print("\nJSON: %s" % OUT_JSON)
    print("MD:   %s" % OUT_MD)

    print("\n=== ÖZET: en iyi frekans (N>=20, WR>=65, mc_p<=0.05) ===")
    allr = {}
    for R in sections.values():
        allr.update(R)
    good = [(k, v) for k, v in allr.items()
            if v and v.get("n", 0) >= 20 and v.get("wr", 0) >= 65
            and v.get("mc_p") is not None and v["mc_p"] <= 0.05]
    good.sort(key=lambda x: -x[1].get("per_month", 0))
    for k, v in good[:15]:
        pstat(k, v)
    print("Done.")


if __name__ == "__main__":
    main()
