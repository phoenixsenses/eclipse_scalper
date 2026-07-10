# encoding: utf-8
"""
S34 Research: Session / time-of-day performance analysis

Tests whether cascade performance differs by UTC session:
  Asia:   00:00-08:00 UTC
  Europe: 08:00-14:00 UTC
  US:     14:00-22:00 UTC
  Late:   22:00-24:00 UTC

Uses historical ETH 500K + SOL 200K cascades from microstructure.db.
OOS split at 70%.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

SESSIONS = [
    ("Asia  ", 0,  8),
    ("Europe", 8,  14),
    ("US    ", 14, 22),
    ("Late  ", 22, 24),
]

FEE_BPS = 8.0

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def _price_path(micro, symbol, gt_ms, le_ms):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_price_path_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V8). No longer called by simulate()'s live path.
    Selects ONLY `mark_price` (no `ts_ms`), unlike prior gates' typical
    `(ts_ms, mark_price)` shape."""
    return micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, gt_ms, le_ms)).fetchall()


def _price_path_v2(root, symbol, gt_ms, le_ms, source_db_path=None):
    """Reader-backed replacement for `_price_path`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter
    (ETHUSDT/SOLUSDT, sourced from each call site).

    Range semantics: same exclusive-lower/inclusive-upper mapping as
    `research_s34_hold_sweep.py`'s identically-shaped query --
    `start_ms=gt_ms+1, end_ms=le_ms+1` reproduces the oracle's
    `ts_ms>gt_ms AND ts_ms<=le_ms` bit-for-bit.

    Column projection: requesting only `columns=("mark_price",)` makes
    `execute_read` return single-element `(mark_price,)` tuples,
    matching the oracle's `SELECT mark_price` shape exactly -- so the
    call site's existing `for (mp,) in rows:` unpacking is unchanged."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(gt_ms) + 1, end_ms=int(le_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=source_db_path)
    return list(result.iter_rows())


def simulate(micro, symbol, entry_ms, tp_bps, sl_bps, be_bps, hold_sec, direction, root, source_db_path):
    p0 = mark_at(micro, symbol, entry_ms)
    if not p0:
        return None, "MISS"
    if direction == "LONG":
        p_tp = p0 * (1 + tp_bps / 10000)
        p_sl = p0 * (1 - sl_bps / 10000)
        p_be = p0 * (1 + be_bps / 10000)
    else:
        p_tp = p0 * (1 - tp_bps / 10000)
        p_sl = p0 * (1 + sl_bps / 10000)
        p_be = p0 * (1 - be_bps / 10000)
    be_on = False
    rows = _price_path_v2(root, symbol, entry_ms, entry_ms + hold_sec * 1000, source_db_path=source_db_path)
    for (mp,) in rows:
        if direction == "LONG":
            if mp >= p_tp: return tp_bps - FEE_BPS, "TP"
            if be_on and mp <= p0: return -FEE_BPS, "BE"
            if not be_on and mp <= p_sl: return -sl_bps - FEE_BPS, "SL"
            if mp >= p_be: be_on = True
        else:
            if mp <= p_tp: return tp_bps - FEE_BPS, "TP"
            if be_on and mp >= p0: return -FEE_BPS, "BE"
            if not be_on and mp >= p_sl: return -sl_bps - FEE_BPS, "SL"
            if mp <= p_be: be_on = True
    p_end = mark_at(micro, symbol, entry_ms + hold_sec * 1000) or p0
    if direction == "LONG":
        return (p_end - p0) / p0 * 10000 - FEE_BPS, "TIME"
    else:
        return (p0 - p_end) / p0 * 10000 - FEE_BPS, "TIME"

def st(label, nets):
    if not nets:
        return f"  {label:30} N=  0"
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    wr = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn = sum(nets) / len(nets)
    return (f"  {label:30} N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:>3.0f}%)")

def session_label(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    h = dt.hour
    for name, h0, h1 in SESSIONS:
        if h0 <= h < h1:
            return name
    return "Late  "

def build_cascades(micro, symbol, side, threshold, cnt_min, bucket_sec=30):
    BUCKET_MS = bucket_sec * 1000
    direction = "LONG" if side == "BUY" else "SHORT"
    rows = micro.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side)).fetchall()
    buckets = defaultdict(lambda: [0.0, 0, None])
    for ts, notional in rows:
        bk = (ts // BUCKET_MS) * BUCKET_MS
        buckets[bk][0] += notional
        buckets[bk][1] += 1
        if buckets[bk][2] is None:
            buckets[bk][2] = ts
    return sorted(
        [(data[2], data[0], data[1], direction)
         for bk, data in buckets.items()
         if data[0] >= threshold and data[1] >= cnt_min and data[2] is not None],
        key=lambda x: x[0]
    )

def analyze(micro, label, cascades, tp_bps, sl_bps, be_bps, hold_sec, symbol, root, source_db_path):
    results = []
    for ts, total, cnt, direction in cascades:
        net, exit_r = simulate(micro, symbol, ts, tp_bps, sl_bps, be_bps, hold_sec, direction, root, source_db_path)
        if net is not None and isinstance(net, (int, float)):
            results.append({"ts": ts, "net": float(net), "exit": exit_r, "session": session_label(ts)})

    split_idx = int(len(results) * 0.70)
    split_ts  = results[split_idx]["ts"] if split_idx < len(results) else 0
    split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d") if split_ts else "?"
    test = results[split_idx:]

    print(f"\n{'='*65}")
    print(f"{label}  (OOS split: {split_dt}, test N={len(test)})")
    print(f"{'='*65}")
    print("ALL DATA:")
    for sname, h0, h1 in SESSIONS:
        nets = [r["net"] for r in results if r["session"] == sname]
        print(st(f"{sname} ({h0:02d}-{h1:02d} UTC)", nets))
    print(st("ALL", [r["net"] for r in results]))

    print("\nOOS (TEST) ONLY:")
    for sname, h0, h1 in SESSIONS:
        nets = [r["net"] for r in test if r["session"] == sname]
        print(st(f"{sname} ({h0:02d}-{h1:02d} UTC)", nets))
    print(st("ALL", [r["net"] for r in test]))

    # Hour-by-hour breakdown for best/worst signal
    hour_nets = defaultdict(list)
    for r in results:
        h = datetime.fromtimestamp(r["ts"]/1000, tz=timezone.utc).hour
        hour_nets[h].append(r["net"])
    print("\nHOUR-BY-HOUR (all data, med bps):")
    for h in range(24):
        nets = hour_nets.get(h, [])
        if nets:
            s = sorted(nets)
            med = s[len(s)//2]
            sl  = sum(1 for x in nets if x < -20)
            bar = "#" * len(nets)
            print(f"  {h:02d}h  N={len(nets):>3}  med={med:>+6.1f}  SL={sl}  {bar}")

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)
    root, _ = PR.resolve_production_root()
    source_db_path = str(MICRO_DB)

    print("S34 SESSION / TIME-OF-DAY ANALYSIS")
    print("Hypothesis: cascade momentum differs by market session")

    # ETH 500K BUY→LONG
    cascades = build_cascades(micro, "ETHUSDT", "BUY", 500_000, 8)
    analyze(micro, "ETH $500K BUY->LONG  TP60 SL40 BE30 hold=510s",
            cascades, tp_bps=60, sl_bps=40, be_bps=30, hold_sec=510, symbol="ETHUSDT",
            root=root, source_db_path=source_db_path)

    # SOL 200K BUY→LONG
    cascades = build_cascades(micro, "SOLUSDT", "BUY", 200_000, 8)
    analyze(micro, "SOL $200K BUY->LONG  TP60 SL40 BE30 hold=510s",
            cascades, tp_bps=60, sl_bps=40, be_bps=30, hold_sec=510, symbol="SOLUSDT",
            root=root, source_db_path=source_db_path)

    # ETH 500K SELL→SHORT
    cascades = build_cascades(micro, "ETHUSDT", "SELL", 500_000, 8)
    analyze(micro, "ETH $500K SELL->SHORT  TP60 SL40 BE40 hold=510s",
            cascades, tp_bps=60, sl_bps=40, be_bps=40, hold_sec=510, symbol="ETHUSDT",
            root=root, source_db_path=source_db_path)

    micro.close()
    print("\nNOTE: Shadow research. No runner change without N>=50 OOS per session bucket.")

if __name__ == "__main__":
    main()
