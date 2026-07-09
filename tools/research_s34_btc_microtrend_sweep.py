# encoding: utf-8
"""
S34 Research: BTC micro-trend at ETH cascade entry — historical backtest
N=20 live data showed BTC 2-5 bps sweet spot (WR=89%, SL=0).
This tests on ALL historical ETH $500K+ cascades to confirm or deny.

Uses microstructure.db liquidations + mark_prices (Feb-Jun 2026).
Simulates real-fill: taker entry at mark price, TP=60, SL=40, BE=30, hold=510s.
Splits: TRAIN (first 70%) / TEST (last 30%) to check OOS stability.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

from ami.storage import production as PR
from ami.storage import research_reader as RR

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

ETH_THRESHOLD = 500_000
CNT_MIN       = 8
BUCKET_SEC    = 30
TP_BPS        = 60.0
SL_BPS        = 40.0
BE_BPS        = 30.0
HOLD_SEC      = 510
FEE_BPS       = 8.0

def mark_at(micro, symbol, ts_ms):
    """Direct-SQL oracle -- kept as the parity reference for `mark_at_v2`
    (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V3).
    No longer called by simulate()/main(); the reader-backed path is
    used instead."""
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def mark_at_v2(root, symbol, ts_ms, source_db_path=None):
    """Reader-backed replacement for `mark_at`, via
    lookup_latest_at_or_before. Used for both ETHUSDT (which has a real
    archived partition, mark_prices/ETHUSDT/2026-05) and BTCUSDT (which
    has none -- always resolves SQLITE_ONLY for that symbol)."""
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol=symbol, ts_ms=ts_ms,
                                            columns=("mark_price",), source_db_path=source_db_path)
    return result.row[0] if result.found else None

def bps_ret(p1, p2):
    if not p1 or not p2 or p1 == 0:
        return None
    return (p2 - p1) / p1 * 10000

def simulate(micro, root, entry_ms):
    p0 = mark_at_v2(root, 'ETHUSDT', entry_ms)
    if not p0:
        return None, 'MISS'
    p_tp = p0 * (1 + TP_BPS / 10000)
    p_sl = p0 * (1 - SL_BPS / 10000)
    p_be = p0 * (1 + BE_BPS / 10000)
    be_on = False
    # Bounded-RANGE scan (not an ORDER BY ts_ms DESC LIMIT 1 point lookup)
    # -- out of scope for this gate's point-lookup helper; belongs to the
    # range-read helper (execute_read), a separate future migration.
    rows = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (entry_ms, entry_ms + HOLD_SEC * 1000)).fetchall()
    for (mp,) in rows:
        if mp >= p_tp:
            return bps_ret(p0, p_tp) - FEE_BPS, 'TP'
        if be_on and mp <= p0:
            return bps_ret(p0, p0) - FEE_BPS, 'BE'
        if not be_on and mp <= p_sl:
            return bps_ret(p0, p_sl) - FEE_BPS, 'SL'
        if mp >= p_be:
            be_on = True
    p_end = mark_at_v2(root, 'ETHUSDT', entry_ms + HOLD_SEC * 1000) or p0
    return bps_ret(p0, p_end) - FEE_BPS, 'TIME'

def st(nets, label):
    if not nets:
        print(f"  {label:40} N=0")
        return
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    wr = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn = sum(nets) / len(nets)
    print(f"  {label:40} N={len(nets):>4}  WR={wr*100:>4.0f}%  med={med:>+7.1f}  "
          f"mean={mn:>+6.1f}  SL={sl}({sl/len(nets)*100:>3.0f}%)")

def main():
    root, _root_source = PR.resolve_production_root()
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    # Build ETH $500K+ cascade events with cnt
    BUCKET_MS = BUCKET_SEC * 1000
    eth_liqs = micro.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol='ETHUSDT' ORDER BY ts_ms"
    ).fetchall()

    buckets = defaultdict(lambda: [0, 0, None])  # [total, cnt, first_ts]
    for ts, notional in eth_liqs:
        bk = (ts // BUCKET_MS) * BUCKET_MS
        buckets[bk][0] += notional
        buckets[bk][1] += 1
        if buckets[bk][2] is None:
            buckets[bk][2] = ts

    cascades = sorted(
        [(data[2], data[0], data[1]) for bk, data in buckets.items()
         if data[0] >= ETH_THRESHOLD and data[1] >= CNT_MIN and data[2] is not None],
        key=lambda x: x[0]
    )

    print("=" * 70)
    print("BTC MICRO-TREND AT ETH CASCADE — HISTORICAL BACKTEST")
    print(f"ETH threshold: ${ETH_THRESHOLD/1e3:.0f}K  cnt>={CNT_MIN}  TP={TP_BPS}  SL={SL_BPS}  BE={BE_BPS}  Fee={FEE_BPS}")
    print("=" * 70)
    print(f"ETH cascades meeting criteria: {len(cascades)}")

    # For each cascade: get BTC 10s return at entry time
    results = []
    for ts, total, cnt in cascades:
        btc_before = mark_at_v2(root, 'BTCUSDT', ts - 10_000)
        btc_at     = mark_at_v2(root, 'BTCUSDT', ts)
        btc_ret    = bps_ret(btc_before, btc_at)
        net, exit_r = simulate(micro, root, ts)
        if net is None:
            continue
        results.append({'ts': ts, 'total': total, 'cnt': cnt,
                         'btc_ret': btc_ret, 'net': net, 'exit': exit_r})

    micro.close()

    print(f"With price data: N={len(results)}")
    n_miss_btc = sum(1 for r in results if r['btc_ret'] is None)
    print(f"  BTC data missing: {n_miss_btc}")
    print()

    # OOS split at 70%
    split_idx = int(len(results) * 0.70)
    split_ts  = results[split_idx]['ts'] if split_idx < len(results) else 0
    split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d") if split_ts else "?"
    train = results[:split_idx]
    test  = results[split_idx:]

    # Segment by BTC return magnitude
    BTC_BANDS = [
        ("neg  (<0 bps)",   lambda r: r['btc_ret'] is not None and r['btc_ret'] < 0),
        ("0-2  bps",        lambda r: r['btc_ret'] is not None and 0 <= r['btc_ret'] < 2),
        ("2-5  bps",        lambda r: r['btc_ret'] is not None and 2 <= r['btc_ret'] < 5),
        ("5-10 bps",        lambda r: r['btc_ret'] is not None and 5 <= r['btc_ret'] < 10),
        ("10+  bps",        lambda r: r['btc_ret'] is not None and r['btc_ret'] >= 10),
        ("BTC_MISS",        lambda r: r['btc_ret'] is None),
    ]

    for split_label, subset in [("TRAIN (70%)", train), ("TEST  (30%)", test), ("ALL", results)]:
        print(f"--- {split_label} ---")
        for band_label, fn in BTC_BANDS:
            nets = [r['net'] for r in subset if fn(r)]
            st(nets, band_label)
        st([r['net'] for r in subset], "ALL")
        print()

    # Also: does BTC band provide info BEYOND cnt? Show SL breakdown
    print("--- SL detail by BTC band (ALL data) ---")
    for band_label, fn in BTC_BANDS:
        matched = [r for r in results if fn(r) and r['net'] < -20]
        if matched:
            for r in matched:
                dt = datetime.fromtimestamp(r['ts']/1000, tz=timezone.utc).strftime("%m/%d %H:%M")
                btc_s = f"{r['btc_ret']:+.1f}" if r['btc_ret'] is not None else "N/A"
                print(f"  {band_label:15} SL: {dt}  btc={btc_s}  cnt={r['cnt']}  total=${r['total']/1e3:.0f}K")

    print()
    print(f"OOS split date: {split_dt}")
    print("NOTE: Shadow research only. N>=50 per bucket required before live action.")

if __name__ == "__main__":
    main()
