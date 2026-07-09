# encoding: utf-8
"""
S34 Research: BTC 5s/10s micro-trend at ETH 500K signal time
Does BTC direction at entry predict ETH trade quality?

Hypothesis:
  When a $500K+ ETH liquidation cascade fires AND BTC was already rising
  in the 5-10 seconds prior, the bounce is stronger (broader market bid).
  When BTC was falling at entry, ETH bounce is weaker or reverses.

Method:
  - Pull ETH 500K BUY signals from feature_factory.db (cnt>=8 filtered)
  - For each signal ts, look up BTC mark price 10s before and at signal
  - Segment: BTC_UP (>+2 bps in 10s), BTC_FLAT (-2 to +2), BTC_DOWN (<-2)
  - Compare ETH net_bps distribution across segments
  - OOS split at median date
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from ami.storage import production as PR
from ami.storage import research_reader as RR

ROOT = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
FEAT_DB  = ROOT / "data" / "s34_feature_factory.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

ETH_RULE = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
BTC_WINDOW_SEC = 10   # look 10s back for BTC trend
BTC_UP_THR    = 2.0   # bps: BTC must be this much UP
BTC_DOWN_THR  = -2.0  # bps: BTC must be this much DOWN

def bps(p1, p2):
    if not p1 or not p2 or p1 == 0:
        return None
    return (p2 - p1) / p1 * 10000

def btc_mark_price_before_or_at(conn, ts_ms):
    """Direct-SQL oracle -- kept as the parity reference for
    `btc_mark_price_before_or_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-
    ASOF-LOOKUP-CONSUMER-MIGRATION-V2). No longer called by main(); the
    reader-backed path is used instead. Extracted verbatim from the two
    identical inline queries main() used to run directly."""
    row = conn.execute("""
        SELECT mark_price FROM mark_prices
        WHERE symbol='BTCUSDT' AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1
    """, (ts_ms,)).fetchone()
    return row[0] if row else None

def btc_mark_price_before_or_at_v2(root, ts_ms, source_db_path=None):
    """Reader-backed replacement for `btc_mark_price_before_or_at`, via
    lookup_latest_at_or_before. mark_prices has no BTCUSDT archive
    partition (only ETHUSDT/2026-05 is archived), so in real production
    use this always resolves SQLITE_ONLY -- documented, not silently
    assumed archive-capable; the synthetic test suite exercises the
    archive-only/hybrid paths this real data can't reach."""
    result = RR.lookup_latest_at_or_before(root, table="mark_prices", symbol="BTCUSDT", ts_ms=ts_ms,
                                            columns=("mark_price",), source_db_path=source_db_path)
    return result.row[0] if result.found else None

def stats(nets):
    if not nets:
        return None
    s = sorted(nets)
    sl = sum(1 for x in nets if x < -20)
    be = sum(1 for x in nets if -20 <= x <= 5)
    tp = sum(1 for x in nets if x > 5)
    return {
        "n": len(nets),
        "wr": sum(1 for x in nets if x > 0) / len(nets),
        "med": s[len(s) // 2],
        "mean": sum(nets) / len(nets),
        "cum": sum(nets),
        "sl": sl, "sl_pct": sl / len(nets) * 100,
        "tp": tp, "be": be,
    }

def print_stats(label, s, indent=2):
    if not s:
        print(" " * indent + f"{label}: N=0")
        return
    pad = " " * indent
    print(f"{pad}{label:35} N={s['n']:>4}  WR={s['wr']*100:.0f}%  med={s['med']:>+7.1f}  "
          f"mean={s['mean']:>+6.1f}  SL={s['sl']}({s['sl_pct']:.0f}%)  cum={s['cum']:>+8.1f}")


def main():
    root, _root_source = PR.resolve_production_root()
    intel = sqlite3.connect(f"file:{INTEL_DB}?mode=ro", uri=True)
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)

    # Pull closed ETH 500K trades
    trades = intel.execute("""
        SELECT trade_id, entry_ts_ms, net_bps, exit_reason
        FROM s34_trades
        WHERE rule_name=? AND status='CLOSED' AND net_bps IS NOT NULL
        ORDER BY entry_ts_ms
    """, (ETH_RULE,)).fetchall()
    intel.close()

    if not trades:
        print("No trades found.")
        return

    # OOS split at median trade
    mid_idx = len(trades) // 2
    mid_ts  = trades[mid_idx][1]
    mid_dt  = datetime.fromtimestamp(mid_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    print("=" * 70)
    print("BTC MICRO-TREND @ ETH 500K ENTRY — QUALITY SEGMENTATION")
    print(f"Rule: {ETH_RULE}")
    print(f"BTC window: {BTC_WINDOW_SEC}s prior to entry | UP>{BTC_UP_THR} / DOWN<{BTC_DOWN_THR} bps")
    print(f"OOS split: {mid_dt} (trade {mid_idx+1}/{len(trades)})")
    print("=" * 70)
    print()

    segments = {"BTC_UP": [], "BTC_FLAT": [], "BTC_DOWN": [], "BTC_MISS": []}
    oos_segments = {"BTC_UP": [], "BTC_FLAT": [], "BTC_DOWN": [], "BTC_MISS": []}

    for i, (tid, entry_ms, net, exit_r) in enumerate(trades):
        is_oos = (entry_ms >= mid_ts)
        window_start_ms = entry_ms - BTC_WINDOW_SEC * 1000

        # Get BTC mark price: find closest row before window_start and at entry
        mark_before = btc_mark_price_before_or_at_v2(root, window_start_ms, source_db_path=str(MICRO_DB))
        mark_entry = btc_mark_price_before_or_at_v2(root, entry_ms, source_db_path=str(MICRO_DB))

        btc_ret = None
        if mark_before is not None and mark_entry is not None:
            btc_ret = bps(mark_before, mark_entry)

        net_f = float(net)
        if btc_ret is None:
            seg = "BTC_MISS"
        elif btc_ret >= BTC_UP_THR:
            seg = "BTC_UP"
        elif btc_ret <= BTC_DOWN_THR:
            seg = "BTC_DOWN"
        else:
            seg = "BTC_FLAT"

        segments[seg].append(net_f)
        if is_oos:
            oos_segments[seg].append(net_f)

    micro.close()

    # Full dataset
    print("ALL DATA:")
    for seg, nets in segments.items():
        print_stats(seg, stats(nets))
    print_stats("ALL", stats([n for nets in segments.values() for n in nets]))

    print()
    print(f"OOS ONLY (>={mid_dt}):")
    for seg, nets in oos_segments.items():
        print_stats(seg, stats(nets))
    print_stats("OOS ALL", stats([n for nets in oos_segments.values() for n in nets]))

    # Check if mark_prices table exists at all
    print()
    print("--- BTC data coverage check ---")
    for seg, nets in segments.items():
        if nets:
            print(f"  {seg}: N={len(nets)}")


if __name__ == "__main__":
    main()
