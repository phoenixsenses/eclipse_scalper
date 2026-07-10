# encoding: utf-8
"""
S34 Research: Hold duration sweep

Tests whether current 510s hold is optimal across ETH/SOL routes.
Sweeps: 120, 180, 300, 420, 510, 600, 900, 1200 seconds.
OOS split at 70%. Checks both TP hit rate and TIME-exit quality.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT     = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

HOLD_DURATIONS = [120, 180, 300, 420, 510, 600, 900, 1200]
FEE_BPS = 8.0

ROUTES = [
    dict(label="ETH BUY  $500K", symbol="ETHUSDT", side="BUY",  threshold=500_000, cnt_min=8,
         tp=60.0, sl=40.0, be=30.0, direction="LONG"),
    dict(label="SOL BUY  $200K", symbol="SOLUSDT", side="BUY",  threshold=200_000, cnt_min=8,
         tp=60.0, sl=40.0, be=30.0, direction="LONG"),
    dict(label="ETH SELL $500K", symbol="ETHUSDT", side="SELL", threshold=500_000, cnt_min=8,
         tp=60.0, sl=40.0, be=40.0, direction="SHORT"),
    dict(label="SOL SELL $200K", symbol="SOLUSDT", side="SELL", threshold=200_000, cnt_min=8,
         tp=60.0, sl=30.0, be=30.0, direction="SHORT"),
]

def mark_at(micro, symbol, ts_ms):
    r = micro.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return r[0] if r else None

def _price_path(micro, symbol, gt_ms, le_ms):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_price_path_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V8). No longer called by simulate()'s live path."""
    return micro.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, gt_ms, le_ms)).fetchall()


def _price_path_v2(root, symbol, gt_ms, le_ms, source_db_path=None):
    """Reader-backed replacement for `_price_path`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter
    (ETHUSDT/SOLUSDT, sourced from each `ROUTES` entry).

    Range semantics: the oracle uses an EXCLUSIVE lower bound
    (`ts_ms>gt_ms`) and an INCLUSIVE upper bound (`ts_ms<=le_ms`) --
    unlike the inclusive/inclusive shape seen in prior range-read
    gates. `plan_read`/`execute_read` use the reader's half-open
    `[start_ms, end_ms)` convention. Since `ts_ms` is always an integer
    column: `ts_ms>gt_ms` is exactly equivalent to `ts_ms>=gt_ms+1`,
    and `ts_ms<=le_ms` is exactly equivalent to `ts_ms<le_ms+1` -- so
    `start_ms=gt_ms+1, end_ms=le_ms+1` reproduces BOTH boundaries
    bit-for-bit, proven by a dedicated parity test."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(gt_ms) + 1, end_ms=int(le_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(px)) for ts, px in result.iter_rows()]


def simulate(micro, symbol, entry_ms, tp_bps, sl_bps, be_bps, hold_sec, direction, root, source_db_path):
    p0 = mark_at(micro, symbol, entry_ms)
    if not p0:
        return None, "MISS", None
    if direction == "LONG":
        p_tp = p0 * (1 + tp_bps / 10000)
        p_sl = p0 * (1 - sl_bps / 10000)
        p_be = p0 * (1 + be_bps / 10000)
    else:
        p_tp = p0 * (1 - tp_bps / 10000)
        p_sl = p0 * (1 + sl_bps / 10000)
        p_be = p0 * (1 - be_bps / 10000)
    be_on = False
    tp_hit_sec = None
    rows = _price_path_v2(root, symbol, entry_ms, entry_ms + hold_sec * 1000, source_db_path=source_db_path)
    for ts, mp in rows:
        elapsed = (ts - entry_ms) / 1000
        if direction == "LONG":
            if mp >= p_tp:
                tp_hit_sec = elapsed
                return float(tp_bps - FEE_BPS), "TP", tp_hit_sec
            if be_on and mp <= p0:
                return float(-FEE_BPS), "BE", None
            if not be_on and mp <= p_sl:
                return float(-sl_bps - FEE_BPS), "SL", None
            if mp >= p_be:
                be_on = True
        else:
            if mp <= p_tp:
                tp_hit_sec = elapsed
                return float(tp_bps - FEE_BPS), "TP", tp_hit_sec
            if be_on and mp >= p0:
                return float(-FEE_BPS), "BE", None
            if not be_on and mp >= p_sl:
                return float(-sl_bps - FEE_BPS), "SL", None
            if mp <= p_be:
                be_on = True
    p_end = mark_at(micro, symbol, entry_ms + hold_sec * 1000) or p0
    if direction == "LONG":
        net = float((p_end - p0) / p0 * 10000 - FEE_BPS)
    else:
        net = float((p0 - p_end) / p0 * 10000 - FEE_BPS)
    return net, "TIME", None

def build_cascades(micro, symbol, side, threshold, cnt_min):
    BUCKET_MS = 30_000
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
        [(data[2], data[0], data[1])
         for bk, data in buckets.items()
         if data[0] >= threshold and data[1] >= cnt_min and data[2] is not None],
        key=lambda x: x[0]
    )

def row(label, nets, exits):
    if not nets:
        return f"  {label:>6}s  N=  0"
    s = sorted(nets)
    sl  = sum(1 for x in nets if x < -20)
    wr  = sum(1 for x in nets if x > 0) / len(nets)
    med = s[len(s) // 2]
    mn  = sum(nets) / len(nets)
    cum = sum(nets)
    tp_n = exits.count("TP")
    return (f"  {label:>6}s  N={len(nets):>4}  WR={wr*100:>4.0f}%  "
            f"med={med:>+7.1f}  mean={mn:>+6.1f}  cum={cum:>+7.0f}  "
            f"TP={tp_n:>3}({tp_n/len(nets)*100:>3.0f}%)  SL={sl}({sl/len(nets)*100:>2.0f}%)")

def main():
    micro = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)
    root, _ = PR.resolve_production_root()
    source_db_path = str(MICRO_DB)

    print("S34 HOLD DURATION SWEEP")
    print("Current default: 510s. Testing 120-1200s range.")
    print("OOS split at 70% of historical data.")

    for route in ROUTES:
        cascades = build_cascades(micro, route["symbol"], route["side"],
                                  route["threshold"], route["cnt_min"])
        if not cascades:
            continue

        split_idx = int(len(cascades) * 0.70)
        split_ts  = cascades[split_idx][0] if split_idx < len(cascades) else 0
        split_dt  = datetime.fromtimestamp(split_ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        test_cascades = cascades[split_idx:]

        print(f"\n{'='*75}")
        print(f"{route['label']}  |  OOS split: {split_dt}  |  test cascades: {len(test_cascades)}")
        print(f"{'='*75}")
        print(f"  {'Hold':>6}   {'N':>4}  {'WR':>5}  {'med':>8}  {'mean':>7}  {'cum':>8}  {'TP%':>8}  {'SL%':>5}")

        # TP timing distribution at current 510s (to understand when TPs fire)
        tp_times = []

        for hold_sec in HOLD_DURATIONS:
            all_nets, all_exits = [], []
            for ts, total, cnt in test_cascades:
                net, exit_r, tp_sec = simulate(micro, route["symbol"], ts,
                                               route["tp"], route["sl"], route["be"],
                                               hold_sec, route["direction"], root, source_db_path)
                if net is not None:
                    all_nets.append(net)
                    all_exits.append(exit_r)
                    if hold_sec == 510 and tp_sec is not None:
                        tp_times.append(tp_sec)
            marker = " <-- current" if hold_sec == 510 else ""
            print(row(str(hold_sec), all_nets, all_exits) + marker)

        if tp_times:
            s = sorted(tp_times)
            print(f"\n  TP timing (at 510s hold): "
                  f"p25={s[len(s)//4]:.0f}s  med={s[len(s)//2]:.0f}s  "
                  f"p75={s[int(len(s)*0.75)]:.0f}s  p90={s[int(len(s)*0.90)]:.0f}s  "
                  f"max={s[-1]:.0f}s")
            under_120 = sum(1 for x in tp_times if x <= 120)
            under_300 = sum(1 for x in tp_times if x <= 300)
            print(f"  TP hit within 120s: {under_120}/{len(tp_times)} ({under_120/len(tp_times)*100:.0f}%)")
            print(f"  TP hit within 300s: {under_300}/{len(tp_times)} ({under_300/len(tp_times)*100:.0f}%)")

    micro.close()
    print("\nNOTE: Shadow research. No runner change without N>=50 OOS per hold bucket.")

if __name__ == "__main__":
    main()
