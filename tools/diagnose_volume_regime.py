"""
diagnose_volume_regime.py — Hourly volume regime split on passive pocket signals.

Tests whether gating signals to LOW hourly-volume periods recovers positive NPA
for the micro_edge_v3_passive_alpha h=60 imb>=0.85 pocket.

Self-contained: loads from SQLite directly, no tools/ imports.

Usage:
    python -m tools.diagnose_volume_regime --db ../eclipse_scalper/data/microstructure.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_buckets(db: str, symbol: str) -> list[dict]:
    """Load 1-sec bucket features + mark prices."""
    con = sqlite3.connect(db)
    trade_rows = con.execute(
        """
        SELECT ts_ms/1000 as ts_sec,
               sum(case when is_buyer_maker=0 then quantity else 0 end) as buy_qty,
               sum(case when is_buyer_maker=1 then quantity else 0 end) as sell_qty,
               sum(price*quantity)/sum(quantity) as vwap,
               count(*) as cnt
        FROM agg_trades WHERE symbol=?
        GROUP BY ts_ms/1000 ORDER BY ts_sec
        """,
        (symbol,),
    ).fetchall()

    # Load mark prices
    mark_rows = con.execute(
        "SELECT ts_ms/1000 as ts_sec, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_sec",
        (symbol,),
    ).fetchall()
    con.close()

    mark_map: dict[int, float] = {r[0]: float(r[1]) for r in mark_rows}
    if not mark_map:
        min_sec = min(r[0] for r in trade_rows)
        max_sec = max(r[0] for r in trade_rows)
        mark_map = {s: float(trade_rows[i][3]) for i, s in enumerate(range(min_sec, max_sec + 1))}

    buckets = []
    for r in trade_rows:
        ts_sec, bv, sv, vwap, cnt = r[0], r[1], r[2], r[3], r[4]
        tot = bv + sv
        imb = (bv - sv) / tot if tot > 0 else 0.0
        spread = abs(vwap - mark_map.get(ts_sec, vwap)) / vwap if vwap > 0 else 0.0
        buckets.append({
            "ts_sec": ts_sec,
            "imbalance": imb,
            "trade_intensity": cnt * 60,
            "spread": spread,
            "vwap": vwap,
            "mark": mark_map.get(ts_sec, vwap),
            "cnt": cnt,
        })
    return buckets


def _compute_rolling_counts(buckets: list[dict], window_sec: int) -> dict[int, int]:
    """Map each ts_sec to rolling aggregate trade count over trailing window_sec seconds."""
    counts_by_sec: dict[int, int] = {b["ts_sec"]: b["cnt"] for b in buckets}
    min_sec = buckets[0]["ts_sec"]
    max_sec = buckets[-1]["ts_sec"]

    result: dict[int, int] = {}
    window_sum = 0
    window_deque: list[tuple[int, int]] = []  # (ts_sec, cnt)

    for sec in range(min_sec, max_sec + 1):
        cnt = counts_by_sec.get(sec, 0)
        window_deque.append((sec, cnt))
        window_sum += cnt
        cutoff = sec - window_sec
        while window_deque and window_deque[0][0] <= cutoff:
            window_sum -= window_deque[0][1]
            window_deque.pop(0)
        result[sec] = window_sum
    return result


def _run(args: argparse.Namespace) -> None:
    print(f"Loading {args.symbol} from {args.db}...")
    buckets = _load_buckets(args.db, args.symbol)
    print(f"  {len(buckets):,} 1-sec buckets loaded")

    win = args.rolling_window_sec
    print(f"Computing rolling {win}s trade counts...")
    hourly_counts = _compute_rolling_counts(buckets, win)

    # Build ts_sec -> bucket index map
    ts_to_idx: dict[int, int] = {b["ts_sec"]: i for i, b in enumerate(buckets)}

    # Find signals: abs(imbalance) >= min_imb AND trade_intensity >= min_int
    min_imb = args.min_imbalance
    min_int = args.min_trade_intensity
    max_spr = args.max_spread
    horizon = args.horizon_sec
    vol_threshold = args.hourly_vol_threshold

    stats: dict[str, dict] = {
        "HIGH": {"n": 0, "n_long": 0, "n_short": 0, "gross_pnl": 0.0, "fills": 0},
        "LOW": {"n": 0, "n_long": 0, "n_short": 0, "gross_pnl": 0.0, "fills": 0},
    }
    pnl_by_regime: dict[str, list[float]] = defaultdict(list)

    for i, b in enumerate(buckets):
        imb = b["imbalance"]
        if abs(imb) < min_imb:
            continue
        if b["trade_intensity"] < min_int:
            continue
        if b["spread"] > max_spr:
            continue

        ts_sec = b["ts_sec"]
        hr_count = hourly_counts.get(ts_sec, 0)
        regime = "HIGH" if hr_count > vol_threshold else "LOW"

        # Simple forward return: mark price at t+horizon vs t
        exit_sec = ts_sec + horizon
        entry_mark = b["mark"]
        exit_mark = None
        for s in range(exit_sec - 5, exit_sec + 6):
            if s in ts_to_idx:
                exit_mark = buckets[ts_to_idx[s]]["mark"]
                break

        if exit_mark is None or entry_mark <= 0:
            stats[regime]["n"] += 1
            continue

        # Direction: imb > 0 = BUY (expect price up), imb < 0 = SELL (expect price down)
        if imb > 0:
            gross = (exit_mark - entry_mark) / entry_mark  # BUY: positive = good
            stats[regime]["n_long"] += 1
        else:
            gross = (entry_mark - exit_mark) / entry_mark  # SELL: positive = good
            stats[regime]["n_short"] += 1

        stats[regime]["n"] += 1
        stats[regime]["gross_pnl"] += gross
        stats[regime]["fills"] += 1
        pnl_by_regime[regime].append(gross)

    # Report
    print(f"\n--- Volume Regime Split ---")
    print(f"Signal params: abs(imb)>={min_imb} int>={min_int} spr<={max_spr} h={horizon}s")
    print(f"Rolling window: {args.rolling_window_sec}s  threshold: {vol_threshold:,} trades\n")

    for reg in ["LOW", "HIGH"]:
        s = stats[reg]
        pnls = pnl_by_regime[reg]
        n = s["n"]
        if n == 0:
            print(f"  {reg}: no signals")
            continue
        mean_gross_bps = (s["gross_pnl"] / n) * 10000 if n > 0 else 0.0
        pnls_sorted = sorted(pnls)
        p50 = pnls_sorted[len(pnls_sorted)//2] * 10000 if pnls_sorted else 0.0
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
        print(f"  {reg}: n_signals={n:4d}  long={s['n_long']:4d}  short={s['n_short']:4d}"
              f"  mean_gross={mean_gross_bps:+.2f}bps  median={p50:+.2f}bps"
              f"  win_rate={win_rate:.1%}"
              f"  hr_count_label={'>' if reg=='HIGH' else '<='}{vol_threshold:,}")

    print()
    total_low = stats["LOW"]["n"]
    total_high = stats["HIGH"]["n"]
    print(f"LOW share of signals: {total_low/(total_low+total_high):.1%}"
          if total_low+total_high > 0 else "no signals")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--min-imbalance", type=float, default=0.85)
    p.add_argument("--min-trade-intensity", type=float, default=4000.0)
    p.add_argument("--max-spread", type=float, default=0.000150)
    p.add_argument("--horizon-sec", type=int, default=60)
    p.add_argument("--hourly-vol-threshold", type=int, default=120000,
                   help="Rolling trade count threshold. Signals above = HIGH regime.")
    p.add_argument("--rolling-window-sec", type=int, default=3600,
                   help="Rolling window in seconds for trade count (default 3600=1h, 86400=24h).")
    args = p.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
