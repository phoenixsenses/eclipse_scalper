from __future__ import annotations

"""
Horizon sensitivity checker for micro-edge backtest (DB-based).

Runs the same rule+gates over multiple horizons and reports gross/net movement.
"""

import argparse
import sqlite3
import time
from typing import Any, Dict, List

from tools.micro_edge_backtest import compute_rule_thresholds, simulate_rule_trades
from tools.micro_edge_lib import build_bucket_features
from tools.micro_edge_smoke import _load_symbol_trades_and_marks, _parse_symbols


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Micro-edge horizon sensitivity checker.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=720)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizons", default="5,10,15,30,60,120")
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--side", default="auto")
    p.add_argument("--exec-models", default="taker")
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--maker-fee-bps", type=float, default=0.5)
    p.add_argument("--maker-penalty-bps", type=float, default=0.5)
    p.add_argument("--max-feature", action="append", default=["spread=0.0003"])
    p.add_argument("--min-feature", action="append", default=["trade_intensity=2500", "imbalance=0.3"])
    return p.parse_args()


def _parse_bounds(min_raw: List[str], max_raw: List[str]) -> tuple[Dict[str, float], Dict[str, float]]:
    min_b: Dict[str, float] = {}
    max_b: Dict[str, float] = {}
    for raw in min_raw:
        if "=" not in str(raw):
            continue
        k, v = str(raw).split("=", 1)
        min_b[k.strip()] = float(v.strip())
    for raw in max_raw:
        if "=" not in str(raw):
            continue
        k, v = str(raw).split("=", 1)
        max_b[k.strip()] = float(v.strip())
    return min_b, max_b


def main() -> int:
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    horizons = parse_int_list(args.horizons)
    exec_models = [x.strip() for x in str(args.exec_models or "taker").split(",") if x.strip()]
    min_b, max_b = _parse_bounds(list(args.min_feature or []), list(args.max_feature or []))
    conn = sqlite3.connect(str(args.db), check_same_thread=False)
    try:
        print(
            f"check_micro_edge_horizons db={args.db} symbols={symbols} lookback_min={args.lookback_min} "
            f"bucket_sec={args.bucket_sec} horizons={horizons} exec_models={exec_models} rule={args.rule}"
        )
        print(f"gates min={min_b} max={max_b}")
        print(
            f"{'symbol':8} {'h_sec':>6} {'exec_model':10} {'n':>6} {'avg_gross':>12} {'avg_cost':>12} "
            f"{'avg_net':>12} {'p10_net':>12} {'p90_net':>12} {'p90<0':>7} {'be_bps':>10}"
        )
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(max(1, args.lookback_min) * 60 * 1000)
        for sym in symbols:
            trades, marks = _load_symbol_trades_and_marks(conn, sym, start_ms=start_ms, end_ms=now_ms)
            rows = build_bucket_features(
                trades, marks, bucket_sec=int(args.bucket_sec), vol_window=max(4, int(60 / max(1, args.bucket_sec)))
            )
            thresholds = compute_rule_thresholds(rows)
            for h in horizons:
                hold = max(1, int(round(float(h) / max(1, int(args.bucket_sec)))))
                for em in exec_models:
                    sim = simulate_rule_trades(
                        rows=rows,
                        rule_name=str(args.rule),
                        side=str(args.side),
                        thresholds=thresholds,
                        labels=None,
                        hold_buckets=hold,
                        cooldown_buckets=0,
                        fee_bps=float(args.fee_bps),
                        slip_bps=float(args.slip_bps),
                        debug_samples=0,
                        debug_symbol=sym,
                        min_feature_bounds=min_b,
                        max_feature_bounds=max_b,
                        exec_model=str(em),
                        maker_fee_bps=float(args.maker_fee_bps),
                        maker_penalty_bps=float(args.maker_penalty_bps),
                    )
                    trades_out = sim.get("trades", [])
                    net = [float(t.get("net_return", 0.0)) for t in trades_out]
                    gross = [float(t.get("raw_return", 0.0)) for t in trades_out]
                    cost = [float(t.get("cost", 0.0)) for t in trades_out]
                    n = len(net)
                    if n == 0:
                        avg_g = avg_c = avg_n = p10 = p90 = be = 0.0
                        neg = False
                    else:
                        xs = sorted(net)
                        p10 = xs[int((n - 1) * 0.10)]
                        p90 = xs[int((n - 1) * 0.90)]
                        avg_g = sum(gross) / n
                        avg_c = sum(cost) / n
                        avg_n = sum(net) / n
                        be = avg_g * 10000.0
                        neg = p90 < 0.0
                    print(
                        f"{sym:8} {int(h):6d} {str(em):10} {n:6d} {avg_g:+.6f} {avg_c:+.6f} "
                        f"{avg_n:+.6f} {p10:+.6f} {p90:+.6f} {('YES' if neg else 'NO'):>7} {be:10.2f}"
                    )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
