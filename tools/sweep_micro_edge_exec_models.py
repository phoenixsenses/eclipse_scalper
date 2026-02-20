from __future__ import annotations

"""
Exec-model x horizon sweep for micro-edge backtest (research-only).
"""

import argparse
import sqlite3
import time
from typing import Any, Dict, List

from execution.passive_execution_simulator import calibrate_passive_model
from tools.micro_edge_backtest import build_passive_calibration_samples, compute_rule_thresholds, simulate_rule_trades
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


def parse_str_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(t)
    return out


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


def _summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    net = [float(t.get("net_return", 0.0)) for t in trades]
    gross = [float(t.get("raw_return", 0.0)) for t in trades]
    cost = [float(t.get("cost", 0.0)) for t in trades]
    n = len(net)
    if n == 0:
        return {
            "n": 0,
            "avg_gross": 0.0,
            "avg_cost": 0.0,
            "avg_net": 0.0,
            "p10_net": 0.0,
            "p90_net": 0.0,
            "p90_net_negative": 0.0,
            "break_even_cost_bps_total": 0.0,
        }
    xs = sorted(net)
    p10 = xs[int((n - 1) * 0.10)]
    p90 = xs[int((n - 1) * 0.90)]
    avg_g = sum(gross) / n
    avg_c = sum(cost) / n
    avg_n = sum(net) / n
    return {
        "n": int(n),
        "avg_gross": avg_g,
        "avg_cost": avg_c,
        "avg_net": avg_n,
        "p10_net": p10,
        "p90_net": p90,
        "p90_net_negative": 1.0 if p90 < 0.0 else 0.0,
        "break_even_cost_bps_total": avg_g * 10000.0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep execution models across horizons.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=720)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizons", default="5,10,15,30,60,120")
    p.add_argument("--exec-models", default="taker,maker,mid,halfspread,passive_realistic")
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--side", default="auto")
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--maker-fee-bps", type=float, default=0.5)
    p.add_argument("--maker-penalty-bps", type=float, default=0.5)
    p.add_argument("--passive-seed", type=int, default=42)
    p.add_argument("--passive-max-wait-buckets", type=int, default=0)
    p.add_argument("--min-feature", action="append", default=["trade_intensity=2500", "imbalance=0.3"])
    p.add_argument("--max-feature", action="append", default=["spread=0.0003"])
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rank-by", choices=["avg_net", "avg_gross"], default="avg_net")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    horizons = parse_int_list(args.horizons)
    exec_models = parse_str_list(args.exec_models)
    min_b, max_b = _parse_bounds(list(args.min_feature or []), list(args.max_feature or []))
    conn = sqlite3.connect(str(args.db), check_same_thread=False)
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(max(1, args.lookback_min) * 60 * 1000)
        scored: List[Dict[str, Any]] = []
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
                        passive_params=(
                            calibrate_passive_model(
                                build_passive_calibration_samples(
                                    rows=rows,
                                    rule_name=str(args.rule),
                                    side=str(args.side),
                                    thresholds=thresholds,
                                    hold_buckets=hold,
                                    min_feature_bounds=min_b,
                                    max_feature_bounds=max_b,
                                    max_wait_buckets=int(args.passive_max_wait_buckets),
                                ),
                                maker_fee_bps=float(args.maker_fee_bps),
                                seed=int(args.passive_seed),
                            )
                            if str(em).lower() == "passive_realistic"
                            else None
                        ),
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
                        passive_max_wait_buckets=int(args.passive_max_wait_buckets),
                    )
                    sm = _summarize_trades(list(sim.get("trades", [])))
                    if int(sm["n"]) < int(args.min_n):
                        continue
                    scored.append(
                        {
                            "symbol": sym,
                            "horizon": int(h),
                            "exec_model": str(em),
                            **sm,
                        }
                    )
        scored.sort(key=lambda r: float(r.get(str(args.rank_by), 0.0) or 0.0), reverse=True)
        top = scored[: max(1, int(args.top_k))]
        print(
            f"sweep_micro_edge_exec_models db={args.db} symbols={symbols} lookback_min={args.lookback_min} "
            f"horizons={horizons} exec_models={exec_models} ranked={len(scored)}"
        )
        print(
            f"{'symbol':8} {'h':>4} {'exec_model':10} {'n':>6} {'avg_gross':>12} {'avg_cost':>12} "
            f"{'avg_net':>12} {'p10_net':>12} {'p90_net':>12} {'p90<0':>7} {'be_bps':>10}"
        )
        for r in top:
            print(
                f"{str(r['symbol']):8} {int(r['horizon']):4d} {str(r['exec_model']):10} {int(r['n']):6d} "
                f"{float(r['avg_gross']):+12.6f} {float(r['avg_cost']):+12.6f} {float(r['avg_net']):+12.6f} "
                f"{float(r['p10_net']):+12.6f} {float(r['p90_net']):+12.6f} "
                f"{('YES' if float(r['p90_net_negative']) > 0 else 'NO'):>7} {float(r['break_even_cost_bps_total']):10.2f}"
            )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
