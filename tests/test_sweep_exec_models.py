from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import simulate_rule_trades


def _rows(n: int = 80):
    rows = []
    mid = 100.0
    for i in range(n):
        mid = mid * 1.0006
        rows.append(
            {
                "ts_ms": float(i * 1000),
                "mid": float(mid),
                "imbalance": 0.6,
                "trade_intensity": 5000.0,
                "spread": 0.0002,
                "ret_1": 0.0001,
            }
        )
    return rows


def _avg_net(sim: dict) -> float:
    trades = sim.get("trades", [])
    if not trades:
        return 0.0
    vals = [float(t.get("net_return", 0.0)) for t in trades]
    return sum(vals) / len(vals)


def test_exec_model_ordering_when_cost_dominates():
    rows = _rows()
    thresholds = {"imb_q10": -1.0, "imb_q90": 1.0, "int_q90": 0.0, "spr_q90": 1.0}
    common = dict(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds=thresholds,
        labels=None,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=4.0,
        slip_bps=2.0,
        debug_samples=0,
        debug_symbol="BTCUSDT",
    )
    taker = simulate_rule_trades(exec_model="taker", maker_fee_bps=0.5, maker_penalty_bps=0.5, **common)
    mid = simulate_rule_trades(exec_model="mid", maker_fee_bps=0.5, maker_penalty_bps=0.5, **common)
    half = simulate_rule_trades(exec_model="halfspread", maker_fee_bps=0.5, maker_penalty_bps=0.5, **common)

    at = _avg_net(taker)
    am = _avg_net(mid)
    ah = _avg_net(half)
    assert am > ah > at

