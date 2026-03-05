from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.execution.cost_models import CostConfig, evaluate_trade_net


def test_execution_cost_math_taker_and_maker() -> None:
    ret = pd.Series([0.01, -0.01])
    spread = pd.Series([0.002, 0.002])
    side = pd.Series([1.0, -1.0])
    r1 = pd.Series([0.001, -0.001])

    taker = evaluate_trade_net(ret, spread, side, r1, CostConfig(fee_bps=1.0, latency_bars=0, mode="taker", fill_prob=0.3))
    maker = evaluate_trade_net(ret, spread, side, r1, CostConfig(fee_bps=1.0, latency_bars=0, mode="maker", fill_prob=0.3))

    # gross for both is +0.01 ; taker should be lower due to spread crossing cost
    assert taker.iloc[0] < maker.iloc[0]
    assert taker.iloc[1] < maker.iloc[1]
