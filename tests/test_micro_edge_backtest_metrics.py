from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import compute_backtest_metrics


def test_backtest_metrics_basic():
    rets = [0.10, -0.05, 0.02]
    holds = [2, 3, 1]
    m = compute_backtest_metrics(rets, holds, total_buckets=12)
    assert m["n_trades"] == 3
    assert abs(m["win_rate"] - (2 / 3)) < 1e-9
    assert abs(m["avg_return"] - (sum(rets) / 3)) < 1e-9
    assert abs(m["median_return"] - 0.02) < 1e-9
    assert abs(m["pnl_sum"] - 0.07) < 1e-9
    assert abs(m["avg_hold_buckets"] - 2.0) < 1e-9
    assert abs(m["exposure_pct"] - (6 / 12)) < 1e-9
    assert m["max_drawdown"] > 0
    assert math.isfinite(m["profit_factor"])
