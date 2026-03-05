from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.eval import evaluate_spec_on_frame
from src.microphys.alpha.spec import SignalSpec


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i % 60):02d}Z" for i in range(n)],
            "mid": [100.0 + 0.01 * i for i in range(n)],
            "spread": [0.002] * n,
            "x": [1.0] * n,
            "regime_id": [0] * n,
        }
    )


def test_max_trades_per_day_cap_enforced() -> None:
    spec = SignalSpec(
        name="always",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "x", "right": 0.0},
        horizon_bars=1,
        cooldown_bars=0,
    )
    trades, stats = evaluate_spec_on_frame(
        _frame(),
        spec,
        fee_bps=0.0,
        latency_bars=0,
        mode="taker",
        fill_prob=1.0,
        max_trades_per_day=5,
    )
    assert len(trades) == 5
    assert int(stats["trade_count"]) == 5
    assert int(stats["capped_trades"]) > 0
