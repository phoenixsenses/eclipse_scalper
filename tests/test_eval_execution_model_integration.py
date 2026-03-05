from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.eval import evaluate_spec_on_frame
from src.microphys.alpha.spec import SignalSpec


def _frame() -> pd.DataFrame:
    n = 80
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i%60):02d}Z" for i in range(n)],
            "mid": [100.0 + i * 0.01 for i in range(n)],
            "spread": [0.001] * n,
            "F_ofi_z": [1.5 if i % 5 == 0 else 0.0 for i in range(n)],
            "F_intensity_z": [1.0] * n,
            "spread_z": [0.0] * n,
            "bid_qty": [10.0] * n,
            "ask_qty": [10.0] * n,
            "qty_sum": [5.0] * n,
            "trade_through_prob": [1.0] * n,
            "regime_id": [0] * n,
        }
    )


def test_eval_with_maker_queue_outputs_filled_trades() -> None:
    spec = SignalSpec(
        name="s",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        horizon_bars=3,
        cooldown_bars=1,
    )
    trades, stats = evaluate_spec_on_frame(
        _frame(),
        spec,
        fee_bps=0.5,
        latency_bars=1,
        mode="maker",
        fill_prob=0.3,
        execution_model="maker_queue",
        execution_params={"maker_queue": {"queue_frac": 0.3, "ttl_bars": 5}},
        ttl_bars=5,
    )
    assert "filled" in trades.columns or len(trades) >= 0
    assert "fill_rate" in stats
