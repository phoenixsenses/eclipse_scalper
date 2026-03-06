from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.eval import evaluate_walkforward
from src.microphys.alpha.spec import SignalSpec


def _frame() -> pd.DataFrame:
    ts = list(range(1000, 1060))
    return pd.DataFrame(
        {
            "ts_ms": ts,
            "ts_utc": [f"2024-03-01T00:00:{i:02d}Z" for i in range(len(ts))],
            "mid": [100.0 + (0.01 * i) for i in range(len(ts))],
            "spread": [0.002] * len(ts),
            "F_ofi_z": [1.5 if i % 7 == 0 else 0.2 for i in range(len(ts))],
            "compression_flag": [1 if i % 5 == 0 else 0 for i in range(len(ts))],
            "regime_id": [0 if i < 30 else 1 for i in range(len(ts))],
        }
    )


def test_walkforward_eval_is_deterministic() -> None:
    spec = SignalSpec(
        name="t",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        horizon_bars=3,
        cooldown_bars=2,
    )
    a_eval, a_trades = evaluate_walkforward(_frame(), [spec], splits=3, fee_bps=0.5, latency_bars=1, mode="taker", fill_prob=0.3)
    b_eval, b_trades = evaluate_walkforward(_frame(), [spec], splits=3, fee_bps=0.5, latency_bars=1, mode="taker", fill_prob=0.3)
    assert a_eval.to_csv(index=False) == b_eval.to_csv(index=False)
    assert a_trades.to_csv(index=False) == b_trades.to_csv(index=False)
