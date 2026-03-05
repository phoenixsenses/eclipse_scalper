from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.diagnostics import cost_decomposition
from src.microphys.alpha.spec import SignalSpec


def test_cost_decomposition_gross_minus_costs_equals_net() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": list(range(30)),
            "ts_utc": [f"2024-03-01T00:00:{i:02d}Z" for i in range(30)],
            "mid": [100.0 + (0.05 * i) for i in range(30)],
            "spread": [0.002] * 30,
            "r_1": [0.0005] * 30,
            "F_ofi_z": [2.0 if i % 5 == 0 else 0.0 for i in range(30)],
            "regime_id": [0] * 30,
        }
    )
    spec = SignalSpec(
        name="s",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        horizon_bars=2,
        cooldown_bars=1,
    )
    out = cost_decomposition(df, [spec], fee_bps=1.0, latency_bars=1, mode="taker", fill_prob=0.3)
    row = out.iloc[0]
    lhs = float(row["gross_mean"]) - float(row["fee_cost_mean"]) - float(row["spread_cost_mean"]) - float(row["adverse_cost_mean"])
    assert abs(lhs - float(row["net_mean"])) < 1e-10
