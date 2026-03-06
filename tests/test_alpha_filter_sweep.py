from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.filter_sweep import default_settings, run_filter_sweep


def test_filter_sweep_is_deterministic() -> None:
    rows = []
    for split in (1, 2, 3):
        rows.append(
            {
                "signal": "a",
                "split_id": split,
                "test_trade_count": 50,
                "test_net_mean": 0.001,
                "test_sharpe": 1.1,
                "stability_score": 0.8,
                "overfit_gap": 0.0001,
                "regime_concentration": 0.4,
                "positive_test_folds": 3,
            }
        )
        rows.append(
            {
                "signal": "b",
                "split_id": split,
                "test_trade_count": 8,
                "test_net_mean": -0.0001,
                "test_sharpe": 0.0,
                "stability_score": 0.2,
                "overfit_gap": 0.002,
                "regime_concentration": 0.9,
                "positive_test_folds": 1,
            }
        )
    df = pd.DataFrame(rows)
    a = run_filter_sweep(df, default_settings())
    b = run_filter_sweep(df, default_settings())
    assert a.to_csv(index=False) == b.to_csv(index=False)
