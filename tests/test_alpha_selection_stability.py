from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.selection import select_robust_signals, summarize_signals


def test_selection_prefers_true_signal() -> None:
    rows = []
    for split in (1, 2, 3):
        rows.append(
            {
                "signal": "true_edge",
                "split_id": split,
                "test_trade_count": 25,
                "test_net_mean": 0.0015,
                "test_sharpe": 1.2,
                "stability_score": 0.8,
                "overfit_gap": 0.0001,
                "regime_concentration": 0.4,
            }
        )
        rows.append(
            {
                "signal": "noise",
                "split_id": split,
                "test_trade_count": 25,
                "test_net_mean": -0.0002 if split == 3 else 0.0002,
                "test_sharpe": 0.1,
                "stability_score": 0.2,
                "overfit_gap": 0.0015,
                "regime_concentration": 0.9,
            }
        )
    summary = summarize_signals(pd.DataFrame(rows))
    selected = select_robust_signals(summary, min_trades_per_split=5, min_stability=0.3)
    assert "true_edge" in selected["signal"].tolist()
    assert "noise" not in selected["signal"].tolist()
