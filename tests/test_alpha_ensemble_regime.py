from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.ensemble import build_ensemble_scores
from src.microphys.alpha.spec import SignalSpec


def test_ensemble_respects_regime_filter() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": [1, 2, 3, 4],
            "ts_utc": ["a", "b", "c", "d"],
            "symbol": ["ETHUSDT"] * 4,
            "regime_id": [0, 0, 1, 1],
            "F_ofi_z": [2.0, 0.0, 2.0, 0.0],
        }
    )
    spec = SignalSpec(
        name="r0_only",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        regime_filter=[0],
    )
    out = build_ensemble_scores(df, [spec])
    assert out["ensemble_score"].tolist() == [1.0, 0.0, 0.0, 0.0]
