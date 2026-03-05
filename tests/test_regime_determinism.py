from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.regime.features import build_regime_features
from src.microphys.regime.models import RegimeFitConfig, fit_regimes


def test_regime_labels_deterministic() -> None:
    n = 300
    df = pd.DataFrame(
        {
            "ts_ms": [1000 + i for i in range(n)],
            "ts_utc": [f"2024-03-01T00:00:{i%60:02d}.000Z" for i in range(n)],
            "symbol": ["ETHUSDT"] * n,
            "rv_short": [0.01 + 0.001 * (i % 5) for i in range(n)],
            "spread_z": [(-1) ** (i % 2) * 0.5 for i in range(n)],
            "F_intensity_z": [0.1 * (i % 10) for i in range(n)],
            "liq_rate_z": [0.0] * n,
            "volume_proxy": [1 + (i % 7) for i in range(n)],
            "r_1": [0.0001 * ((i % 6) - 3) for i in range(n)],
            "F_ofi": [(-1) ** i * (1 + (i % 3)) for i in range(n)],
        }
    )
    feats = build_regime_features(df, rolling=60)
    a = fit_regimes(feats, RegimeFitConfig(method="kmeans", n_regimes=3, seed=7))
    b = fit_regimes(feats, RegimeFitConfig(method="kmeans", n_regimes=3, seed=7))
    assert a["regime_id"].tolist() == b["regime_id"].tolist()
