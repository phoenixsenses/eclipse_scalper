from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.regime.alignment import (
    AlignmentConfig,
    assign_aligned_regimes,
    build_shared_alignment_frame,
)


def _mk_frame(symbol: str, n: int, shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "ts_ms": 1_700_000_000_000 + i * 100,
                "ts_utc": f"2026-03-01T00:00:{i%60:02d}Z",
                "symbol": symbol,
                "rv_short": 0.01 + shift + (i % 5) * 0.001,
                "spread": 0.001 + (i % 4) * 0.0001,
                "F_intensity_z": (i % 10) * 0.2,
                "liq_rate_z": (i % 3) * 0.1,
                "r_1": ((i % 7) - 3) * 0.0001,
                "F_ofi": (-1.0) ** i * (1 + (i % 3)),
                "volume_proxy": 10 + (i % 6),
            }
        )
    return pd.DataFrame(rows)


def test_alignment_assignment_is_deterministic() -> None:
    frames = {
        "ETHUSDT": _mk_frame("ETHUSDT", 300, shift=0.0),
        "BTCUSDT": _mk_frame("BTCUSDT", 300, shift=0.002),
    }
    shared, warnings = build_shared_alignment_frame(frames)
    assert not shared.empty
    cfg = AlignmentConfig(method="kmeans_global", k=4, seed=7)
    a = assign_aligned_regimes(shared, cfg)
    b = assign_aligned_regimes(shared, cfg)
    assert a["aligned_regime_id"].tolist() == b["aligned_regime_id"].tolist()


def test_alignment_handles_missing_features_gracefully() -> None:
    frames = {
        "ETHUSDT": pd.DataFrame({"ts_ms": [1, 2, 3], "ts_utc": ["2026-03-01T00:00:00Z"] * 3, "symbol": ["ETHUSDT"] * 3}),
        "BTCUSDT": pd.DataFrame({"ts_ms": [1, 2, 3], "ts_utc": ["2026-03-01T00:00:00Z"] * 3, "symbol": ["BTCUSDT"] * 3}),
    }
    shared, warnings = build_shared_alignment_frame(frames)
    assert not shared.empty
    assert warnings
    out = assign_aligned_regimes(shared, AlignmentConfig(method="quantile_buckets", k=3, seed=1))
    assert "aligned_regime_id" in out.columns

