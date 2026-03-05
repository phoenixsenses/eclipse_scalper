from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_physics_signals import compute_physics_signals_frame


def test_physics_signal_math_returns_velocity_acceleration() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": [1000, 1100, 1200, 1300],
            "ts_utc": ["2024-03-01T00:00:00.000Z", "2024-03-01T00:00:00.100Z", "2024-03-01T00:00:00.200Z", "2024-03-01T00:00:00.300Z"],
            "symbol": ["ETHUSDT"] * 4,
            "mid": [100.0, 101.0, 102.0, 103.0],
            "microprice": [100.0, 101.0, 102.0, 103.0],
            "spread": [0.001, 0.0012, 0.0009, 0.0011],
            "ofi": [1.0, 2.0, 3.0, 4.0],
            "trade_intensity": [10.0, 11.0, 12.0, 13.0],
            "top_depth_imbalance": [0.1, 0.15, 0.05, 0.2],
            "rv_short": [0.0, 0.0, 0.0, 0.0],
            "liq_rate": [0.0, 0.0, 0.0, 0.0],
            "qty_sum": [1.0, 1.0, 1.0, 1.0],
            "trade_count": [1, 1, 1, 1],
        }
    )
    out = compute_physics_signals_frame(df, horizons=[1, 2], rolling=3)
    assert "r_1" in out.columns
    assert "r_2" in out.columns
    assert "velocity" in out.columns
    assert "acceleration" in out.columns

    expected_r1 = pd.Series([0.009950330853168092, 0.00985229644301164, 0.009756174945364656, float("nan")])
    assert abs(float(out.loc[0, "r_1"]) - float(expected_r1.loc[0])) < 1e-12
    assert abs(float(out.loc[1, "velocity"]) - float(expected_r1.loc[1])) < 1e-12
    assert abs(float(out.loc[2, "acceleration"]) - (float(expected_r1.loc[2]) - float(expected_r1.loc[1]))) < 1e-12

    # z-score fields exist and are finite where enough data exists
    assert "F_ofi_z" in out.columns
    assert "F_intensity_z" in out.columns
    assert "spread_z" in out.columns
