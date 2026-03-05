from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.microphys.execution.calibration import calibrate_execution_models, calibrate_queue_position_params


def _physics(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": "ETHUSDT" if i % 2 == 0 else "BTCUSDT",
                "regime_id": 1 if i % 3 == 0 else 0,
                "r_1": 0.0001 if i % 4 == 0 else -0.0001,
                "F_ofi_z": 1.0 if i % 2 == 0 else -1.0,
                "spread_z": 0.2 + 0.01 * (i % 5),
                "F_intensity_z": 0.5 + 0.1 * (i % 7),
                "trade_through_prob": 0.4 + 0.05 * (i % 4),
            }
        )
    return pd.DataFrame(rows)


def test_calibrate_queue_position_params_bounded() -> None:
    p = calibrate_queue_position_params(_physics())
    assert 0.01 <= float(p["initial_queue_frac"]) <= 0.99
    assert 0.0 <= float(p["same_side_join_rate"]) <= 1.0
    assert 0.0 <= float(p["same_side_cancel_rate"]) <= 1.0
    assert 0.0 <= float(p["opposite_flow_scale"]) <= 2.0
    assert 0.0 < float(p["pressure_floor"]) <= 1.0
    assert int(p["ttl_bars"]) >= 1


def test_calibrate_queue_position_params_symbol_regime_slice() -> None:
    p_all = calibrate_queue_position_params(_physics())
    p_slice = calibrate_queue_position_params(_physics(), symbol="ETHUSDT", regime_id=1)
    # both valid and deterministic dictionaries
    assert isinstance(p_all, dict) and isinstance(p_slice, dict)
    assert set(p_all.keys()) == set(p_slice.keys())


def test_calibrate_execution_models_contains_queue_v2() -> None:
    out = calibrate_execution_models(_physics())
    assert "maker_queue" in out
    assert "maker_queue_v2" in out
    assert "initial_queue_frac" in out["maker_queue_v2"]

