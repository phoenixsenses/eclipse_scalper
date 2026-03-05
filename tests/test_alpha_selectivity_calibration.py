from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import compute_calibration
from src.microphys.alpha.generator import generate_candidates


def _frame(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i % 60):02d}Z" for i in range(n)],
            "mid": [100.0 + (0.001 * i) for i in range(n)],
            "F_ofi_z": [(-1.0 + 2.0 * (i / max(1, n - 1))) for i in range(n)],
            "F_intensity_z": [0.1 + ((i % 20) / 10.0) for i in range(n)],
            "spread_z": [(-0.5 if i % 2 else 0.5) for i in range(n)],
            "compression_flag": [1 if i % 5 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 9 == 0 else 0 for i in range(n)],
            "regime_id": [0] * n,
        }
    )


def test_selectivity_calibration_sets_trigger_rate_metadata() -> None:
    df = _frame()
    ctx = compute_calibration(df, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    specs = generate_candidates(
        horizons=[5],
        compression_options=[False, True],
        vacuum_options=[False],
        limit=25,
        calibration=ctx,
        frame=df,
        coverage_guarantee=True,
        min_triggered=5,
        max_tries=8,
        target_triggers_per_day=100.0,
        min_triggers_per_day=20.0,
        max_triggers_per_day=300.0,
        available_columns=df.columns.tolist(),
    )
    assert specs
    rates = [float((s.meta or {}).get("trigger_rate_per_day", 0.0) or 0.0) for s in specs]
    assert any(r > 0.0 for r in rates)
    assert all("tighten_steps" in (s.meta or {}) for s in specs)
