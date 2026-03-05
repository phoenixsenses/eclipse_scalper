from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import compute_calibration
from src.microphys.alpha.column_guard import collect_expr_columns
from src.microphys.alpha.generator import generate_candidates


def _frame() -> pd.DataFrame:
    n = 120
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "mid": [100.0 + (0.01 * i) for i in range(n)],
            "F_ofi_z": [(-1.0 + (2.0 * (i / max(1, n - 1)))) for i in range(n)],
            "F_intensity_z": [0.5 + ((i % 10) / 10.0) for i in range(n)],
            "spread_z": [(-1.0 if i % 3 else 0.5) for i in range(n)],
            "compression_flag": [1 if i % 4 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 7 == 0 else 0 for i in range(n)],
            "regime_id": [0] * n,
        }
    )


def test_generator_coverage_guarantee_nonzero_triggers() -> None:
    df = _frame()
    ctx = compute_calibration(df, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    specs = generate_candidates(
        horizons=[5],
        compression_options=[False, True],
        vacuum_options=[False],
        limit=40,
        calibration=ctx,
        frame=df,
        coverage_guarantee=True,
        min_triggered=5,
        max_tries=10,
        available_columns=df.columns.tolist(),
    )
    assert specs
    assert any(int((s.meta or {}).get("calibration_triggered", 0) or 0) > 0 for s in specs)


def test_generated_specs_reference_existing_columns() -> None:
    df = _frame()
    ctx = compute_calibration(df, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    specs = generate_candidates(
        horizons=[5],
        compression_options=[True],
        vacuum_options=[True],
        limit=20,
        calibration=ctx,
        frame=df,
        coverage_guarantee=False,
        available_columns=df.columns.tolist(),
    )
    used = sorted({c for s in specs for c in collect_expr_columns(s.condition)})
    missing = [c for c in used if c not in df.columns]
    assert not missing
