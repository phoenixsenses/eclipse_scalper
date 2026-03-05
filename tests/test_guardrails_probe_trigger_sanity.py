from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import compute_calibration
from src.microphys.live.guardrails import evaluate_probe_trigger_sanity


def _base_frame(n: int = 5000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": [1_700_000_000_000 + i * 100 for i in range(n)],
            "ts_utc": [f"2024-03-01T00:{(i//600)%60:02d}:{i%60:02d}Z" for i in range(n)],
            "F_ofi_z": [(-3.0 + 6.0 * (i / max(1, n - 1))) for i in range(n)],
            "F_intensity_z": [(-2.0 + 4.0 * ((i % 200) / 199.0)) for i in range(n)],
            "spread_z": [(-2.0 + 4.0 * ((i % 100) / 99.0)) for i in range(n)],
            "compression_flag": [1 if i % 30 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 80 == 0 else 0 for i in range(n)],
            "liq_burst_flag": [1 if i % 120 == 0 else 0 for i in range(n)],
        }
    )


def test_probe_sanity_passes_on_moderate_distribution() -> None:
    frame = _base_frame(6000)
    ctx = compute_calibration(frame, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    ok, errs, summary = evaluate_probe_trigger_sanity(frame, ctx.to_dict())
    assert ok is True
    assert not errs
    assert float(summary.get("total_density", 0.0)) > 0.001


def test_probe_sanity_fails_when_everything_triggers() -> None:
    frame = _base_frame(3000)
    payload = {
        "quantiles": {
            "F_ofi_z": {"0.9000": -999.0, "0.9500": -999.0, "0.8500": -999.0},
            "abs(F_ofi_z)": {"0.8500": 0.0, "0.9000": 0.0, "0.9500": 0.0},
            "F_intensity_z": {"0.8000": -999.0, "0.9000": -999.0},
            "spread_z": {"0.1000": 999.0, "0.2000": 999.0},
        },
        "nan_ratio": {"F_ofi_z": 0.0, "F_intensity_z": 0.0, "spread_z": 0.0},
        "sample_count": len(frame),
    }
    frame["compression_flag"] = 1
    frame["vacuum_flag"] = 1
    frame["liq_burst_flag"] = 1
    ok, errs, summary = evaluate_probe_trigger_sanity(frame, payload, total_density_max=0.60)
    assert ok is False
    assert any("total_density_too_high" in e for e in errs)
    assert float(summary.get("total_density", 0.0)) > 0.60


def test_probe_sanity_fails_when_nothing_triggers() -> None:
    frame = _base_frame(3000)
    payload = {
        "quantiles": {
            "F_ofi_z": {"0.9000": 999.0, "0.9500": 999.0, "0.8500": 999.0},
            "abs(F_ofi_z)": {"0.8500": 999.0, "0.9000": 999.0, "0.9500": 999.0},
            "F_intensity_z": {"0.8000": 999.0, "0.9000": 999.0},
            "spread_z": {"0.1000": -999.0, "0.2000": -999.0},
        },
        "nan_ratio": {"F_ofi_z": 0.0, "F_intensity_z": 0.0, "spread_z": 0.0},
        "sample_count": len(frame),
    }
    frame["compression_flag"] = 0
    frame["vacuum_flag"] = 0
    frame["liq_burst_flag"] = 0
    ok, errs, summary = evaluate_probe_trigger_sanity(frame, payload, total_density_min=0.001)
    assert ok is False
    assert any("total_density_too_low" in e for e in errs)
    assert float(summary.get("total_density", 1.0)) < 0.001

