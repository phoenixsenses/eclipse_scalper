from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import compute_calibration
from src.microphys.live.guardrails import evaluate_probe_directional_sanity


def _frame(n: int = 3000, aligned: bool = True) -> pd.DataFrame:
    ofi = [(-2.0 + 4.0 * (i / max(1, n - 1))) for i in range(n)]
    intensity = [(-1.0 + 2.0 * ((i % 200) / 199.0)) for i in range(n)]
    spread = [(-1.5 + 3.0 * ((i % 100) / 99.0)) for i in range(n)]
    mid = [100.0]
    for i in range(1, n):
        sign = 1.0 if ofi[i - 1] >= 0 else -1.0
        step = 0.02 * (sign if aligned else -sign)
        mid.append(mid[-1] * (1.0 + step / 100.0))
    return pd.DataFrame(
        {
            "ts_ms": [1_700_000_000_000 + i * 100 for i in range(n)],
            "ts_utc": [f"2024-03-01T00:{(i//60)%60:02d}:{i%60:02d}Z" for i in range(n)],
            "mid": mid,
            "F_ofi_z": ofi,
            "F_intensity_z": intensity,
            "spread_z": spread,
            "compression_flag": [1 if i % 30 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 80 == 0 else 0 for i in range(n)],
            "liq_burst_flag": [1 if i % 120 == 0 else 0 for i in range(n)],
        }
    )


def test_directional_sanity_passes_on_aligned_data() -> None:
    frame = _frame(aligned=True)
    cal = compute_calibration(frame, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    ok, errs, summary = evaluate_probe_directional_sanity(
        frame,
        cal.to_dict(),
        min_dir_triggers=30,
        max_fail_probes=2,
        horizons=(1, 5),
    )
    assert ok is True
    assert not errs
    assert int(summary.get("failed_count", 0)) <= 2


def test_directional_sanity_fails_on_inverted_data() -> None:
    frame = _frame(aligned=False)
    cal = compute_calibration(frame, columns=["F_ofi_z", "F_intensity_z", "spread_z"])
    ok, errs, summary = evaluate_probe_directional_sanity(
        frame,
        cal.to_dict(),
        min_dir_triggers=30,
        max_fail_probes=1,
        horizons=(1, 5),
    )
    assert ok is False
    assert any("directional_failed_count_exceeded" in e for e in errs)
    assert int(summary.get("failed_count", 0)) > 1

