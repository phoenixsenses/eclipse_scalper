from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import compute_calibration, save_calibration
from src.microphys.execution.calibration import save_execution_params
from tools import activate_online_artifacts as tool


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"activate_integrated_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_physics(root: Path, symbol: str = "ETHUSDT", interval_ms: int = 100, n: int = 3000) -> pd.DataFrame:
    ofi = [(-3.0 + 6.0 * (i / max(1, n - 1))) for i in range(n)]
    mid = [100.0]
    for i in range(1, n):
        sign = 1.0 if ofi[i - 1] >= 0 else -1.0
        mid.append(mid[-1] * (1.0 + (0.02 * sign) / 100.0))
    frame = pd.DataFrame(
        {
            "ts_ms": [1_700_000_000_000 + i * 100 for i in range(n)],
            "ts_utc": [f"2024-03-01T00:{(i//60)%60:02d}:{i%60:02d}Z" for i in range(n)],
            "mid": mid,
            "F_ofi_z": ofi,
            "F_intensity_z": [(-2.0 + 4.0 * ((i % 200) / 199.0)) for i in range(n)],
            "spread_z": [(-2.0 + 4.0 * ((i % 100) / 99.0)) for i in range(n)],
            "compression_flag": [1 if i % 30 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 80 == 0 else 0 for i in range(n)],
            "liq_burst_flag": [1 if i % 120 == 0 else 0 for i in range(n)],
        }
    )
    p = root / f"interval_ms={interval_ms}" / f"symbol={symbol}" / "date=2024-03-01"
    p.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p / "physics.parquet", index=False)
    return frame


def _write_exec(path: Path) -> None:
    save_execution_params(
        path,
        {
            "maker_hazard": {"a": 1.0, "b": -0.5, "c": 0.3, "d": 0.0, "fill_threshold": 0.5, "ttl_bars": 5},
            "maker_queue": {"queue_frac": 0.2, "ttl_bars": 5, "min_depth": 1.0},
            "adverse": {"buy_mean": 0.0, "sell_mean": 0.0},
        },
    )


def test_activate_integrated_validate_pass(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        live = tmp / "live"
        physics = tmp / "physics"
        frame = _write_physics(physics)
        cal = tmp / "cal.json"
        save_calibration(compute_calibration(frame, columns=["F_ofi_z", "F_intensity_z", "spread_z"]), cal)
        exe = tmp / "exe.json"
        _write_exec(exe)
        out_report = tmp / "validate.md"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "activate_online_artifacts",
                "--calibration",
                str(cal),
                "--execution",
                str(exe),
                "--live-root",
                str(live),
                "--physics",
                str(physics),
                "--symbol",
                "ETHUSDT",
                "--interval-ms",
                "100",
                "--sanity-days",
                "1",
                "--directional-sanity",
                "--directional-min-triggers",
                "20",
                "--out-validation-report",
                str(out_report),
                "--run-id",
                "r_int",
            ],
        )
        assert tool.main() == 0
        active = json.loads((live / "active_artifacts.json").read_text(encoding="utf-8"))
        assert str(active.get("validation_report_path", "")) == str(out_report)
        assert bool(active.get("directional_sanity_enabled", False)) is True
        rows = [json.loads(x) for x in (live / "calibration_history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert rows and rows[-1].get("validation_passed") is True
        assert int(rows[-1].get("directional_failed_count", 0) or 0) >= 0
        assert out_report.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_activate_integrated_validate_fail(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        live = tmp / "live"
        physics = tmp / "physics"
        _write_physics(physics)
        cal = tmp / "bad_cal.json"
        cal.write_text(
            json.dumps(
                {
                    "quantiles": {
                        "F_ofi_z": {"0.5000": 998.0, "0.9000": 999.0},
                        "abs(F_ofi_z)": {"0.8500": 999.0, "0.9000": 999.0, "0.9500": 999.0},
                        "F_intensity_z": {"0.5000": 998.0, "0.9000": 999.0},
                        "spread_z": {"0.1000": -999.0, "0.2000": -999.0, "0.5000": -999.0},
                    },
                    "nan_ratio": {"F_ofi_z": 0.0, "F_intensity_z": 0.0, "spread_z": 0.0},
                    "sample_count": 1000,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        exe = tmp / "exe.json"
        _write_exec(exe)
        out_report = tmp / "validate_fail.md"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "activate_online_artifacts",
                "--calibration",
                str(cal),
                "--execution",
                str(exe),
                "--live-root",
                str(live),
                "--physics",
                str(physics),
                "--symbol",
                "ETHUSDT",
                "--interval-ms",
                "100",
                "--sanity-days",
                "1",
                "--out-validation-report",
                str(out_report),
            ],
        )
        assert tool.main() == 2
        assert not (live / "active_artifacts.json").exists()
        assert out_report.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
