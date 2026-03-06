from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_partition(root: Path, name: str, symbol: str, interval_ms: int, df: pd.DataFrame) -> None:
    p = root / f"interval_ms={interval_ms}" / f"symbol={symbol}" / "date=2024-03-01"
    p.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p / f"{name}.parquet", index=False)


def _mk_local_tmp() -> Path:
    base = Path("localtests") / f"run_alpha_pipeline_experts_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _physics_df(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i % 60):02d}Z" for i in range(n)],
            "symbol": ["ETHUSDT"] * n,
            "mid": [100.0 + 0.02 * i for i in range(n)],
            "spread": [0.001 + (0.00005 if i % 5 else 0.0) for i in range(n)],
            "F_ofi_z": [(-1.0 + (2.0 * (i / max(1, n - 1)))) for i in range(n)],
            "F_intensity_z": [0.1 + ((i % 10) / 6.0) for i in range(n)],
            "spread_z": [(-0.5 if i % 2 else 0.5) for i in range(n)],
            "compression_flag": [1 if i % 7 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 11 == 0 else 0 for i in range(n)],
        }
    )


def _regime_df(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame({"ts_ms": list(range(n)), "regime_id": [0 if i < (n // 2) else 1 for i in range(n)], "regime_name": ["R0" if i < (n // 2) else "R1" for i in range(n)]})


def test_run_alpha_pipeline_emits_regime_experts_pointers(monkeypatch) -> None:
    from tools import run_alpha_pipeline as rp

    tmp = _mk_local_tmp()
    try:
        physics_root = tmp / "physics"
        regimes_root = tmp / "regimes"
        _write_partition(physics_root, "physics", "ETHUSDT", 100, _physics_df())
        _write_partition(regimes_root, "regimes", "ETHUSDT", 100, _regime_df())
        aligned = tmp / "aligned.parquet"
        pd.DataFrame([{"ts_ms": i, "ts_utc": f"2024-03-01T00:00:{(i % 60):02d}Z", "symbol": "ETHUSDT", "aligned_regime_id": (0 if i < 200 else 1)} for i in range(400)]).to_parquet(
            aligned, index=False
        )
        transfer = tmp / "transfer_by_regime.parquet"
        pd.DataFrame([{"aligned_regime_id": 0, "mean_net_ret": 0.001}, {"aligned_regime_id": 1, "mean_net_ret": -0.001}]).to_parquet(
            transfer, index=False
        )
        run_dir = tmp / "run"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_alpha_pipeline",
                "--symbol",
                "ETHUSDT",
                "--interval-ms",
                "100",
                "--physics",
                str(physics_root),
                "--regimes",
                str(regimes_root),
                "--run-dir",
                str(run_dir),
                "--quick",
                "--build-regime-experts",
                "--aligned-regimes",
                str(aligned),
                "--transfer-by-regime",
                str(transfer),
            ],
        )
        assert rp.main() == 0
        ptr = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
        for key in (
            "ensemble_regime_experts_parquet",
            "ensemble_gating_parquet",
            "ensemble_regime_experts_manifest_json",
            "ensemble_regime_experts_report_md",
            "ensemble_gating_report_md",
            "aligned_regimes_path",
            "transfer_by_regime_path",
        ):
            assert str(ptr.get(key, "")).strip()
            assert Path(str(ptr[key])).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

