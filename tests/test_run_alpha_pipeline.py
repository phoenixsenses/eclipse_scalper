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
    base = Path("localtests") / f"run_alpha_pipeline_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _physics_df(n: int = 500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i % 60):02d}Z" for i in range(n)],
            "symbol": ["ETHUSDT"] * n,
            "mid": [100.0 + 0.01 * i for i in range(n)],
            "spread": [0.001 + (0.0001 if i % 5 else 0.0) for i in range(n)],
            "F_ofi_z": [(-1.0 + (2.0 * (i / max(1, n - 1)))) for i in range(n)],
            "F_intensity_z": [0.2 + ((i % 10) / 5.0) for i in range(n)],
            "spread_z": [(-0.5 if i % 2 else 0.5) for i in range(n)],
            "compression_flag": [1 if i % 7 == 0 else 0 for i in range(n)],
            "vacuum_flag": [1 if i % 11 == 0 else 0 for i in range(n)],
        }
    )


def _regime_df(n: int = 500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "regime_id": [0 if i < (n // 2) else 1 for i in range(n)],
            "regime_name": ["up" if i < (n // 2) else "down" for i in range(n)],
        }
    )


def test_run_alpha_pipeline_creates_manifests(monkeypatch) -> None:
    from tools import run_alpha_pipeline as rp

    tmp_path = _mk_local_tmp()
    physics_root = tmp_path / "physics"
    regimes_root = tmp_path / "regimes"
    _write_partition(physics_root, "physics", "ETHUSDT", 100, _physics_df())
    _write_partition(regimes_root, "regimes", "ETHUSDT", 100, _regime_df())
    run_dir = tmp_path / "runA"

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
        ],
    )
    rc = rp.main()
    assert rc == 0
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "pointers.json").exists()
    assert (run_dir / "params.json").exists()
    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    assert "candidates_jsonl" in pointers
    assert "eval_parquet" in pointers
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_alpha_pipeline_fail_fast_triggered_zero(monkeypatch) -> None:
    from tools import run_alpha_pipeline as rp
    from src.microphys.alpha.spec import SignalSpec

    tmp_path = _mk_local_tmp()
    physics_root = tmp_path / "physics"
    regimes_root = tmp_path / "regimes"
    _write_partition(physics_root, "physics", "ETHUSDT", 100, _physics_df())
    _write_partition(regimes_root, "regimes", "ETHUSDT", 100, _regime_df())
    run_dir = tmp_path / "runB"

    def _fake_generate_candidates(**kwargs):
        return [
            SignalSpec(
                name="x",
                side="buy",
                condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 0.0},
                meta={"calibration_triggered": 0},
            )
        ]

    monkeypatch.setattr(rp, "generate_candidates", _fake_generate_candidates)
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
        ],
    )
    rc = rp.main()
    assert rc == 2
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m.get("status") == "failed"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_alpha_pipeline_resume_skips_generate(monkeypatch) -> None:
    from tools import run_alpha_pipeline as rp

    tmp_path = _mk_local_tmp()
    physics_root = tmp_path / "physics"
    regimes_root = tmp_path / "regimes"
    _write_partition(physics_root, "physics", "ETHUSDT", 100, _physics_df())
    _write_partition(regimes_root, "regimes", "ETHUSDT", 100, _regime_df())
    run_dir = tmp_path / "runC"

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
        ],
    )
    assert rp.main() == 0

    def _boom(**kwargs):
        raise AssertionError("generate_candidates should not run in resume mode")

    monkeypatch.setattr(rp, "generate_candidates", _boom)
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
            "--resume",
            "--quick",
        ],
    )
    assert rp.main() == 0
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_alpha_pipeline_execution_params_pointer(monkeypatch) -> None:
    from tools import run_alpha_pipeline as rp

    tmp_path = _mk_local_tmp()
    physics_root = tmp_path / "physics"
    regimes_root = tmp_path / "regimes"
    _write_partition(physics_root, "physics", "ETHUSDT", 100, _physics_df())
    _write_partition(regimes_root, "regimes", "ETHUSDT", 100, _regime_df())
    run_dir = tmp_path / "runD"

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
            "--calibrate-execution",
        ],
    )
    assert rp.main() == 0
    pointers = json.loads((run_dir / "pointers.json").read_text(encoding="utf-8"))
    p = Path(str(pointers.get("execution_params_json", "")))
    assert str(pointers.get("execution_params_json", "")).strip()
    assert p.exists()
    assert str(pointers.get("execution_realism_report_md", "")).strip()
    shutil.rmtree(tmp_path, ignore_errors=True)
