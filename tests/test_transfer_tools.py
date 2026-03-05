from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.calibration import CalibrationContext, save_calibration
from tools import eval_transfer, export_selected_specs


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"transfer_tools_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_source_run(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run_source"
    run_dir.mkdir(parents=True, exist_ok=True)
    base = run_dir / "artifacts" / "interval_ms=100" / "symbol=ETHUSDT"
    base.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "name": "ofi_q_signal",
            "side": "buy",
            "condition": {"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": 0.9},
            "entry": "market",
            "horizon_bars": 1,
            "cooldown_bars": 0,
            "regime_filter": [],
            "entry_mode_preference": "both",
            "meta": {"tags": ["ofi"]},
        }
    ]
    cand = base / "candidates_deduped.jsonl"
    cand.write_text("\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in specs) + "\n", encoding="utf-8")
    sel = base / "selected.parquet"
    pd.DataFrame([{"signal": "ofi_q_signal"}]).to_parquet(sel, index=False)
    ev = base / "eval.parquet"
    pd.DataFrame([{"signal": "ofi_q_signal", "split_id": 1, "test_net_mean": 0.002, "test_sharpe": 1.2, "test_trade_count": 10, "fill_rate": 1.0}]).to_parquet(
        ev, index=False
    )
    cal = base / "calibration.json"
    save_calibration(
        CalibrationContext(
            quantiles={"F_ofi_z": {"0.9000": 0.2}, "abs(F_ofi_z)": {"0.9000": 0.2}},
            nan_ratio={"F_ofi_z": 0.0},
            sample_count=100,
        ),
        cal,
    )
    (run_dir / "pointers.json").write_text(
        json.dumps(
            {
                "candidates_deduped_jsonl": str(cand),
                "selected_parquet": str(sel),
                "eval_parquet": str(ev),
                "calibration_json": str(cal),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, cal


def _mk_target_data(root: Path) -> None:
    pdir = root / "physics" / "interval_ms=100" / "symbol=BTCUSDT" / "date=2026-03-01"
    rdir = root / "regimes" / "interval_ms=100" / "symbol=BTCUSDT" / "date=2026-03-01"
    pdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(120):
        rows.append(
            {
                "ts_ms": 1_700_000_000_000 + i * 100,
                "ts_utc": f"2026-03-01T00:00:{i%60:02d}Z",
                "mid": 100.0 + i * 0.01,
                "spread": 0.001,
                "F_ofi_z": 0.5 if i % 3 == 0 else 0.0,
                "r_1": 0.0001,
            }
        )
    pd.DataFrame(rows).to_parquet(pdir / "physics.parquet", index=False)
    pd.DataFrame([{"ts_ms": r["ts_ms"], "regime_id": 1} for r in rows]).to_parquet(rdir / "regimes.parquet", index=False)


def test_export_selected_specs_deterministic() -> None:
    tmp = _mk_local_tmp()
    try:
        run_dir, _ = _mk_source_run(tmp)
        out_root = tmp / "transfer"
        argv = [
            "export_selected_specs",
            "--run-dir",
            str(run_dir),
            "--from",
            "selected",
            "--out",
            str(out_root),
            "--source-symbol",
            "ETHUSDT",
        ]
        old = sys.argv
        try:
            sys.argv = argv
            assert export_selected_specs.main() == 0
            sys.argv = argv
            assert export_selected_specs.main() == 0
        finally:
            sys.argv = old
        out_specs = out_root / "source=ETHUSDT" / f"run={run_dir.name}" / "exported_specs.jsonl"
        a = out_specs.read_bytes()
        b = out_specs.read_bytes()
        assert a == b
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_transfer_uses_source_calibration_mode() -> None:
    tmp = _mk_local_tmp()
    try:
        run_dir, cal_path = _mk_source_run(tmp)
        _mk_target_data(tmp)
        out_root = tmp / "transfer"
        old = sys.argv
        try:
            sys.argv = [
                "export_selected_specs",
                "--run-dir",
                str(run_dir),
                "--from",
                "selected",
                "--out",
                str(out_root),
                "--source-symbol",
                "ETHUSDT",
            ]
            assert export_selected_specs.main() == 0
            exported = out_root / "source=ETHUSDT" / f"run={run_dir.name}" / "exported_specs.jsonl"
            report = tmp / "report.md"
            sys.argv = [
                "eval_transfer",
                "--exported",
                str(exported),
                "--physics",
                str(tmp / "physics"),
                "--regimes",
                str(tmp / "regimes"),
                "--source-symbol",
                "ETHUSDT",
                "--target-symbol",
                "BTCUSDT",
                "--interval-ms",
                "100",
                "--splits",
                "2",
                "--calibration-mode",
                "source",
                "--out",
                str(tmp / "out"),
                "--report",
                str(report),
            ]
            assert eval_transfer.main() == 0
        finally:
            sys.argv = old
        manifest = json.loads((tmp / "out" / "source=ETHUSDT" / "target=BTCUSDT" / "interval_ms=100" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["calibration_mode"] == "source"
        assert Path(manifest["calibration_path_used"]) == cal_path
        assert (tmp / "report.md").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_transfer_target_mode_requires_existing_calibration() -> None:
    tmp = _mk_local_tmp()
    try:
        run_dir, _ = _mk_source_run(tmp)
        _mk_target_data(tmp)
        out_root = tmp / "transfer"
        old = sys.argv
        try:
            sys.argv = [
                "export_selected_specs",
                "--run-dir",
                str(run_dir),
                "--from",
                "selected",
                "--out",
                str(out_root),
                "--source-symbol",
                "ETHUSDT",
            ]
            assert export_selected_specs.main() == 0
            exported = out_root / "source=ETHUSDT" / f"run={run_dir.name}" / "exported_specs.jsonl"
            sys.argv = [
                "eval_transfer",
                "--exported",
                str(exported),
                "--physics",
                str(tmp / "physics"),
                "--regimes",
                str(tmp / "regimes"),
                "--source-symbol",
                "ETHUSDT",
                "--target-symbol",
                "BTCUSDT",
                "--interval-ms",
                "100",
                "--calibration-mode",
                "target",
                "--live-root",
                str(tmp / "live_missing"),
                "--out",
                str(tmp / "out"),
            ]
            assert eval_transfer.main() == 2
        finally:
            sys.argv = old
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

