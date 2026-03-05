from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_alpha_multi as tool


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"run_alpha_multi_reports_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_ok_artifacts(run_dir: Path, symbol: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base = run_dir / "artifacts" / "interval_ms=100" / f"symbol={symbol}"
    base.mkdir(parents=True, exist_ok=True)
    cand = base / "candidates_deduped.jsonl"
    cand.write_text(json.dumps({"name": f"{symbol.lower()}_ofi_sig"}) + "\n", encoding="utf-8")
    selected = base / "selected.parquet"
    pd.DataFrame([{"signal": f"{symbol.lower()}_ofi_sig"}]).to_parquet(selected, index=False)
    summary = base / "selection_summary.parquet"
    pd.DataFrame(
        [{"signal": f"{symbol.lower()}_ofi_sig", "test_trade_count": 42, "test_net_mean": 0.001, "regime_concentration": 0.5}]
    ).to_parquet(summary, index=False)
    evalp = base / "eval.parquet"
    pd.DataFrame([{"test_net_mean": 0.001, "test_sharpe": 1.1, "fill_rate": 0.4, "regime_concentration": 0.5}]).to_parquet(
        evalp, index=False
    )
    (run_dir / "pointers.json").write_text(
        json.dumps(
            {
                "candidates_jsonl": str(cand),
                "candidates_deduped_jsonl": str(cand),
                "selected_parquet": str(selected),
                "selection_summary_parquet": str(summary),
                "eval_parquet": str(evalp),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")


def test_run_alpha_multi_with_reports_partial_failure_and_latest(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_root = tmp / "out"
        reports_out = tmp / "reports"
        metrics_out = tmp / "metrics"

        def _fake_main() -> int:
            argv = list(sys.argv)
            symbol = str(argv[argv.index("--symbol") + 1])
            run_dir = Path(str(argv[argv.index("--run-dir") + 1]))
            if symbol == "ETHUSDT":
                _write_ok_artifacts(run_dir, symbol)
                return 0
            run_dir.mkdir(parents=True, exist_ok=True)
            return 2

        monkeypatch.setattr(tool.alpha_pipeline, "main", _fake_main)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_alpha_multi",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--out-root",
                str(out_root),
                "--with-reports",
                "--reports-out",
                str(reports_out),
                "--metrics-out",
                str(metrics_out),
            ],
        )
        assert tool.main() == 0

        rollup_md = reports_out / "rollup.md"
        general_md = reports_out / "generalization.md"
        assert rollup_md.exists()
        assert general_md.exists()
        rollup_txt = rollup_md.read_text(encoding="utf-8")
        general_txt = general_md.read_text(encoding="utf-8")
        assert "Failed Runs" in rollup_txt
        assert "BTCUSDT" in rollup_txt
        assert "Failed Runs" in general_txt
        assert "BTCUSDT" in general_txt

        latest = json.loads((out_root / "LATEST.json").read_text(encoding="utf-8"))
        assert latest["ok"] is False
        assert Path(str(latest["outputs"]["rollup_md"])).exists()
        assert Path(str(latest["outputs"]["generalization_md"])).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_alpha_multi_require_all_ok_returns_nonzero(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_root = tmp / "out"
        reports_out = tmp / "reports"
        metrics_out = tmp / "metrics"

        def _fake_main() -> int:
            argv = list(sys.argv)
            symbol = str(argv[argv.index("--symbol") + 1])
            run_dir = Path(str(argv[argv.index("--run-dir") + 1]))
            if symbol == "ETHUSDT":
                _write_ok_artifacts(run_dir, symbol)
                return 0
            run_dir.mkdir(parents=True, exist_ok=True)
            return 2

        monkeypatch.setattr(tool.alpha_pipeline, "main", _fake_main)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_alpha_multi",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--out-root",
                str(out_root),
                "--with-reports",
                "--reports-out",
                str(reports_out),
                "--metrics-out",
                str(metrics_out),
                "--require-all-ok",
            ],
        )
        assert tool.main() == 2
        assert (reports_out / "rollup.md").exists()
        assert (reports_out / "generalization.md").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

