from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_transfer_matrix as tool


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"transfer_matrix_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _mk_alpha_multi_latest(root: Path) -> Path:
    alpha_multi = root / "data" / "runs" / "alpha_multi"
    multi_run = alpha_multi / "multi_fake"
    eth = multi_run / "symbol=ETHUSDT"
    btc = multi_run / "symbol=BTCUSDT"
    eth.mkdir(parents=True, exist_ok=True)
    btc.mkdir(parents=True, exist_ok=True)
    _write_json(
        multi_run / "manifest.json",
        {
            "runs": [
                {"symbol": "ETHUSDT", "run_dir": str(eth), "ok": True},
                {"symbol": "BTCUSDT", "run_dir": str(btc), "ok": True},
            ]
        },
    )
    _write_json(alpha_multi / "LATEST.json", {"multi_run_dir": str(multi_run)})
    return alpha_multi / "LATEST.json"


def test_run_transfer_matrix_writes_manifest_report_and_latest(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        latest_multi = _mk_alpha_multi_latest(tmp)

        def _fake_export_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            run_dir = Path(str(argv[argv.index("--run-dir") + 1]))
            out = Path(str(argv[argv.index("--out") + 1]))
            exported = out / f"source={source}" / f"run={run_dir.name}" / "exported_specs.jsonl"
            exported.parent.mkdir(parents=True, exist_ok=True)
            exported.write_text(
                json.dumps(
                    {
                        "name": "s1",
                        "side": "buy",
                        "condition": {"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": 0.9},
                        "entry": "market",
                        "horizon_bars": 1,
                        "cooldown_bars": 0,
                        "regime_filter": [],
                        "entry_mode_preference": "both",
                        "meta": {},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                exported.parent / "manifest.json",
                {"exported_specs_jsonl": str(exported), "pointers": {"eval_parquet": str(run_dir / "dummy_eval.parquet")}},
            )
            pd.DataFrame([{"signal": "s1", "test_net_mean": 0.01, "test_sharpe": 1.0, "test_trade_count": 10, "fill_rate": 1.0}]).to_parquet(
                run_dir / "dummy_eval.parquet", index=False
            )
            return 0

        def _fake_eval_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            target = str(argv[argv.index("--target-symbol") + 1])
            out = Path(str(argv[argv.index("--out") + 1]))
            report = Path(str(argv[argv.index("--report") + 1]))
            interval = int(argv[argv.index("--interval-ms") + 1])
            base = out / f"source={source}" / f"target={target}" / f"interval_ms={interval}"
            base.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "signal": "s1",
                        "source_test_net_mean": 0.01,
                        "target_test_net_mean": 0.005,
                        "source_test_sharpe": 1.0,
                        "target_test_sharpe": 0.4,
                        "source_test_trade_count": 10,
                        "target_test_trade_count": 8,
                        "source_fill_rate": 1.0,
                        "target_fill_rate": 0.8,
                        "delta_sharpe": -0.6,
                        "delta_net_mean": -0.005,
                    }
                ]
            ).to_parquet(base / "eval_transfer.parquet", index=False)
            pd.DataFrame([{"signal": "s1", "net_ret": 0.001}]).to_parquet(base / "trades_transfer.parquet", index=False)
            _write_json(base / "manifest.json", {"calibration_path_used": "cal.json"})
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("# ok\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(tool.export_selected_specs, "main", _fake_export_main)
        monkeypatch.setattr(tool.eval_transfer, "main", _fake_eval_main)
        old = sys.argv
        try:
            sys.argv = [
                "run_transfer_matrix",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--alpha-multi-latest",
                str(latest_multi),
                "--out",
                str(tmp / "out"),
                "--reports-out",
                str(tmp / "reports"),
            ]
            assert tool.main() == 0
        finally:
            sys.argv = old

        assert (tmp / "out" / "LATEST.json").exists()
        assert (tmp / "reports" / "transfer_matrix.md").exists()
        assert (tmp / "out" / "transfer_matrix.parquet").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_transfer_matrix_partial_failure_and_require_all_ok(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        latest_multi = _mk_alpha_multi_latest(tmp)

        def _fake_export_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            run_dir = Path(str(argv[argv.index("--run-dir") + 1]))
            out = Path(str(argv[argv.index("--out") + 1]))
            exported = out / f"source={source}" / f"run={run_dir.name}" / "exported_specs.jsonl"
            exported.parent.mkdir(parents=True, exist_ok=True)
            exported.write_text(
                json.dumps(
                    {
                        "name": "s1",
                        "side": "buy",
                        "condition": {"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": 0.9},
                        "entry": "market",
                        "horizon_bars": 1,
                        "cooldown_bars": 0,
                        "regime_filter": [],
                        "entry_mode_preference": "both",
                        "meta": {},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                exported.parent / "manifest.json",
                {"exported_specs_jsonl": str(exported), "pointers": {"eval_parquet": str(run_dir / "dummy_eval.parquet")}},
            )
            pd.DataFrame([{"signal": "s1", "test_net_mean": 0.01, "test_sharpe": 1.0, "test_trade_count": 10, "fill_rate": 1.0}]).to_parquet(
                run_dir / "dummy_eval.parquet", index=False
            )
            return 0

        def _fake_eval_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            target = str(argv[argv.index("--target-symbol") + 1])
            out = Path(str(argv[argv.index("--out") + 1]))
            report = Path(str(argv[argv.index("--report") + 1]))
            interval = int(argv[argv.index("--interval-ms") + 1])
            if source == "ETHUSDT" and target == "BTCUSDT":
                return 2
            base = out / f"source={source}" / f"target={target}" / f"interval_ms={interval}"
            base.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "signal": "s1",
                        "source_test_net_mean": 0.01,
                        "target_test_net_mean": 0.005,
                        "source_test_sharpe": 1.0,
                        "target_test_sharpe": 0.4,
                        "source_test_trade_count": 10,
                        "target_test_trade_count": 8,
                        "source_fill_rate": 1.0,
                        "target_fill_rate": 0.8,
                        "delta_sharpe": -0.6,
                        "delta_net_mean": -0.005,
                    }
                ]
            ).to_parquet(base / "eval_transfer.parquet", index=False)
            _write_json(base / "manifest.json", {"calibration_path_used": "cal.json"})
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("# ok\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(tool.export_selected_specs, "main", _fake_export_main)
        monkeypatch.setattr(tool.eval_transfer, "main", _fake_eval_main)
        old = sys.argv
        try:
            sys.argv = [
                "run_transfer_matrix",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--alpha-multi-latest",
                str(latest_multi),
                "--out",
                str(tmp / "out"),
                "--reports-out",
                str(tmp / "reports"),
            ]
            assert tool.main() == 0
            sys.argv = [
                "run_transfer_matrix",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--alpha-multi-latest",
                str(latest_multi),
                "--out",
                str(tmp / "out2"),
                "--reports-out",
                str(tmp / "reports2"),
                "--require-all-ok",
            ]
            assert tool.main() == 2
        finally:
            sys.argv = old

        assert (tmp / "reports" / "transfer_matrix.md").exists()
        text = (tmp / "reports" / "transfer_matrix.md").read_text(encoding="utf-8")
        assert "Failed Directions" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_transfer_matrix_with_regime_alignment(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        latest_multi = _mk_alpha_multi_latest(tmp)

        def _fake_export_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            run_dir = Path(str(argv[argv.index("--run-dir") + 1]))
            out = Path(str(argv[argv.index("--out") + 1]))
            exported = out / f"source={source}" / f"run={run_dir.name}" / "exported_specs.jsonl"
            exported.parent.mkdir(parents=True, exist_ok=True)
            exported.write_text(
                json.dumps(
                    {
                        "name": "s1",
                        "side": "buy",
                        "condition": {"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": 0.9},
                        "entry": "market",
                        "horizon_bars": 1,
                        "cooldown_bars": 0,
                        "regime_filter": [],
                        "entry_mode_preference": "both",
                        "meta": {},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                exported.parent / "manifest.json",
                {"exported_specs_jsonl": str(exported), "pointers": {"eval_parquet": str(run_dir / "dummy_eval.parquet")}},
            )
            pd.DataFrame([{"signal": "s1", "test_net_mean": 0.01, "test_sharpe": 1.0, "test_trade_count": 10, "fill_rate": 1.0}]).to_parquet(
                run_dir / "dummy_eval.parquet", index=False
            )
            return 0

        def _fake_eval_main() -> int:
            argv = list(sys.argv)
            source = str(argv[argv.index("--source-symbol") + 1])
            target = str(argv[argv.index("--target-symbol") + 1])
            out = Path(str(argv[argv.index("--out") + 1]))
            report = Path(str(argv[argv.index("--report") + 1]))
            interval = int(argv[argv.index("--interval-ms") + 1])
            base = out / f"source={source}" / f"target={target}" / f"interval_ms={interval}"
            base.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "signal": "s1",
                        "source_test_net_mean": 0.01,
                        "target_test_net_mean": 0.005,
                        "source_test_sharpe": 1.0,
                        "target_test_sharpe": 0.4,
                        "source_test_trade_count": 10,
                        "target_test_trade_count": 8,
                        "source_fill_rate": 1.0,
                        "target_fill_rate": 0.8,
                        "delta_sharpe": -0.6,
                        "delta_net_mean": -0.005,
                    }
                ]
            ).to_parquet(base / "eval_transfer.parquet", index=False)
            pd.DataFrame([{"ts_ms": 1, "signal": "s1", "net_ret": 0.001}]).to_parquet(base / "trades_transfer.parquet", index=False)
            _write_json(base / "manifest.json", {"calibration_path_used": "cal.json", "source_eval_path": str(Path(argv[argv.index("--exported")+1]).parent / "dummy_eval.parquet"), "target_trades_transfer_parquet": str(base / "trades_transfer.parquet")})
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("# ok\n", encoding="utf-8")
            return 0

        def _fake_align_main() -> int:
            argv = list(sys.argv)
            out = Path(str(argv[argv.index("--out") + 1]))
            interval = int(argv[argv.index("--interval-ms") + 1])
            p = out / f"interval_ms={interval}" / "aligned_regimes.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"ts_ms": 1, "symbol": "ETHUSDT", "aligned_regime_id": 0},
                    {"ts_ms": 1, "symbol": "BTCUSDT", "aligned_regime_id": 0},
                ]
            ).to_parquet(p, index=False)
            report = Path(str(argv[argv.index("--report") + 1]))
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("# alignment\n", encoding="utf-8")
            return 0

        def _fake_transfer_regime_main() -> int:
            argv = list(sys.argv)
            out_pq = Path(str(argv[argv.index("--out-parquet") + 1]))
            out_md = Path(str(argv[argv.index("--out-md") + 1]))
            out_pq.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"source_symbol": "ETHUSDT", "target_symbol": "BTCUSDT", "aligned_regime_id": 0}]).to_parquet(out_pq, index=False)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("# transfer by regime\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(tool.export_selected_specs, "main", _fake_export_main)
        monkeypatch.setattr(tool.eval_transfer, "main", _fake_eval_main)
        monkeypatch.setattr(tool.build_regime_alignment, "main", _fake_align_main)
        monkeypatch.setattr(tool.report_transfer_by_aligned_regime, "main", _fake_transfer_regime_main)
        old = sys.argv
        try:
            sys.argv = [
                "run_transfer_matrix",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--alpha-multi-latest",
                str(latest_multi),
                "--out",
                str(tmp / "out"),
                "--reports-out",
                str(tmp / "reports"),
                "--with-regime-alignment",
            ]
            assert tool.main() == 0
        finally:
            sys.argv = old

        latest = json.loads((tmp / "out" / "LATEST.json").read_text(encoding="utf-8"))
        assert latest["aligned_regimes_parquet"]
        assert latest["transfer_by_aligned_regime_md"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
