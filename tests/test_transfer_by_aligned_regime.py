from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import report_transfer_by_aligned_regime


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"transfer_regime_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_transfer_by_aligned_regime_groups_correctly() -> None:
    tmp = _mk_local_tmp()
    try:
        aligned = tmp / "aligned.parquet"
        pd.DataFrame(
            [
                {"ts_ms": 1, "symbol": "ETHUSDT", "aligned_regime_id": 0},
                {"ts_ms": 2, "symbol": "ETHUSDT", "aligned_regime_id": 0},
                {"ts_ms": 3, "symbol": "ETHUSDT", "aligned_regime_id": 1},
                {"ts_ms": 1, "symbol": "BTCUSDT", "aligned_regime_id": 0},
                {"ts_ms": 2, "symbol": "BTCUSDT", "aligned_regime_id": 1},
                {"ts_ms": 3, "symbol": "BTCUSDT", "aligned_regime_id": 1},
            ]
        ).to_parquet(aligned, index=False)

        pair_root = tmp / "pair"
        pair_root.mkdir(parents=True, exist_ok=True)
        source_eval = pair_root / "eval.parquet"
        source_trades = pair_root / "trades.parquet"
        pd.DataFrame([{"signal": "s1", "test_net_mean": 0.01, "test_sharpe": 1.0, "test_trade_count": 3, "fill_rate": 1.0}]).to_parquet(
            source_eval, index=False
        )
        pd.DataFrame(
            [
                {"ts_ms": 1, "signal": "s1", "net_ret": 0.01},
                {"ts_ms": 2, "signal": "s1", "net_ret": -0.01},
                {"ts_ms": 3, "signal": "s1", "net_ret": 0.02},
            ]
        ).to_parquet(source_trades, index=False)
        target_trades = pair_root / "trades_transfer.parquet"
        pd.DataFrame(
            [
                {"ts_ms": 1, "signal": "s1", "net_ret": 0.01},
                {"ts_ms": 2, "signal": "s1", "net_ret": -0.02},
                {"ts_ms": 3, "signal": "s1", "net_ret": 0.03},
            ]
        ).to_parquet(target_trades, index=False)
        transfer_manifest = pair_root / "manifest_transfer.json"
        _write_json(
            transfer_manifest,
            {
                "source_eval_path": str(source_eval),
                "target_trades_transfer_parquet": str(target_trades),
            },
        )

        matrix_manifest = tmp / "matrix_manifest.json"
        _write_json(
            matrix_manifest,
            {
                "pairs": [
                    {
                        "pair_id": "ETHUSDT_to_BTCUSDT__cal=source",
                        "source_symbol": "ETHUSDT",
                        "target_symbol": "BTCUSDT",
                        "calibration_mode": "source",
                        "ok": True,
                        "transfer_manifest_json": str(transfer_manifest),
                    }
                ]
            },
        )
        out_pq = tmp / "out.parquet"
        out_md = tmp / "out.md"
        old = sys.argv
        try:
            sys.argv = [
                "report_transfer_by_aligned_regime",
                "--matrix-manifest",
                str(matrix_manifest),
                "--aligned-regimes",
                str(aligned),
                "--out-parquet",
                str(out_pq),
                "--out-md",
                str(out_md),
            ]
            assert report_transfer_by_aligned_regime.main() == 0
        finally:
            sys.argv = old
        got = pd.read_parquet(out_pq)
        assert not got.empty
        assert set(got["aligned_regime_id"].tolist()) == {0, 1}
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

