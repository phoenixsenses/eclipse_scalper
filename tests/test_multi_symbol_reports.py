from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import report_generalization, report_multi_symbol_rollup


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"multi_reports_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_run(root: Path, symbol: str, with_optional: bool = True) -> Path:
    rd = root / f"symbol={symbol}"
    rd.mkdir(parents=True, exist_ok=True)
    ptr = {}
    base = rd / "artifacts" / "interval_ms=100" / f"symbol={symbol}"
    base.mkdir(parents=True, exist_ok=True)
    cand = base / "candidates_deduped.jsonl"
    cand.write_text(
        "\n".join(
            [
                json.dumps({"name": f"{symbol.lower()}_ofi_sig"}),
                json.dumps({"name": f"{symbol.lower()}_compression_sig"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ptr["candidates_deduped_jsonl"] = str(cand)
    if with_optional:
        sel = base / "selected.parquet"
        pd.DataFrame([{"signal": f"{symbol.lower()}_ofi_sig"}]).to_parquet(sel, index=False)
        summ = base / "selection_summary.parquet"
        pd.DataFrame(
            [
                {"signal": f"{symbol.lower()}_ofi_sig", "test_net_mean": 0.1, "regime_concentration": 0.4},
                {"signal": f"{symbol.lower()}_compression_sig", "test_net_mean": 0.05, "regime_concentration": 0.5},
            ]
        ).to_parquet(summ, index=False)
        evalp = base / "eval.parquet"
        pd.DataFrame([{"test_net_mean": 0.1, "test_sharpe": 1.2, "fill_rate": 0.6, "regime_concentration": 0.4}]).to_parquet(evalp, index=False)
        ptr["selected_parquet"] = str(sel)
        ptr["selection_summary_parquet"] = str(summ)
        ptr["eval_parquet"] = str(evalp)
    (rd / "pointers.json").write_text(json.dumps(ptr) + "\n", encoding="utf-8")
    (rd / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    return rd


def test_multi_rollup_and_generalization_handle_missing_optional(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        multi = tmp / "multi"
        multi.mkdir(parents=True, exist_ok=True)
        r1 = _mk_run(multi, "ETHUSDT", with_optional=True)
        r2 = _mk_run(multi, "BTCUSDT", with_optional=False)
        manifest = multi / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "runs": [
                        {"symbol": "ETHUSDT", "run_dir": str(r1), "ok": True},
                        {"symbol": "BTCUSDT", "run_dir": str(r2), "ok": True},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

        out_md = tmp / "rollup.md"
        out_pq = tmp / "rollup.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "report_multi_symbol_rollup",
                "--multi-manifest",
                str(manifest),
                "--out-md",
                str(out_md),
                "--out-parquet",
                str(out_pq),
            ],
        )
        assert report_multi_symbol_rollup.main() == 0
        assert out_md.exists() and out_pq.exists()

        g_md = tmp / "generalization.md"
        g_pq = tmp / "generalization.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "report_generalization",
                "--multi-manifest",
                str(manifest),
                "--out-md",
                str(g_md),
                "--out-parquet",
                str(g_pq),
            ],
        )
        assert report_generalization.main() == 0
        assert g_md.exists() and g_pq.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

