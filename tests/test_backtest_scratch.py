from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import backtest_scratch as bs


def test_backtest_scratch_main_writes_run_summary(monkeypatch) -> None:
    out_md = Path("reports/test_backtest_scratch/out.md")
    out_json = Path("reports/test_backtest_scratch/out.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        bs,
        "run_analysis",
        lambda **kwargs: {
            "symbol": "ETHUSDT",
            "side": "SELL",
            "regime": "UP",
            "lookback_min": 10,
            "bucket_sec": 1,
            "horizon_sec": 120,
            "min_imbalance": 0.5,
            "min_trade_intensity": 3000.0,
            "max_spread": 0.0003,
            "scratch_taker_fee_bps": 0.0,
            "scratch_slippage_bps": 0.0,
            "exec_model": "passive_realistic",
            "baseline": {"n": 5.0, "mean_net": 0.0001, "scratch_frac": 0.2, "horizon_frac": 0.8},
            "adverse_sweep": [],
            "trailing_sweep": [],
            "best_adverse": None,
            "best_trailing": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--db", "data/microstructure.db", "--symbol", "ETHUSDT", "--out-md", str(out_md), "--out-json", str(out_json)],
    )
    assert bs.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "backtest_scratch"
