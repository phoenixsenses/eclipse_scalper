from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import generate_liq_reversal_candidates as glrc


def test_build_candidates_count_and_sort() -> None:
    rows = glrc.build_candidates(
        symbols=["ETHUSDT", "BTCUSDT"],
        horizons=[60],
        min_imbalances=[0.4, 0.3],
        min_trade_intensities=[400.0],
        max_spreads=[0.0003, 0.0002],
        rule="high_liq_reversal_regime",
        regime="liq_reversal_research",
    )
    assert len(rows) == 8
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["max_spread"] == 0.0002
    assert rows[-1]["symbol"] == "ETHUSDT"


def test_main_writes_md_and_json(monkeypatch) -> None:
    out_dir = Path("reports/test_generate_liq_reversal_candidates")
    out_md = out_dir / "candidates.md"
    out_json = out_dir / "candidates.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--symbols",
            "ETHUSDT",
            "--horizons-sec",
            "30,60",
            "--min-imbalances",
            "0.30,0.50",
            "--min-trade-intensities",
            "200,400",
            "--max-spreads",
            "0.00025",
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ],
    )
    rc = glrc.main()
    assert rc == 0
    assert out_md.exists()
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["rule"] == "high_liq_reversal_regime"
    assert payload["count"] == 8
    assert payload["run_summary"]["run_type"] == "generate_liq_reversal_candidates"
    md = out_md.read_text(encoding="utf-8")
    assert "| symbol | rule | regime | horizon_sec | min_imbalance | min_trade_intensity | max_spread | pass |" in md
    assert "| ETHUSDT | high_liq_reversal_regime | liq_reversal_research | 30 | 0.30 | 200 | 0.000250 | YES |" in md
