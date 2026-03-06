from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tools import toxicity_report as tr


def test_toxicity_report_writes_run_summary(monkeypatch) -> None:
    base = Path("reports/test_toxicity_report")
    base.mkdir(parents=True, exist_ok=True)
    inp = base / "input.csv"
    out_json = base / "out.json"
    out_md = base / "out.md"
    pd.DataFrame(
        [
            {"side": "buy", "pnl_bps": 1.0, "max_adverse_bps": 2.0},
            {"side": "sell", "pnl_bps": -0.5, "max_adverse_bps": 1.5},
        ]
    ).to_csv(inp, index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(inp), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert tr.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "toxicity_report"
    assert payload["run_summary"]["metrics"]["side_count"] == 2
