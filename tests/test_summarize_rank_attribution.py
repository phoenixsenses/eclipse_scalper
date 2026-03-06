from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_rank_attribution as sra


def test_summarize_rank_attribution_output(capsys, monkeypatch) -> None:
    p = Path("reports/test_rank_attribution_input.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": 3,
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "failure_reason_top": "fees_dominate",
                "gate_reject_ratio": 0.10,
                "fill_rate_after_gate": 0.45,
                "avg_fee_bps": 2.0,
                "avg_adverse_bps_on_fills": 1.0,
                "avg_raw_return_bps_on_fills": -0.5,
                "avg_net_return_bps_on_fills": -2.5,
                "pass_rate_core": 0.20,
                "npa_core": -1.0e-5,
                "score_raw_core": 1.0e-4,
            },
            {
                "symbol": "BTCUSDT",
                "failure_reason_top": "fees_dominate",
                "gate_reject_ratio": 0.12,
                "fill_rate_after_gate": 0.40,
                "avg_fee_bps": 2.1,
                "avg_adverse_bps_on_fills": 1.2,
                "avg_raw_return_bps_on_fills": -0.4,
                "avg_net_return_bps_on_fills": -2.6,
                "pass_rate_core": 0.10,
                "npa_core": -2.0e-5,
                "score_raw_core": 0.9e-4,
            },
            {
                "symbol": "ETHUSDT",
                "failure_reason_top": "adverse_dominates",
                "gate_reject_ratio": 0.20,
                "fill_rate_after_gate": 0.35,
                "avg_fee_bps": 1.2,
                "avg_adverse_bps_on_fills": 2.4,
                "avg_raw_return_bps_on_fills": -0.8,
                "avg_net_return_bps_on_fills": -2.7,
                "pass_rate_core": 0.30,
                "npa_core": -1.5e-5,
                "score_raw_core": 1.1e-4,
            },
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(p), "--top-n", "2"])
    rc = sra.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "rows_total=3 top_n=2" in out
    assert "Failure Reason Share" in out
    assert "fees_dominate: 2/3" in out
    assert "Next action: fees dominate" in out


def test_summarize_rank_attribution_writes_json(monkeypatch) -> None:
    p = Path("reports/test_rank_attribution_input_out.json")
    out_json = Path("reports/test_rank_attribution_summary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": 2,
        "ranking": [
            {"symbol": "ETHUSDT", "failure_reason_top": "fees_dominate", "gate_reject_ratio": 0.1},
            {"symbol": "BTCUSDT", "failure_reason_top": "mixed", "gate_reject_ratio": 0.7},
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(p), "--top-n", "1", "--out-json", str(out_json)])
    rc = sra.main()
    assert rc == 0
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["rows_total"] == 2
    assert summary["top_n"] == 1
    assert summary["run_summary"]["run_type"] == "summarize_rank_attribution"
    assert summary["run_summary"]["artifacts"]["json"].endswith("test_rank_attribution_summary.json")
