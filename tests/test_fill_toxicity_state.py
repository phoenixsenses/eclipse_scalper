from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import fill_toxicity_state as fts


def test_build_state_payload_severe() -> None:
    payload = fts.build_state_payload(
        source="data/live/papertrades_live.parquet",
        report_payload={
            "rows": 10,
            "sides": {
                "buy": {"rows": 5, "adverse_bps_mean": 2.2, "pnl_bps_mean": -0.5, "toxicity_score": 1.7},
                "sell": {"rows": 5, "adverse_bps_mean": 0.5, "pnl_bps_mean": 0.2, "toxicity_score": 0.3},
            },
        },
        out_json="reports/FILL_TOXICITY_STATE.json",
        out_md="reports/FILL_TOXICITY_STATE.md",
    )
    assert payload["top_side"] == "buy"
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "reduce_passive_aggression"


def test_main_writes_files(monkeypatch) -> None:
    monkeypatch.setattr(
        fts,
        "build_toxicity_report",
        lambda df: {
            "rows": 4,
            "sides": {
                "sell": {"rows": 4, "adverse_bps_mean": 1.2, "pnl_bps_mean": -0.2, "toxicity_score": 0.9}
            },
        },
    )
    monkeypatch.setattr(fts, "_load_rows", lambda path: object())
    out_dir = Path("localtests/test_fill_toxicity_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    monkeypatch.setattr(sys, "argv", ["x", "--in", "dummy.csv", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert fts.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "fill_toxicity_state"
    assert payload["state"]["level"] == "elevated"
    assert out_md.exists()
