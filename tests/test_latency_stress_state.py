from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import latency_stress_state as lss


def test_build_state_payload_severe() -> None:
    payload = lss.build_state_payload(
        source="data/live/papertrades_live.parquet",
        diag={
            "rows": 20,
            "fill_rate": 0.18,
            "latency_fill_delay_sec_p50": 4.5,
            "latency_fill_delay_sec_p95": 12.0,
            "latency_impact_vs_net_corr": -0.3,
            "queue_competition_score": 0.5,
            "toxicity_score": 0.8,
            "adverse_selection_bps_mean": 1.2,
        },
        out_json="reports/LATENCY_STRESS_STATE.json",
        out_md="reports/LATENCY_STRESS_STATE.md",
    )
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "escalate_monitoring"


def test_main_writes_files(monkeypatch) -> None:
    monkeypatch.setattr(
        lss,
        "compute_execution_diagnostics",
        lambda df: {
            "rows": 5,
            "fill_rate": 0.35,
            "latency_fill_delay_sec_p50": 2.5,
            "latency_fill_delay_sec_p95": 6.0,
            "latency_impact_vs_net_corr": -0.1,
            "queue_competition_score": 0.3,
            "toxicity_score": 0.4,
            "adverse_selection_bps_mean": 0.8,
        },
    )
    monkeypatch.setattr(lss, "_load_rows", lambda path: object())
    out_dir = Path("localtests/test_latency_stress_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    monkeypatch.setattr(sys, "argv", ["x", "--in", "dummy.csv", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert lss.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "latency_stress_state"
    assert payload["state"]["level"] == "elevated"
    assert out_md.exists()
