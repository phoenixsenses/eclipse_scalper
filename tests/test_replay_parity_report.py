from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import replay_parity_report as rpr


class _FakeResult:
    def to_dict(self) -> dict:
        return {
            "sim_count": 3,
            "live_count": 2,
            "matched_count": 1,
            "match_rate_vs_sim": 0.33,
            "sim_fill_rate": 0.5,
            "live_fill_rate": 0.5,
            "fill_rate_delta": 0.0,
            "mean_abs_dt_sec": 1.0,
            "mean_fill_delay_delta_sec": 0.5,
            "mean_pnl_bps_delta": 0.1,
            "mean_adverse_bps_delta": 0.2,
            "matches": [],
        }


def test_replay_parity_report_writes_run_summary(monkeypatch) -> None:
    base = Path("reports/test_replay_parity_report")
    base.mkdir(parents=True, exist_ok=True)
    out_json = base / "out.json"
    out_md = base / "out.md"
    monkeypatch.setattr(rpr, "load_simulated_fill_rows", lambda _: [])
    monkeypatch.setattr(rpr, "load_live_fill_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(rpr, "compute_replay_parity", lambda *_args, **_kwargs: _FakeResult())
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--sim", "logs/x.jsonl", "--live-db", "data/paper_trades.db", "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert rpr.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "replay_parity_report"
    assert payload["run_summary"]["metrics"]["matched_count"] == 1
