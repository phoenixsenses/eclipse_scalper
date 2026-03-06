from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_watchboard_snapshot_append as ewsa


def test_build_append_payload() -> None:
    payload = ewsa.build_append_payload(
        payload={
            "summary": {"top_lane": "liquidation", "state_counts": {"severe": 2}},
            "top_event": {
                "lane": "liquidation",
                "level": "severe",
                "recommended_action": "monitor_only",
                "headline": "Liq top ETH",
            },
            "banner": {
                "headline": "Liq top ETH",
                "recommended_action": "monitor_only",
                "top_lane": "liquidation",
                "top_level": "severe",
            },
            "run_summary": {"run_type": "research_event_watchboard"},
        },
        source="reports/RESEARCH_EVENT_WATCHBOARD_REAL.json",
        history_path="reports/HISTORY.jsonl",
        out_json="reports/APPEND.json",
    )
    assert payload["appended"]["top_lane"] == "liquidation"
    assert payload["run_summary"]["run_type"] == "event_watchboard_snapshot_append"


def test_main_appends_history(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_watchboard_snapshot_append")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "watchboard.json"
    source.write_text(
        json.dumps(
            {
                "summary": {"top_lane": "liquidation", "state_counts": {"severe": 2}},
                "top_event": {
                    "lane": "liquidation",
                    "level": "severe",
                    "recommended_action": "monitor_only",
                    "headline": "Liq top ETH",
                },
                "banner": {
                    "headline": "Liq top ETH",
                    "recommended_action": "monitor_only",
                    "top_lane": "liquidation",
                    "top_level": "severe",
                },
                "run_summary": {"run_type": "research_event_watchboard"},
            }
        ),
        encoding="utf-8",
    )
    history = out_dir / "history.jsonl"
    out_json = out_dir / "append.json"
    monkeypatch.setattr(sys, "argv", ["x", "--source", str(source), "--history-jsonl", str(history), "--out-json", str(out_json)])
    assert ewsa.main() == 0
    lines = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["top_lane"] == "liquidation"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_watchboard_snapshot_append"
