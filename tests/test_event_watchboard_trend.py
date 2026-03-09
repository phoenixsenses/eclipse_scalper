from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_watchboard_trend as ewt


def test_build_trend_payload_rising() -> None:
    payload = ewt.build_trend_payload(
        snapshots=[
            {
                "summary": {"top_lane": "spread_stress"},
                "top_event": {"lane": "spread_stress", "level": "elevated", "recommended_action": "show_caution"},
                "lanes": [
                    {"lane": "spread_stress", "priority_score": 125.0},
                    {"lane": "liquidation", "priority_score": 0.0},
                ],
            },
            {
                "summary": {"top_lane": "liquidation"},
                "top_event": {"lane": "liquidation", "level": "severe", "recommended_action": "escalate_monitoring"},
                "lanes": [
                    {"lane": "spread_stress", "priority_score": 50.0},
                    {"lane": "liquidation", "priority_score": 225.0},
                ],
            },
        ],
        source_paths=["a.json", "b.json"],
        out_json="reports/RESEARCH_EVENT_WATCHBOARD_TREND.json",
        out_md="reports/RESEARCH_EVENT_WATCHBOARD_TREND.md",
    )
    assert payload["summary"]["snapshot_count"] == 2
    assert payload["summary"]["trend"] == "rising_fast"
    assert payload["latest"]["top_lane"] == "liquidation"
    assert payload["lane_deltas"][0]["lane"] == "liquidation"
    assert payload["lane_deltas"][0]["trend"] == "rising_fast"


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_watchboard_trend")
    out_dir.mkdir(parents=True, exist_ok=True)
    a = out_dir / "a.json"
    b = out_dir / "b.json"
    a.write_text(json.dumps({"summary": {"top_lane": "spread_stress"}, "top_event": {"lane": "spread_stress", "level": "elevated", "recommended_action": "show_caution"}, "lanes": [{"lane": "spread_stress", "priority_score": 125.0}, {"lane": "liquidation", "priority_score": 0.0}]}), encoding="utf-8")
    b.write_text(json.dumps({"summary": {"top_lane": "liquidation"}, "top_event": {"lane": "liquidation", "level": "severe", "recommended_action": "escalate_monitoring"}, "lanes": [{"lane": "spread_stress", "priority_score": 50.0}, {"lane": "liquidation", "priority_score": 225.0}]}), encoding="utf-8")
    out_json = out_dir / "trend.json"
    out_md = out_dir / "trend.md"
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(a), str(b), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert ewt.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_watchboard_trend"
    assert payload["lane_deltas"][0]["lane"] == "liquidation"
    assert out_md.exists()
