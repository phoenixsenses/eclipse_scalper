from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_watchboard_trend_from_history as ewth


def test_build_trend_from_history_payload() -> None:
    payload = ewth.build_trend_from_history_payload(
        history_rows=[
            {
                "source": "a.json",
                "top_lane": "spread_stress",
                "state_counts": {"elevated": 1},
                "lanes": [
                    {"lane": "spread_stress", "priority_score": 125.0},
                    {"lane": "liquidation", "priority_score": 0.0},
                ],
                "top_event": {"lane": "spread_stress", "level": "elevated", "recommended_action": "show_caution", "headline": "Spread top"},
                "banner": {"headline": "Spread top", "recommended_action": "show_caution", "top_lane": "spread_stress", "top_level": "elevated"},
            },
            {
                "source": "b.json",
                "top_lane": "liquidation",
                "state_counts": {"severe": 1},
                "lanes": [
                    {"lane": "spread_stress", "priority_score": 50.0},
                    {"lane": "liquidation", "priority_score": 225.0},
                ],
                "top_event": {"lane": "liquidation", "level": "severe", "recommended_action": "monitor_only", "headline": "Liq top"},
                "banner": {"headline": "Liq top", "recommended_action": "monitor_only", "top_lane": "liquidation", "top_level": "severe"},
            },
        ],
        history_path="reports/HISTORY.jsonl",
        last_n=2,
        out_json="reports/TREND.json",
        out_md="reports/TREND.md",
    )
    assert payload["run_summary"]["run_type"] == "event_watchboard_trend_from_history"
    assert payload["history"]["used_rows"] == 2
    assert payload["summary"]["end_top_lane"] == "liquidation"
    assert payload["lane_deltas"][0]["lane"] == "liquidation"


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_watchboard_trend_from_history")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = out_dir / "history.jsonl"
    history.write_text(
        json.dumps(
            {
                "source": "a.json",
                "top_lane": "spread_stress",
                "state_counts": {"elevated": 1},
                "lanes": [{"lane": "spread_stress", "priority_score": 125.0}, {"lane": "liquidation", "priority_score": 0.0}],
                "top_event": {"lane": "spread_stress", "level": "elevated", "recommended_action": "show_caution", "headline": "Spread top"},
                "banner": {"headline": "Spread top", "recommended_action": "show_caution", "top_lane": "spread_stress", "top_level": "elevated"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "source": "b.json",
                "top_lane": "liquidation",
                "state_counts": {"severe": 1},
                "lanes": [{"lane": "spread_stress", "priority_score": 50.0}, {"lane": "liquidation", "priority_score": 225.0}],
                "top_event": {"lane": "liquidation", "level": "severe", "recommended_action": "monitor_only", "headline": "Liq top"},
                "banner": {"headline": "Liq top", "recommended_action": "monitor_only", "top_lane": "liquidation", "top_level": "severe"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_json = out_dir / "trend.json"
    out_md = out_dir / "trend.md"
    monkeypatch.setattr(sys, "argv", ["x", "--history-jsonl", str(history), "--last-n", "2", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert ewth.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_watchboard_trend_from_history"
    assert payload["lane_deltas"][0]["lane"] == "liquidation"
    assert out_md.exists()
