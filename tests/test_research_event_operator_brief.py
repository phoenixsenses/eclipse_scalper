from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import research_event_operator_brief as reob


def test_build_operator_brief_payload() -> None:
    out_dir = Path("localtests/test_research_event_operator_brief")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard_path = out_dir / "watchboard.json"
    trend_path = out_dir / "trend.json"
    watchboard_path.write_text(
        json.dumps(
            {
                "summary": {"top_lane": "liquidation"},
                "top_event": {"recommended_action": "monitor_only", "headline": "top headline"},
                "lanes": [
                    {"lane": "liquidation", "state_level": "severe", "freshness_status": "stale"},
                    {"lane": "spread_stress", "state_level": "quiet", "freshness_status": "fresh"},
                ],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(
        json.dumps(
            {
                "summary": {"trend": "rising"},
                "lane_deltas": [{"lane": "return_shock", "trend": "rising_fast", "delta_priority_score": 125.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = reob.build_operator_brief_payload(
        watchboard_json=str(watchboard_path),
        trend_json=str(trend_path),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["top_lane"] == "liquidation"
    assert payload["summary"]["stale_lane_count"] == 1
    assert payload["summary"]["trend"] == "rising"
    assert payload["summary"]["strongest_delta_lane"] == "return_shock"
    assert payload["brief"]["strongest_delta"]["trend"] == "rising_fast"
    assert payload["run_summary"]["run_type"] == "research_event_operator_brief"


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_research_event_operator_brief_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard_path = out_dir / "watchboard.json"
    trend_path = out_dir / "trend.json"
    watchboard_path.write_text(json.dumps({"summary": {"top_lane": "liquidation"}, "top_event": {}, "lanes": []}), encoding="utf-8")
    trend_path.write_text(json.dumps({"summary": {"trend": "flat"}, "lane_deltas": []}), encoding="utf-8")

    out_json = out_dir / "brief.json"
    out_md = out_dir / "brief.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--watchboard-json",
            str(watchboard_path),
            "--trend-json",
            str(trend_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    assert reob.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "research_event_operator_brief"
    assert out_md.exists()
