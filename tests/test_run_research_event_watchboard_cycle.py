from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_research_event_watchboard_cycle as rwc


def test_build_cycle_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        rwc,
        "build_watchboard_payload",
        lambda **kwargs: {
            "summary": {"top_lane": "liquidation"},
            "top_event": {"recommended_action": "monitor_only"},
        },
    )
    monkeypatch.setattr(
        rwc,
        "build_append_payload",
        lambda **kwargs: {
            "max_history": 2,
            "appended": {"top_lane": "liquidation", "state_counts": {"severe": 1}},
            "run_summary": {"metrics": {}},
        },
    )
    monkeypatch.setattr(rwc, "append_history_record", lambda **kwargs: {"history_rows": 1, "trimmed_rows": 0})
    monkeypatch.setattr(rwc, "_load_history", lambda path: [{"top_lane": "liquidation"}])
    monkeypatch.setattr(
        rwc,
        "build_trend_from_history_payload",
        lambda **kwargs: {"history": {"available_rows": 1}, "summary": {"trend": "flat"}},
    )
    monkeypatch.setattr(
        rwc,
        "build_overlap_payload",
        lambda **kwargs: {"summary": {"top_overlap_pair": "liquidation::spread_stress", "active_snapshot_count": 1}},
    )
    monkeypatch.setattr(
        rwc,
        "build_operator_brief_payload",
        lambda **kwargs: {"brief": {"headline": "brief", "operator_note": "note"}},
    )
    out_dir = Path("localtests/test_run_research_event_watchboard_cycle")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = rwc.build_cycle_payload(
        micro_db="data/microstructure.db",
        trade_source="data/live/papertrades_live.parquet",
        symbols=["ETHUSDT", "BTCUSDT"],
        lookback_min=240,
        bucket_sec=5,
        recent_limit=20,
        top_n=5,
        watchboard_json=str(out_dir / "watchboard.json"),
        watchboard_md=str(out_dir / "watchboard.md"),
        history_jsonl=str(out_dir / "history.jsonl"),
        max_history=2,
        append_json=str(out_dir / "append.json"),
        overlap_json=str(out_dir / "overlap.json"),
        overlap_md=str(out_dir / "overlap.md"),
        overlap_top_n=5,
        trend_json=str(out_dir / "trend.json"),
        trend_md=str(out_dir / "trend.md"),
        brief_json=str(out_dir / "brief.json"),
        brief_md=str(out_dir / "brief.md"),
        out_json=str(out_dir / "cycle.json"),
        out_md=str(out_dir / "cycle.md"),
    )
    assert payload["summary"]["top_lane"] == "liquidation"
    assert payload["summary"]["trend"] == "flat"
    assert payload["summary"]["trimmed_rows"] == 0
    assert payload["summary"]["top_overlap_pair"] == "liquidation::spread_stress"
    assert payload["run_summary"]["run_type"] == "run_research_event_watchboard_cycle"
    assert (out_dir / "watchboard.json").exists()
    assert (out_dir / "append.json").exists()
    assert (out_dir / "overlap.json").exists()
    assert (out_dir / "trend.json").exists()
    assert (out_dir / "brief.json").exists()


def test_main_writes_files(monkeypatch) -> None:
    monkeypatch.setattr(
        rwc,
        "build_cycle_payload",
        lambda **kwargs: {
            "watchboard_json": "reports/RESEARCH_EVENT_WATCHBOARD.json",
            "append_json": "reports/RESEARCH_EVENT_WATCHBOARD_SNAPSHOT_APPEND.json",
            "overlap_json": "reports/EVENT_LANE_OVERLAP.json",
            "trend_json": "reports/RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.json",
            "brief_json": "reports/RESEARCH_EVENT_OPERATOR_BRIEF.json",
            "history_jsonl": "reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl",
            "summary": {"top_lane": "liquidation", "top_action": "monitor_only", "history_rows": 1, "trend": "flat", "trimmed_rows": 0, "top_overlap_pair": "liquidation::spread_stress"},
            "run_summary": {
                "version": "v1",
                "run_type": "run_research_event_watchboard_cycle",
                "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
                "metrics": {"top_lane": "liquidation", "history_rows": 1, "trend": "flat", "trimmed_rows": 0, "top_overlap_pair": "liquidation::spread_stress"},
                "artifacts": {"json": "reports/x.json", "md": "reports/x.md"},
            },
        },
    )
    out_dir = Path("localtests/test_run_research_event_watchboard_cycle_main")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cycle.json"
    out_md = out_dir / "cycle.md"
    monkeypatch.setattr(sys, "argv", ["x", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rwc.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "run_research_event_watchboard_cycle"
    assert out_md.exists()
