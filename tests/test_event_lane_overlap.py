from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_lane_overlap as elo


def test_build_overlap_payload() -> None:
    rows = [
        {
            "top_lane": "liquidation",
            "lanes": [
                {"lane": "liquidation", "level": "severe", "freshness_status": "fresh"},
                {"lane": "spread_stress", "level": "elevated", "freshness_status": "fresh"},
                {"lane": "fill_toxicity", "level": "quiet", "freshness_status": "stale"},
            ],
        },
        {
            "top_lane": "spread_stress",
            "lanes": [
                {"lane": "liquidation", "level": "severe", "freshness_status": "stale"},
                {"lane": "spread_stress", "level": "elevated", "freshness_status": "fresh"},
            ],
        },
        {
            "top_lane": "return_shock",
            "lanes": [
                {"lane": "return_shock", "level": "severe", "freshness_status": "fresh"},
                {"lane": "liquidation", "level": "quiet", "freshness_status": "stale"},
            ],
        },
    ]
    payload = elo.build_overlap_payload(
        history_rows=rows,
        history_jsonl="reports/history.jsonl",
        min_level="elevated",
        top_n=3,
        out_json="reports/out.json",
        out_md="reports/out.md",
    )
    assert payload["summary"]["lane_count"] == 4
    assert payload["summary"]["active_snapshot_count"] == 3
    assert payload["summary"]["top_overlap_pair"] == "liquidation::spread_stress"
    assert {row["lane"] for row in payload["lane_stats"][:2]} == {"liquidation", "spread_stress"}
    assert payload["strongest_overlaps"][0]["coactive_count"] == 2
    assert payload["run_summary"]["run_type"] == "event_lane_overlap"


def test_main_writes_overlap_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_lane_overlap")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = out_dir / "history.jsonl"
    history.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "top_lane": "liquidation",
                    "lanes": [
                        {"lane": "liquidation", "level": "severe", "freshness_status": "fresh"},
                        {"lane": "spread_stress", "level": "elevated", "freshness_status": "fresh"},
                    ],
                },
                {
                    "top_lane": "spread_stress",
                    "lanes": [
                        {"lane": "liquidation", "level": "severe", "freshness_status": "stale"},
                        {"lane": "spread_stress", "level": "elevated", "freshness_status": "fresh"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "overlap.json"
    out_md = out_dir / "overlap.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--history-jsonl", str(history), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert elo.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["top_overlap_pair"] == "liquidation::spread_stress"
    assert payload["run_summary"]["run_type"] == "event_lane_overlap"
    assert out_md.exists()


def test_build_overlap_payload_falls_back_to_top_event_for_legacy_rows() -> None:
    payload = elo.build_overlap_payload(
        history_rows=[
            {
                "top_lane": "liquidation",
                "top_event": {"lane": "liquidation", "level": "severe", "recommended_action": "monitor_only"},
                "banner": {"top_level": "severe"},
            }
        ],
        history_jsonl="reports/history.jsonl",
        min_level="elevated",
        top_n=3,
        out_json="reports/out.json",
        out_md="reports/out.md",
    )
    assert payload["summary"]["lane_count"] == 1
    assert payload["summary"]["active_snapshot_count"] == 1
    assert payload["lane_stats"][0]["lane"] == "liquidation"
