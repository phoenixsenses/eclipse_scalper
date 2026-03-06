from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_lane_consolidation as elc


def test_build_consolidation_payload() -> None:
    out_dir = Path("localtests/test_event_lane_consolidation")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    overlap = out_dir / "overlap.json"
    watchboard.write_text(
        json.dumps({"summary": {"top_lane": "spread_stress"}, "top_event": {"lane": "spread_stress"}}),
        encoding="utf-8",
    )
    overlap.write_text(
        json.dumps(
            {
                "summary": {"top_overlap_pair": "spread_stress::volume_vacuum"},
                "lane_stats": [
                    {"lane": "spread_stress", "active_count": 5, "top_count": 3},
                    {"lane": "volume_vacuum", "active_count": 5, "top_count": 1},
                    {"lane": "return_shock", "active_count": 2, "top_count": 1},
                ],
                "strongest_overlaps": [
                    {"lane_a": "spread_stress", "lane_b": "volume_vacuum", "jaccard": 0.9, "coactive_count": 4},
                    {"lane_a": "return_shock", "lane_b": "spread_stress", "jaccard": 0.3, "coactive_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = elc.build_consolidation_payload(
        watchboard_json=str(watchboard),
        overlap_json=str(overlap),
        top_n=2,
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["top_lane"] == "spread_stress"
    assert payload["summary"]["decision_count"] == 2
    assert payload["decisions"][0]["recommendation"] == "candidate_suppress_secondary"
    assert payload["decisions"][0]["secondary_lane"] == "volume_vacuum"
    assert payload["decisions"][1]["recommendation"] == "keep_separate"


def test_main_writes_consolidation_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_lane_consolidation_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    overlap = out_dir / "overlap.json"
    watchboard.write_text(json.dumps({"summary": {"top_lane": "spread_stress"}, "top_event": {}}), encoding="utf-8")
    overlap.write_text(
        json.dumps(
            {
                "summary": {"top_overlap_pair": "spread_stress::volume_vacuum"},
                "lane_stats": [{"lane": "spread_stress", "active_count": 2, "top_count": 1}, {"lane": "volume_vacuum", "active_count": 2, "top_count": 0}],
                "strongest_overlaps": [{"lane_a": "spread_stress", "lane_b": "volume_vacuum", "jaccard": 0.9, "coactive_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "consolidation.json"
    out_md = out_dir / "consolidation.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--watchboard-json", str(watchboard), "--overlap-json", str(overlap), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert elc.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_lane_consolidation"
    assert out_md.exists()
