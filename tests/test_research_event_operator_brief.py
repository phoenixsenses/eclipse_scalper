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
    overlap_path = out_dir / "overlap.json"
    consolidation_path = out_dir / "consolidation.json"
    persistence_path = out_dir / "persistence.json"
    merged_banner_path = out_dir / "merged_banner.json"
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
    overlap_path.write_text(
        json.dumps(
            {
                "summary": {"top_overlap_pair": "liquidation::spread_stress"},
                "strongest_overlaps": [
                    {"lane_a": "liquidation", "lane_b": "spread_stress", "jaccard": 0.75, "coactive_count": 3}
                ],
            }
        ),
        encoding="utf-8",
    )
    consolidation_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "lane_a": "liquidation",
                        "lane_b": "spread_stress",
                        "secondary_lane": "spread_stress",
                        "recommendation": "candidate_suppress_secondary",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    persistence_path.write_text(
        json.dumps(
            {
                "summary": {"noisy_lane_count": 1, "primary_noisy_lane": "liquidation"},
                "lanes": [
                    {
                        "lane": "liquidation",
                        "is_noisy": True,
                        "recommended_min_persist_snapshots": 2,
                        "recommended_cooldown_snapshots": 1,
                        "recommendation": "stabilize_banner",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    merged_banner_path.write_text(
        json.dumps(
            {
                "summary": {"banner_mode": "merged", "focus_lane_count": 2, "focus_lanes": ["liquidation", "return_shock"]},
                "banner": {"headline": "merged headline", "recommended_action": "monitor_only"},
            }
        ),
        encoding="utf-8",
    )

    payload = reob.build_operator_brief_payload(
        watchboard_json=str(watchboard_path),
        trend_json=str(trend_path),
        overlap_json=str(overlap_path),
        consolidation_json=str(consolidation_path),
        persistence_json=str(persistence_path),
        merged_banner_json=str(merged_banner_path),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["top_lane"] == "liquidation"
    assert payload["summary"]["stale_lane_count"] == 1
    assert payload["summary"]["trend"] == "rising"
    assert payload["summary"]["strongest_delta_lane"] == "return_shock"
    assert payload["summary"]["strongest_overlap_pair"] == "liquidation::spread_stress"
    assert payload["summary"]["suppression_candidate_count"] == 1
    assert payload["summary"]["primary_suppression_lane"] == "spread_stress"
    assert payload["summary"]["noisy_lane_count"] == 1
    assert payload["summary"]["primary_noisy_lane"] == "liquidation"
    assert payload["summary"]["merged_banner_mode"] == "merged"
    assert payload["summary"]["merged_focus_lane_count"] == 2
    assert payload["brief"]["strongest_delta"]["trend"] == "rising_fast"
    assert payload["brief"]["strongest_overlap"]["jaccard"] == 0.75
    assert payload["brief"]["primary_suppression"]["secondary_lane"] == "spread_stress"
    assert payload["brief"]["primary_persistence"]["lane"] == "liquidation"
    assert payload["brief"]["merged_banner"]["headline"] == "merged headline"
    assert payload["run_summary"]["run_type"] == "research_event_operator_brief"


def test_build_operator_brief_payload_accepts_watchboard_level_field() -> None:
    out_dir = Path("localtests/test_research_event_operator_brief_level")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard_path = out_dir / "watchboard.json"
    trend_path = out_dir / "trend.json"
    overlap_path = out_dir / "overlap.json"
    consolidation_path = out_dir / "consolidation.json"
    persistence_path = out_dir / "persistence.json"
    merged_banner_path = out_dir / "merged_banner.json"
    watchboard_path.write_text(
        json.dumps(
            {
                "summary": {"top_lane": "spread_stress"},
                "top_event": {"recommended_action": "reduce_passive_aggression", "headline": "top headline"},
                "lanes": [{"lane": "spread_stress", "level": "severe", "freshness_status": "fresh"}],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(json.dumps({"summary": {"trend": "flat"}, "lane_deltas": []}), encoding="utf-8")
    overlap_path.write_text(json.dumps({"summary": {"top_overlap_pair": ""}, "strongest_overlaps": []}), encoding="utf-8")
    consolidation_path.write_text(json.dumps({"decisions": []}), encoding="utf-8")
    persistence_path.write_text(json.dumps({"summary": {"noisy_lane_count": 0, "primary_noisy_lane": ""}, "lanes": []}), encoding="utf-8")
    merged_banner_path.write_text(json.dumps({"summary": {"banner_mode": "single", "focus_lane_count": 1, "focus_lanes": ["spread_stress"]}, "banner": {"headline": "single", "recommended_action": "reduce_passive_aggression"}}), encoding="utf-8")
    payload = reob.build_operator_brief_payload(
        watchboard_json=str(watchboard_path),
        trend_json=str(trend_path),
        overlap_json=str(overlap_path),
        consolidation_json=str(consolidation_path),
        persistence_json=str(persistence_path),
        merged_banner_json=str(merged_banner_path),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["severe_lane_count"] == 1
    assert payload["brief"]["severe_lanes"] == ["spread_stress"]


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_research_event_operator_brief_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard_path = out_dir / "watchboard.json"
    trend_path = out_dir / "trend.json"
    overlap_path = out_dir / "overlap.json"
    consolidation_path = out_dir / "consolidation.json"
    persistence_path = out_dir / "persistence.json"
    merged_banner_path = out_dir / "merged_banner.json"
    watchboard_path.write_text(json.dumps({"summary": {"top_lane": "liquidation"}, "top_event": {}, "lanes": []}), encoding="utf-8")
    trend_path.write_text(json.dumps({"summary": {"trend": "flat"}, "lane_deltas": []}), encoding="utf-8")
    overlap_path.write_text(json.dumps({"summary": {"top_overlap_pair": ""}, "strongest_overlaps": []}), encoding="utf-8")
    consolidation_path.write_text(json.dumps({"decisions": []}), encoding="utf-8")
    persistence_path.write_text(json.dumps({"summary": {"noisy_lane_count": 0, "primary_noisy_lane": ""}, "lanes": []}), encoding="utf-8")
    merged_banner_path.write_text(json.dumps({"summary": {"banner_mode": "single", "focus_lane_count": 1, "focus_lanes": ["liquidation"]}, "banner": {"headline": "single", "recommended_action": "monitor_only"}}), encoding="utf-8")

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
            "--overlap-json",
            str(overlap_path),
            "--consolidation-json",
            str(consolidation_path),
            "--persistence-json",
            str(persistence_path),
            "--merged-banner-json",
            str(merged_banner_path),
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
