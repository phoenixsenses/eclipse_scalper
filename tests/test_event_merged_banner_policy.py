from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_merged_banner_policy as embp


def test_build_merged_banner_policy_payload() -> None:
    out_dir = Path("localtests/test_event_merged_banner_policy")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective = out_dir / "effective.json"
    effective.write_text(
        json.dumps(
            {
                "summary": {"degraded_lane_count": 1, "noisy_lane_count": 1},
                "effective_top_event": {"lane": "return_shock", "recommended_action": "escalate_monitoring"},
                "lanes": [
                    {
                        "lane": "return_shock",
                        "level": "severe",
                        "freshness_status": "fresh",
                        "recommended_action": "escalate_monitoring",
                        "effective_display_mode": "keep",
                        "effective_priority_score": 225.0,
                        "headline": "Return shock",
                    },
                    {
                        "lane": "book_proxy_pressure",
                        "level": "severe",
                        "freshness_status": "fresh",
                        "recommended_action": "show_caution",
                        "effective_display_mode": "keep",
                        "effective_priority_score": 225.0,
                        "headline": "Book proxy pressure",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = embp.build_merged_banner_policy_payload(
        effective_json=str(effective),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["banner_mode"] == "merged"
    assert payload["summary"]["focus_lane_count"] == 2
    assert payload["banner"]["focus_lanes"] == ["return_shock", "book_proxy_pressure"]
    assert "multiple_fresh_high_priority_lanes" in payload["banner"]["reasons"]
    assert payload["run_summary"]["run_type"] == "event_merged_banner_policy"


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_merged_banner_policy_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective = out_dir / "effective.json"
    effective.write_text(
        json.dumps(
            {
                "summary": {"degraded_lane_count": 0, "noisy_lane_count": 0},
                "effective_top_event": {"lane": "spread_stress", "recommended_action": "monitor_only"},
                "lanes": [],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "policy.json"
    out_md = out_dir / "policy.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--effective-json", str(effective), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert embp.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_merged_banner_policy"
    assert out_md.exists()
