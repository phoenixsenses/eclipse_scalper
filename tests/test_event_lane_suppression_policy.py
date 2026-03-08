from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_lane_suppression_policy as elsp


def test_build_suppression_policy_payload() -> None:
    out_dir = Path("localtests/test_event_lane_suppression_policy")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    consolidation = out_dir / "consolidation.json"
    watchboard.write_text(
        json.dumps(
            {
                "summary": {"top_lane": "spread_stress"},
                "lanes": [
                    {"lane": "spread_stress", "level": "severe", "recommended_action": "reduce_passive_aggression"},
                    {"lane": "volume_vacuum", "level": "severe", "recommended_action": "show_caution"},
                    {"lane": "liquidation", "level": "severe", "recommended_action": "monitor_only"},
                ],
            }
        ),
        encoding="utf-8",
    )
    consolidation.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "lane_a": "spread_stress",
                        "lane_b": "volume_vacuum",
                        "secondary_lane": "volume_vacuum",
                        "recommendation": "candidate_suppress_secondary",
                        "reason": "overlap",
                    },
                    {
                        "lane_a": "spread_stress",
                        "lane_b": "liquidation",
                        "secondary_lane": "liquidation",
                        "recommendation": "candidate_suppress_secondary",
                        "reason": "overlap",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = elsp.build_suppression_policy_payload(
        watchboard_json=str(watchboard),
        consolidation_json=str(consolidation),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["top_lane"] == "spread_stress"
    assert payload["summary"]["rule_count"] == 2
    assert sorted(payload["summary"]["suppressed_lanes"]) == ["liquidation", "volume_vacuum"]
    by_lane = {row["secondary_lane"]: row for row in payload["rules"]}
    assert by_lane["volume_vacuum"]["display_mode"] == "degrade"
    assert by_lane["liquidation"]["display_mode"] == "hide"


def test_main_writes_policy_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_lane_suppression_policy_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    consolidation = out_dir / "consolidation.json"
    watchboard.write_text(json.dumps({"summary": {"top_lane": "spread_stress"}, "lanes": []}), encoding="utf-8")
    consolidation.write_text(json.dumps({"decisions": []}), encoding="utf-8")
    out_json = out_dir / "policy.json"
    out_md = out_dir / "policy.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--watchboard-json", str(watchboard), "--consolidation-json", str(consolidation), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert elsp.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_lane_suppression_policy"
    assert out_md.exists()
