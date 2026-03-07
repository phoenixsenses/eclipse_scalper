from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_watchboard_effective as ewe


def test_build_effective_watchboard_payload() -> None:
    out_dir = Path("localtests/test_event_watchboard_effective")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    suppression = out_dir / "suppression.json"
    persistence = out_dir / "persistence.json"
    watchboard.write_text(
        json.dumps(
            {
                "summary": {"top_lane": "spread_stress"},
                "lanes": [
                    {"lane": "spread_stress", "level": "severe", "recommended_action": "reduce_passive_aggression", "priority_score": 225.0},
                    {"lane": "volume_vacuum", "level": "severe", "recommended_action": "show_caution", "priority_score": 225.0},
                    {"lane": "liquidation", "level": "severe", "recommended_action": "monitor_only", "priority_score": 200.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    suppression.write_text(
        json.dumps(
            {
                "rules": [
                    {"secondary_lane": "volume_vacuum", "display_mode": "degrade"},
                    {"secondary_lane": "liquidation", "display_mode": "hide"},
                ]
            }
        ),
        encoding="utf-8",
    )
    persistence.write_text(
        json.dumps(
            {
                "summary": {"noisy_lane_count": 1, "primary_noisy_lane": "spread_stress"},
                "lanes": [
                    {
                        "lane": "spread_stress",
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
    payload = ewe.build_effective_watchboard_payload(
        watchboard_json=str(watchboard),
        suppression_json=str(suppression),
        persistence_json=str(persistence),
        out_json=str(out_dir / "out.json"),
        out_md=str(out_dir / "out.md"),
    )
    assert payload["summary"]["raw_top_lane"] == "spread_stress"
    assert payload["summary"]["effective_top_lane"] == "spread_stress"
    assert payload["summary"]["hidden_lane_count"] == 1
    assert payload["summary"]["noisy_lane_count"] == 1
    by_lane = {row["lane"]: row for row in payload["lanes"]}
    assert by_lane["volume_vacuum"]["effective_display_mode"] == "degrade"
    assert by_lane["liquidation"]["effective_display_mode"] == "hide"
    assert by_lane["spread_stress"]["persistence_recommendation"] == "stabilize_banner"
    assert payload["effective_top_event"]["recommended_min_persist_snapshots"] == 2


def test_main_writes_effective_watchboard(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_watchboard_effective_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchboard = out_dir / "watchboard.json"
    suppression = out_dir / "suppression.json"
    persistence = out_dir / "persistence.json"
    watchboard.write_text(json.dumps({"summary": {"top_lane": "spread_stress"}, "lanes": []}), encoding="utf-8")
    suppression.write_text(json.dumps({"rules": []}), encoding="utf-8")
    persistence.write_text(json.dumps({"summary": {"noisy_lane_count": 0, "primary_noisy_lane": ""}, "lanes": []}), encoding="utf-8")
    out_json = out_dir / "effective.json"
    out_md = out_dir / "effective.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--watchboard-json", str(watchboard), "--suppression-json", str(suppression), "--persistence-json", str(persistence), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert ewe.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_watchboard_effective"
    assert out_md.exists()
