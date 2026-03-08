from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import event_lane_persistence_policy as elpp


def test_build_persistence_policy_payload_marks_noisy_lane() -> None:
    history_rows = [
        {"source": "a.json", "top_lane": "spread_stress", "top_event": {"lane": "spread_stress"}},
        {"source": "b.json", "top_lane": "volume_vacuum", "top_event": {"lane": "volume_vacuum"}},
        {"source": "c.json", "top_lane": "spread_stress", "top_event": {"lane": "spread_stress"}},
        {"source": "d.json", "top_lane": "volume_vacuum", "top_event": {"lane": "volume_vacuum"}},
    ]
    payload = elpp.build_persistence_policy_payload(
        history_rows=history_rows,
        history_path="reports/HISTORY.jsonl",
        last_n=4,
        out_json="reports/PERSIST.json",
        out_md="reports/PERSIST.md",
    )
    assert payload["run_summary"]["run_type"] == "event_lane_persistence_policy"
    assert payload["summary"]["flip_count"] == 3
    assert payload["summary"]["noisy_lane_count"] >= 1
    by_lane = {row["lane"]: row for row in payload["lanes"]}
    assert by_lane["spread_stress"]["is_noisy"] is True
    assert by_lane["spread_stress"]["recommended_min_persist_snapshots"] == 2


def test_main_writes_files(monkeypatch) -> None:
    out_dir = Path("localtests/test_event_lane_persistence_policy")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = out_dir / "history.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"source": "a.json", "top_lane": "spread_stress", "top_event": {"lane": "spread_stress"}}),
                json.dumps({"source": "b.json", "top_lane": "volume_vacuum", "top_event": {"lane": "volume_vacuum"}}),
                json.dumps({"source": "c.json", "top_lane": "spread_stress", "top_event": {"lane": "spread_stress"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_json = out_dir / "policy.json"
    out_md = out_dir / "policy.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--history-jsonl", str(history), "--last-n", "3", "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert elpp.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "event_lane_persistence_policy"
    assert payload["summary"]["used_rows"] == 3
    assert out_md.exists()
