from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.review_event_lane_gate_shadow import build_summary


def test_build_summary_counts_allowed_and_would_block() -> None:
    with tempfile.TemporaryDirectory(prefix="event-gate-") as tmpdir:
        telemetry = Path(tmpdir) / "telemetry.jsonl"
        telemetry.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "entry.event_lane_gate",
                            "symbol": "ETHUSDT",
                            "data": {"symbol": "ETHUSDT", "decision": "allowed", "blocking_lanes": []},
                        }
                    ),
                    json.dumps(
                        {
                            "event": "entry.event_lane_gate",
                            "symbol": "ETHUSDT",
                            "data": {
                                "symbol": "ETHUSDT",
                                "decision": "would_block",
                                "blocking_lanes": ["book_proxy_pressure"],
                            },
                        }
                    ),
                    json.dumps({"event": "other", "data": {}}),
                ]
            ),
            encoding="utf-8",
        )

        summary = build_summary(str(telemetry), symbol="ETHUSDT")
        assert summary["rows_total"] == 2
        assert summary["allowed_count"] == 1
        assert summary["would_block_count"] == 1
        assert summary["blocking_lane_counts"]["book_proxy_pressure"] == 1


def test_build_summary_handles_missing_file() -> None:
    summary = build_summary("does-not-exist.jsonl")
    assert summary["rows_total"] == 0
    assert summary["allowed_count"] == 0
    assert summary["would_block_count"] == 0
