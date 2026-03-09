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


def test_last_min_filters_old_events() -> None:
    import time
    now = time.time()
    with tempfile.TemporaryDirectory(prefix="event-gate-") as tmpdir:
        telemetry = Path(tmpdir) / "telemetry.jsonl"
        telemetry.write_text(
            "\n".join([
                # old event (2 hours ago)
                json.dumps({"event": "entry.event_lane_gate", "ts": now - 7200,
                            "data": {"symbol": "ETHUSDT", "decision": "would_block", "blocking_lanes": []}}),
                # recent event
                json.dumps({"event": "entry.event_lane_gate", "ts": now - 30,
                            "data": {"symbol": "ETHUSDT", "decision": "allowed", "blocking_lanes": []}}),
            ]),
            encoding="utf-8",
        )
        # last 5 min: only recent event should appear
        summary = build_summary(str(telemetry), last_min=5)
        assert summary["rows_total"] == 1
        assert summary["allowed_count"] == 1
        assert summary["would_block_count"] == 0


def test_human_print_runs_without_error(capsys) -> None:
    from tools.review_event_lane_gate_shadow import _print_human
    summary = {
        "symbol": "ETHUSDT",
        "last_min": 60.0,
        "rows_total": 10,
        "allowed_count": 8,
        "would_block_count": 2,
        "allowed_rate": 0.8,
        "would_block_rate": 0.2,
        "blocking_lane_counts": {"book_proxy_pressure": 2},
        "latest": {"latest_ts_ms": 1000000, "latest_abs_imbalance": 0.91},
    }
    _print_human(summary)
    out = capsys.readouterr().out
    assert "80.0%" in out
    assert "would_block" in out
    assert "book_proxy_pressure" in out
