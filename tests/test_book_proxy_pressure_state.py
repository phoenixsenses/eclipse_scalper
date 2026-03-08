from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import book_proxy_pressure_state as bps


def test_build_state_payload_marks_severe() -> None:
    payload = bps.build_state_payload(
        alert_payload={
            "lane": "book_proxy_pressure",
            "symbol": "ETHUSDT",
            "bucket_sec": 5,
            "summary": {
                "rows_total": 100,
                "tagged_count": 5,
                "tagged_rate": 0.05,
                "recent_alert_count": 4,
                "high_count": 2,
                "medium_count": 2,
                "avg_abs_imbalance_tagged": 0.9,
                "avg_trade_intensity_tagged": 500.0,
                "avg_spread_tagged": 0.0008,
                "side_bias_counts": {"LONG": 3, "SHORT": 1, "NEUTRAL": 0},
            },
            "alerts": [{"ts_ms": 1700000000000, "side_bias": "LONG"}],
        },
        source_json="reports/BOOK_PROXY_PRESSURE_ALERTS_REAL.json",
        out_json="reports/BOOK_PROXY_PRESSURE_STATE.json",
        out_md="reports/BOOK_PROXY_PRESSURE_STATE.md",
        now_ts_ms=1700000005000,
    )
    assert payload["lane"] == "book_proxy_pressure"
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "show_caution"
    assert payload["state"]["primary_side_bias"] == "LONG"


def test_main_writes_json_and_md() -> None:
    out_dir = Path("localtests/test_book_proxy_pressure_state")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = out_dir / "alerts.json"
    alerts_json.write_text(
        json.dumps(
            {
                "lane": "book_proxy_pressure",
                "symbol": "ETHUSDT",
                "bucket_sec": 5,
                "summary": {
                    "rows_total": 100,
                    "tagged_count": 1,
                    "tagged_rate": 0.01,
                    "recent_alert_count": 1,
                    "high_count": 0,
                    "medium_count": 1,
                    "avg_abs_imbalance_tagged": 0.7,
                    "avg_trade_intensity_tagged": 350.0,
                    "avg_spread_tagged": 0.0005,
                    "side_bias_counts": {"LONG": 1, "SHORT": 0, "NEUTRAL": 0},
                },
                "alerts": [{"ts_ms": 1000, "side_bias": "LONG"}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    rc = bps.main(["--alerts-json", str(alerts_json), "--out-json", str(out_json), "--out-md", str(out_md), "--now-ts-ms", "50000"])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lane"] == "book_proxy_pressure"
    assert payload["run_summary"]["run_type"] == "book_proxy_pressure_state"
    assert out_md.exists()
