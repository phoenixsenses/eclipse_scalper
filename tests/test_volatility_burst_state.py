from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import volatility_burst_state as vbs


def test_build_state_payload_marks_severe() -> None:
    payload = vbs.build_state_payload(
        alert_payload={
            "lane": "volatility_burst",
            "symbol": "ETHUSDT",
            "bucket_sec": 5,
            "summary": {
                "rows_total": 100,
                "tagged_count": 5,
                "tagged_rate": 0.05,
                "recent_alert_count": 4,
                "high_count": 2,
                "medium_count": 2,
                "avg_abs_ret_1_tagged": 0.002,
                "avg_trade_intensity_tagged": 500.0,
                "direction_counts": {"UP": 1, "DOWN": 3, "FLAT": 0},
            },
            "alerts": [{"ts_ms": 1700000000000, "direction": "DOWN"}],
        },
        source_json="reports/VOLATILITY_BURST_ALERTS_REAL.json",
        out_json="reports/VOLATILITY_BURST_STATE.json",
        out_md="reports/VOLATILITY_BURST_STATE.md",
        now_ts_ms=1700000005000,
    )
    assert payload["lane"] == "volatility_burst"
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "escalate_monitoring"
    assert payload["state"]["dominant_direction"] == "DOWN"


def test_main_writes_json_and_md() -> None:
    out_dir = Path("localtests/test_volatility_burst_state")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = out_dir / "alerts.json"
    alerts_json.write_text(
        json.dumps(
            {
                "lane": "volatility_burst",
                "symbol": "ETHUSDT",
                "bucket_sec": 5,
                "summary": {
                    "rows_total": 100,
                    "tagged_count": 1,
                    "tagged_rate": 0.01,
                    "recent_alert_count": 1,
                    "high_count": 0,
                    "medium_count": 1,
                    "avg_abs_ret_1_tagged": 0.0004,
                    "avg_trade_intensity_tagged": 350.0,
                    "direction_counts": {"UP": 1, "DOWN": 0, "FLAT": 0},
                },
                "alerts": [{"ts_ms": 1000, "direction": "UP"}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    rc = vbs.main(["--alerts-json", str(alerts_json), "--out-json", str(out_json), "--out-md", str(out_md), "--now-ts-ms", "50000"])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lane"] == "volatility_burst"
    assert payload["run_summary"]["run_type"] == "volatility_burst_state"
    assert out_md.exists()
