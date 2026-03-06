from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import spread_stress_state as sss


def test_build_state_payload_marks_elevated() -> None:
    payload = sss.build_state_payload(
        alert_payload={
            "symbol": "ETHUSDT",
            "bucket_sec": 5,
            "summary": {
                "rows_total": 100,
                "tagged_count": 5,
                "tagged_rate": 0.05,
                "recent_alert_count": 4,
                "high_count": 0,
                "medium_count": 4,
                "avg_spread_tagged": 0.00015,
                "avg_trade_intensity_tagged": 500.0,
            },
            "alerts": [{"ts_ms": 1700000000000}],
        },
        source_json="reports/SPREAD_STRESS_ALERTS_REAL.json",
        out_json="reports/SPREAD_STRESS_STATE.json",
        out_md="reports/SPREAD_STRESS_STATE.md",
        now_ts_ms=1700000005000,
    )
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "reduce_passive_aggression"
    assert payload["state"]["freshness"]["status"] == "fresh"


def test_main_writes_json_and_md() -> None:
    out_dir = Path("localtests/test_spread_stress_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = out_dir / "alerts.json"
    alerts_json.write_text(
        json.dumps(
            {
                "symbol": "ETHUSDT",
                "bucket_sec": 5,
                "summary": {
                    "rows_total": 100,
                    "tagged_count": 1,
                    "tagged_rate": 0.01,
                    "recent_alert_count": 1,
                    "high_count": 0,
                    "medium_count": 1,
                    "avg_spread_tagged": 0.00008,
                    "avg_trade_intensity_tagged": 600.0,
                },
                "alerts": [{"ts_ms": 1000}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    rc = sss.main(["--alerts-json", str(alerts_json), "--out-json", str(out_json), "--out-md", str(out_md), "--now-ts-ms", "50000"])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "spread_stress_state"
    assert out_md.exists()
