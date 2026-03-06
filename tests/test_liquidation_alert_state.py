from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import liquidation_alert_state as las


def test_build_state_payload_marks_severe_when_alerts_are_dense() -> None:
    payload = las.build_state_payload(
        alert_payload={
            "symbol": "ETHUSDT",
            "rule": "high_liq_reversal_regime",
            "summary": {
                "rows_total": 100,
                "tagged_count": 8,
                "tagged_rate": 0.08,
                "recent_alert_count": 9,
                "max_consecutive_tagged": 3,
                "max_liq_rate_recent": 9.2,
                "side_bias_counts": {"LONG": 6, "SHORT": 3},
                "severity_counts": {"high": 2, "medium": 4, "low": 3},
            },
            "alerts": [{"ts_ms": 123}],
        },
        source_json="reports/LIQUIDATION_REGIME_ALERTS_REAL.json",
        out_json="reports/LIQUIDATION_ALERT_STATE.json",
        out_md="reports/LIQUIDATION_ALERT_STATE.md",
    )
    assert payload["state"]["level"] == "severe"
    assert payload["state"]["primary_side_bias"] == "LONG"
    assert payload["state"]["dominant_severity"] == "medium"
    assert payload["card"]["latest_alert_ts_ms"] == 123


def test_main_writes_json_and_md() -> None:
    out_dir = Path("localtests/test_liquidation_alert_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = out_dir / "alerts.json"
    alerts_json.write_text(
        json.dumps(
            {
                "symbol": "ETHUSDT",
                "rule": "high_liq_reversal_regime",
                "summary": {
                    "rows_total": 50,
                    "tagged_count": 2,
                    "tagged_rate": 0.04,
                    "recent_alert_count": 3,
                    "max_consecutive_tagged": 2,
                    "max_liq_rate_recent": 5.2,
                    "side_bias_counts": {"SHORT": 2},
                    "severity_counts": {"medium": 2},
                },
                "alerts": [{"ts_ms": 456}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    rc = las.main(["--alerts-json", str(alerts_json), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["state"]["level"] == "elevated"
    assert payload["run_summary"]["run_type"] == "liquidation_alert_state"
    assert out_md.exists()
