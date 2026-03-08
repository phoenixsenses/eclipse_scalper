from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import volume_vacuum_state as vvs


def test_build_state_payload_marks_severe() -> None:
    payload = vvs.build_state_payload(
        alert_payload={
            "lane": "volume_vacuum",
            "symbol": "ETHUSDT",
            "bucket_sec": 5,
            "summary": {
                "rows_total": 100,
                "tagged_count": 5,
                "tagged_rate": 0.05,
                "recent_alert_count": 8,
                "high_count": 2,
                "medium_count": 6,
                "avg_trade_intensity_tagged": 20.0,
                "avg_spread_tagged": 0.0002,
            },
            "alerts": [{"ts_ms": 1700000000000}],
        },
        source_json="reports/VOLUME_VACUUM_ALERTS_REAL.json",
        out_json="reports/VOLUME_VACUUM_STATE.json",
        out_md="reports/VOLUME_VACUUM_STATE.md",
        now_ts_ms=1700000005000,
    )
    assert payload["lane"] == "volume_vacuum"
    assert payload["state"]["level"] == "severe"
    assert payload["recommended_action"] == "show_caution"


def test_main_writes_json_and_md() -> None:
    out_dir = Path("localtests/test_volume_vacuum_state")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = out_dir / "alerts.json"
    alerts_json.write_text(
        json.dumps(
            {
                "lane": "volume_vacuum",
                "symbol": "ETHUSDT",
                "bucket_sec": 5,
                "summary": {
                    "rows_total": 100,
                    "tagged_count": 1,
                    "tagged_rate": 0.01,
                    "recent_alert_count": 1,
                    "high_count": 0,
                    "medium_count": 1,
                    "avg_trade_intensity_tagged": 80.0,
                    "avg_spread_tagged": 0.0001,
                },
                "alerts": [{"ts_ms": 1000}],
            }
        ),
        encoding="utf-8",
    )
    out_json = out_dir / "state.json"
    out_md = out_dir / "state.md"
    rc = vvs.main(["--alerts-json", str(alerts_json), "--out-json", str(out_json), "--out-md", str(out_md), "--now-ts-ms", "50000"])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lane"] == "volume_vacuum"
    assert payload["run_summary"]["run_type"] == "volume_vacuum_state"
    assert out_md.exists()
