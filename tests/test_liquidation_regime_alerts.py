from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import liquidation_regime_alerts as lra


def test_build_alert_payload_filters_recent_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        lra,
        "_load_rows",
        lambda db, symbol, lookback_min, bucket_sec: [{"ts_ms": 1}, {"ts_ms": 2}, {"ts_ms": 3}],
    )
    monkeypatch.setattr(
        lra,
        "_tag_rows",
        lambda rows, rule: [
            {"ts_ms": 1, "tag": "normal", "rule_fired": False, "liq_rate_per_sec": 0.1, "liq_imbalance": 0.0, "spread": 0.05, "trade_intensity": 1.0, "ret_1": 0.0},
            {"ts_ms": 2, "tag": "high_liq_reversal", "rule_fired": True, "liq_rate_per_sec": 4.0, "liq_imbalance": 0.8, "spread": 0.01, "trade_intensity": 8.0, "ret_1": -0.002},
            {"ts_ms": 3, "tag": "high_liq_reversal", "rule_fired": True, "liq_rate_per_sec": 1.0, "liq_imbalance": -0.9, "spread": 0.02, "trade_intensity": 7.0, "ret_1": 0.001},
        ],
    )
    payload = lra.build_alert_payload(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=60,
        bucket_sec=5,
        rule="high_liq_reversal_regime",
        recent_limit=5,
        min_liq_rate=2.0,
        out_json="reports/x.json",
        out_md="reports/x.md",
    )
    assert payload["summary"]["tagged_count"] == 2
    assert payload["summary"]["recent_alert_count"] == 1
    assert payload["alerts"][0]["side_bias"] == "LONG"
    assert payload["alerts"][0]["severity"] == "medium"
    assert payload["summary"]["side_bias_counts"]["LONG"] == 1
    assert payload["summary"]["severity_counts"]["medium"] == 1


def test_main_writes_json_and_md(monkeypatch) -> None:
    monkeypatch.setattr(
        lra,
        "build_alert_payload",
        lambda **kwargs: {
            "symbol": "ETHUSDT",
            "rule": "high_liq_reversal_regime",
            "lookback_min": 60,
            "bucket_sec": 5,
            "recent_limit": 5,
            "min_liq_rate": 2.0,
            "summary": {
                "rows_total": 10,
                "tagged_count": 3,
                "tagged_rate": 0.3,
                "recent_alert_count": 2,
                "max_consecutive_tagged": 2,
                "max_liq_rate_recent": 5.0,
                "side_bias_counts": {"LONG": 2},
                "severity_counts": {"high": 1, "medium": 1},
            },
            "alerts": [
                {"ts_ms": 1, "side_bias": "LONG", "severity": "high", "liq_rate_per_sec": 5.0, "liq_imbalance": 0.8, "spread": 0.01, "trade_intensity": 10.0, "ret_1": -0.002}
            ],
            "run_summary": {
                "version": "v1",
                "run_type": "liquidation_regime_alerts",
                "inputs": {"symbol": "ETHUSDT"},
                "metrics": {"rows_total": 10},
                "artifacts": {"json": "reports/test_liq_alerts/out.json", "md": "reports/test_liq_alerts/out.md"},
            },
        },
    )
    out_dir = Path("reports/test_liq_alerts")
    out_json = out_dir / "out.json"
    out_md = out_dir / "out.md"
    rc = lra.main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "liquidation_regime_alerts"
    assert out_md.exists()
