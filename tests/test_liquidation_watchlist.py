from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import liquidation_watchlist as lw


def test_build_watchlist_payload_ranks_symbols(monkeypatch) -> None:
    def fake_alert_payload(**kwargs):
        symbol = kwargs["symbol"]
        if symbol == "ETHUSDT":
            return {
                "symbol": symbol,
                "rule": "high_liq_reversal_regime",
                "bucket_sec": 5,
                "summary": {
                    "rows_total": 100,
                    "tagged_count": 5,
                    "tagged_rate": 0.05,
                    "recent_alert_count": 6,
                    "max_consecutive_tagged": 3,
                    "max_liq_rate_recent": 7.0,
                    "side_bias_counts": {"LONG": 4},
                    "severity_counts": {"high": 1, "medium": 3},
                },
                "alerts": [{"ts_ms": 1700000000000}],
            }
        return {
            "symbol": symbol,
            "rule": "high_liq_reversal_regime",
            "bucket_sec": 5,
            "summary": {
                "rows_total": 100,
                "tagged_count": 1,
                "tagged_rate": 0.01,
                "recent_alert_count": 1,
                "max_consecutive_tagged": 1,
                "max_liq_rate_recent": 1.0,
                "side_bias_counts": {"SHORT": 1},
                "severity_counts": {"low": 1},
            },
            "alerts": [{"ts_ms": 1}],
        }

    def fake_state_payload(*, alert_payload, source_json, out_json, out_md, now_ts_ms=None):
        symbol = alert_payload["symbol"]
        if symbol == "ETHUSDT":
            return {
                "symbol": symbol,
                "state": {
                    "level": "elevated",
                    "primary_side_bias": "LONG",
                    "dominant_severity": "medium",
                    "freshness": {"status": "fresh", "age_sec": 5.0},
                },
                "recommended_action": "show_caution",
                "dashboard_summary": "ETH elevated",
                "card": {
                    "recent_alert_count": 6,
                    "max_liq_rate_recent": 7.0,
                },
                "summary_snapshot": {"tagged_rate": 0.05},
            }
        return {
            "symbol": symbol,
            "state": {
                "level": "quiet",
                "primary_side_bias": "SHORT",
                "dominant_severity": "low",
                "freshness": {"status": "stale", "age_sec": 100.0},
            },
            "recommended_action": "monitor_only",
            "dashboard_summary": "BTC quiet",
            "card": {
                "recent_alert_count": 1,
                "max_liq_rate_recent": 1.0,
            },
            "summary_snapshot": {"tagged_rate": 0.01},
        }

    monkeypatch.setattr(lw, "build_alert_payload", fake_alert_payload)
    monkeypatch.setattr(lw, "build_state_payload", fake_state_payload)
    payload = lw.build_watchlist_payload(
        db="data/microstructure.db",
        symbols=["BTCUSDT", "ETHUSDT"],
        lookback_min=240,
        bucket_sec=5,
        rule="high_liq_reversal_regime",
        recent_limit=20,
        min_liq_rate=0.0,
        top_n=2,
        out_json="reports/LIQUIDATION_WATCHLIST.json",
        out_md="reports/LIQUIDATION_WATCHLIST.md",
    )
    assert payload["summary"]["symbol_count"] == 2
    assert payload["summary"]["top_symbol"] == "ETHUSDT"
    assert payload["rows"][0]["symbol"] == "ETHUSDT"
    assert payload["rows"][0]["recommended_action"] == "show_caution"


def test_main_writes_watchlist_files(monkeypatch) -> None:
    monkeypatch.setattr(
        lw,
        "build_watchlist_payload",
        lambda **kwargs: {
            "rule": "high_liq_reversal_regime",
            "lookback_min": 240,
            "bucket_sec": 5,
            "recent_limit": 20,
            "min_liq_rate": 0.0,
            "summary": {"symbol_count": 1, "top_n": 1, "state_counts": {"elevated": 1}, "top_symbol": "ETHUSDT"},
            "rows": [
                {
                    "symbol": "ETHUSDT",
                    "state_level": "elevated",
                    "freshness_status": "fresh",
                    "recommended_action": "show_caution",
                    "primary_side_bias": "LONG",
                    "dominant_severity": "medium",
                    "recent_alert_count": 3,
                    "max_liq_rate_recent": 5.2,
                    "tagged_rate": 0.04,
                    "age_sec": 4.0,
                    "dashboard_summary": "ETH summary",
                    "priority_score": 120.0,
                }
            ],
            "run_summary": {
                "version": "v1",
                "run_type": "liquidation_watchlist",
                "inputs": {"symbols": ["ETHUSDT"]},
                "metrics": {"symbol_count": 1},
                "artifacts": {"json": "reports/x.json", "md": "reports/x.md"},
            },
        },
    )
    out_dir = Path("localtests/test_liquidation_watchlist")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "watchlist.json"
    out_md = out_dir / "watchlist.md"
    rc = lw.main(["--symbols", "ETHUSDT", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "liquidation_watchlist"
    assert out_md.exists()
