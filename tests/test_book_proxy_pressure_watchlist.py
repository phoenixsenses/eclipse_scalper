from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import book_proxy_pressure_watchlist as bpw


def test_build_watchlist_payload_ranks_symbols(monkeypatch) -> None:
    def fake_alert_payload(**kwargs):
        return {
            "lane": "book_proxy_pressure",
            "symbol": kwargs["symbol"],
            "bucket_sec": 5,
            "summary": {},
            "alerts": [],
        }

    def fake_state_payload(*, alert_payload, source_json, out_json, out_md, now_ts_ms=None):
        symbol = alert_payload["symbol"]
        if symbol == "ETHUSDT":
            return {
                "lane": "book_proxy_pressure",
                "symbol": symbol,
                "state": {"level": "severe", "primary_side_bias": "SHORT", "freshness": {"status": "fresh", "age_sec": 3.0}},
                "recommended_action": "show_caution",
                "dashboard_summary": "ETH severe book proxy pressure",
                "card": {
                    "recent_alert_count": 6,
                    "high_count": 2,
                    "medium_count": 4,
                    "avg_abs_imbalance_tagged": 0.95,
                    "avg_trade_intensity_tagged": 400.0,
                    "avg_spread_tagged": 0.0008,
                },
                "summary_snapshot": {},
            }
        return {
            "lane": "book_proxy_pressure",
            "symbol": symbol,
            "state": {"level": "elevated", "primary_side_bias": "LONG", "freshness": {"status": "stale", "age_sec": 100.0}},
            "recommended_action": "monitor_only",
            "dashboard_summary": "BTC stale book proxy pressure",
            "card": {
                "recent_alert_count": 3,
                "high_count": 0,
                "medium_count": 3,
                "avg_abs_imbalance_tagged": 0.75,
                "avg_trade_intensity_tagged": 500.0,
                "avg_spread_tagged": 0.0005,
            },
            "summary_snapshot": {},
        }

    monkeypatch.setattr(bpw, "build_alert_payload", fake_alert_payload)
    monkeypatch.setattr(bpw, "build_state_payload", fake_state_payload)
    payload = bpw.build_watchlist_payload(
        db="data/microstructure.db",
        symbols=["BTCUSDT", "ETHUSDT"],
        lookback_min=240,
        bucket_sec=5,
        recent_limit=20,
        top_n=2,
        out_json="reports/BOOK_PROXY_PRESSURE_WATCHLIST.json",
        out_md="reports/BOOK_PROXY_PRESSURE_WATCHLIST.md",
    )
    assert payload["lane"] == "book_proxy_pressure"
    assert payload["summary"]["top_symbol"] == "ETHUSDT"
    assert payload["top_summary"]["recommended_action"] == "show_caution"
    assert payload["banner"]["top_symbol"] == "ETHUSDT"


def test_main_writes_watchlist_files(monkeypatch) -> None:
    monkeypatch.setattr(
        bpw,
        "build_watchlist_payload",
        lambda **kwargs: {
            "lane": "book_proxy_pressure",
            "lookback_min": 240,
            "bucket_sec": 5,
            "recent_limit": 20,
            "summary": {"symbol_count": 1, "top_n": 1, "state_counts": {"severe": 1}, "top_symbol": "ETHUSDT"},
            "top_summary": {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "dashboard_summary": "ETH severe book proxy pressure",
            },
            "banner": {
                "headline": "Book proxy pressure watchlist top=ETHUSDT level=severe freshness=fresh action=show_caution",
                "recommended_action": "show_caution",
                "top_symbol": "ETHUSDT",
                "top_state_level": "severe",
                "top_freshness_status": "fresh",
                "severe_count": 1,
                "elevated_count": 0,
                "quiet_count": 0,
            },
            "rows": [
                {
                    "symbol": "ETHUSDT",
                    "state_level": "severe",
                    "freshness_status": "fresh",
                    "recommended_action": "show_caution",
                    "primary_side_bias": "SHORT",
                    "recent_alert_count": 6,
                    "high_count": 2,
                    "medium_count": 4,
                    "avg_abs_imbalance_tagged": 0.95,
                    "avg_trade_intensity_tagged": 400.0,
                    "avg_spread_tagged": 0.0008,
                    "age_sec": 3.0,
                    "dashboard_summary": "ETH severe book proxy pressure",
                    "priority_score": 247.0,
                }
            ],
            "run_summary": {
                "version": "v1",
                "run_type": "book_proxy_pressure_watchlist",
                "inputs": {"symbols": ["ETHUSDT"]},
                "metrics": {"symbol_count": 1},
                "artifacts": {"json": "reports/x.json", "md": "reports/x.md"},
            },
        },
    )
    out_dir = Path("localtests/test_book_proxy_pressure_watchlist")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "watchlist.json"
    out_md = out_dir / "watchlist.md"
    rc = bpw.main(["--symbols", "ETHUSDT", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lane"] == "book_proxy_pressure"
    assert payload["run_summary"]["run_type"] == "book_proxy_pressure_watchlist"
    assert out_md.exists()
