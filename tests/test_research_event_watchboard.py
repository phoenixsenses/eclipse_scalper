from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import research_event_watchboard as rew


def test_build_watchboard_payload_ranks_lanes(monkeypatch) -> None:
    monkeypatch.setattr(
        rew,
        "build_liquidation_watchlist",
        lambda **kwargs: {
            "top_summary": {
                "symbol": "ETHUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "escalate_monitoring",
                "dashboard_summary": "ETH severe liq",
            },
            "banner": {"headline": "Liq top ETH"},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_spread_stress_watchlist",
        lambda **kwargs: {
            "top_summary": {
                "symbol": "BTCUSDT",
                "state_level": "elevated",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "dashboard_summary": "BTC elevated spread",
            },
            "banner": {"headline": "Spread top BTC"},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_return_shock_watchlist",
        lambda **kwargs: {
            "top_summary": {
                "symbol": "BTCUSDT",
                "state_level": "severe",
                "freshness_status": "stale",
                "recommended_action": "monitor_only",
                "dashboard_summary": "BTC severe return shock",
            },
            "banner": {"headline": "Return shock top BTC"},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_volume_vacuum_watchlist",
        lambda **kwargs: {
            "top_summary": {
                "symbol": "ETHUSDT",
                "state_level": "elevated",
                "freshness_status": "fresh",
                "recommended_action": "show_caution",
                "dashboard_summary": "ETH elevated volume vacuum",
            },
            "banner": {"headline": "Volume vacuum top ETH"},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_volatility_burst_watchlist",
        lambda **kwargs: {
            "top_summary": {
                "symbol": "BTCUSDT",
                "state_level": "severe",
                "freshness_status": "fresh",
                "recommended_action": "escalate_monitoring",
                "dashboard_summary": "BTC severe volatility burst",
            },
            "banner": {"headline": "Volatility burst top BTC"},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_fill_toxicity_state",
        lambda **kwargs: {
            "rows": 0,
            "state": {"level": "quiet"},
            "recommended_action": "monitor_only",
            "dashboard_summary": "fill quiet",
            "card": {"headline": "Fill quiet", "rows": 0},
        },
    )
    monkeypatch.setattr(
        rew,
        "build_latency_stress_state",
        lambda **kwargs: {
            "rows": 0,
            "state": {"level": "quiet"},
            "recommended_action": "monitor_only",
            "dashboard_summary": "latency quiet",
            "card": {"headline": "Latency quiet", "rows": 0},
        },
    )
    monkeypatch.setattr(rew, "build_toxicity_report", lambda df: {})
    monkeypatch.setattr(rew, "load_toxicity_rows", lambda path: object())
    monkeypatch.setattr(rew, "compute_execution_diagnostics", lambda df: {})
    monkeypatch.setattr(rew, "load_execution_rows", lambda path: object())
    payload = rew.build_watchboard_payload(
        micro_db="data/microstructure.db",
        trade_source="data/live/papertrades_live.parquet",
        symbols=["ETHUSDT", "BTCUSDT"],
        lookback_min=240,
        bucket_sec=5,
        recent_limit=20,
        top_n=5,
        out_json="reports/RESEARCH_EVENT_WATCHBOARD.json",
        out_md="reports/RESEARCH_EVENT_WATCHBOARD.md",
    )
    assert payload["summary"]["top_lane"] == "liquidation"
    assert payload["top_event"]["recommended_action"] == "escalate_monitoring"
    assert payload["banner"]["top_lane"] == "liquidation"
    assert payload["summary"]["lane_count"] == 7


def test_main_writes_watchboard_files(monkeypatch) -> None:
    monkeypatch.setattr(
        rew,
        "build_watchboard_payload",
        lambda **kwargs: {
            "summary": {"lane_count": 7, "state_counts": {"severe": 3, "quiet": 2, "elevated": 2}, "top_lane": "liquidation"},
            "top_event": {
                "lane": "liquidation",
                "level": "severe",
                "recommended_action": "escalate_monitoring",
                "headline": "Liq top ETH",
                "detail": "ETH severe liq",
            },
            "banner": {
                "headline": "Liq top ETH",
                "recommended_action": "escalate_monitoring",
                "top_lane": "liquidation",
                "top_level": "severe",
            },
            "lanes": [
                {
                    "lane": "liquidation",
                    "level": "severe",
                    "freshness_status": "fresh",
                    "recommended_action": "escalate_monitoring",
                    "headline": "Liq top ETH",
                    "detail": "ETH severe liq",
                    "priority_score": 225.0,
                }
            ],
            "run_summary": {
                "version": "v1",
                "run_type": "research_event_watchboard",
                "inputs": {"symbols": ["ETHUSDT", "BTCUSDT"]},
                "metrics": {"lane_count": 7},
                "artifacts": {"json": "reports/x.json", "md": "reports/x.md"},
            },
        },
    )
    out_dir = Path("localtests/test_research_event_watchboard")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "watchboard.json"
    out_md = out_dir / "watchboard.md"
    monkeypatch.setattr(sys, "argv", ["x", "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rew.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "research_event_watchboard"
    assert out_md.exists()
