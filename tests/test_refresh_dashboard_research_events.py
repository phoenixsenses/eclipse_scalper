from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import refresh_dashboard_research_events as rdr


def test_build_refresh_payload_writes_real_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(rdr, "build_liquidation_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_liquidation_alert_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "liq_state"}})
    monkeypatch.setattr(rdr, "build_liquidation_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "liq_watch"}})
    monkeypatch.setattr(rdr, "build_spread_stress_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_spread_stress_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "spread_state"}})
    monkeypatch.setattr(rdr, "build_spread_stress_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "spread_watch"}})
    monkeypatch.setattr(rdr, "build_toxicity_report", lambda rows: {"rows": 0})
    monkeypatch.setattr(rdr, "load_toxicity_rows", lambda path: [])
    monkeypatch.setattr(rdr, "build_fill_toxicity_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "fill_state"}})
    monkeypatch.setattr(rdr, "load_execution_rows", lambda path: [])
    monkeypatch.setattr(rdr, "compute_execution_diagnostics", lambda rows: {"rows": 0})
    monkeypatch.setattr(rdr, "build_latency_stress_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "latency_state"}})
    monkeypatch.setattr(rdr, "build_return_shock_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_return_shock_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "ret_state"}})
    monkeypatch.setattr(rdr, "build_return_shock_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "ret_watch"}})
    monkeypatch.setattr(rdr, "build_volume_vacuum_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_volume_vacuum_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "vac_state"}})
    monkeypatch.setattr(rdr, "build_volume_vacuum_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "vac_watch"}})
    monkeypatch.setattr(rdr, "build_volatility_burst_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_volatility_burst_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "vol_state"}})
    monkeypatch.setattr(rdr, "build_volatility_burst_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "vol_watch"}})
    monkeypatch.setattr(rdr, "build_book_proxy_pressure_alerts", lambda **kwargs: {"summary": {}, "alerts": []})
    monkeypatch.setattr(rdr, "build_book_proxy_pressure_state", lambda **kwargs: {"state": {"level": "quiet"}, "run_summary": {"run_type": "book_state"}})
    monkeypatch.setattr(rdr, "build_book_proxy_pressure_watchlist", lambda **kwargs: {"rows": [], "run_summary": {"run_type": "book_watch"}})
    monkeypatch.setattr(
        rdr,
        "build_watchboard_payload",
        lambda **kwargs: {"summary": {"top_lane": "liquidation"}, "lanes": [], "run_summary": {"run_type": "watchboard"}},
    )

    out_dir = Path("localtests/test_refresh_dashboard_research_events")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = rdr.build_refresh_payload(
        micro_db="data/microstructure.db",
        trade_source="data/live/papertrades_live.parquet",
        primary_symbol="ETHUSDT",
        symbols=["ETHUSDT", "BTCUSDT"],
        lookback_min=240,
        bucket_sec=5,
        recent_limit=20,
        top_n=5,
        reports_dir=str(out_dir),
    )

    assert payload["summary"]["artifact_count"] == 15
    assert payload["summary"]["watchboard_top_lane"] == "liquidation"
    assert (out_dir / "RESEARCH_EVENT_WATCHBOARD_REAL.json").exists()
    assert (out_dir / "LIQUIDATION_ALERT_STATE_REAL.json").exists()
    assert (out_dir / "SPREAD_STRESS_WATCHLIST_REAL.json").exists()
    assert (out_dir / "BOOK_PROXY_PRESSURE_WATCHLIST_REAL.json").exists()


def test_main_writes_summary_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        rdr,
        "build_refresh_payload",
        lambda **kwargs: {
            "summary": {"watchboard_top_lane": "liquidation", "artifact_count": 15},
            "artifacts": {"RESEARCH_EVENT_WATCHBOARD_REAL_json": "reports/RESEARCH_EVENT_WATCHBOARD_REAL.json"},
            "run_summary": {"run_type": "refresh_dashboard_research_events"},
        },
    )
    out_dir = Path("localtests/test_refresh_dashboard_research_events_main")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "refresh.json"
    out_md = out_dir / "refresh.md"
    assert rdr.main(["--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    body = json.loads(out_json.read_text(encoding="utf-8"))
    assert body["run_summary"]["run_type"] == "refresh_dashboard_research_events"
    assert out_md.exists()
