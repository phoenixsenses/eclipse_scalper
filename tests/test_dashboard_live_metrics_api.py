from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app
from dashboard.backend import data_sources as ds


def test_live_metrics_endpoint_contract(monkeypatch) -> None:
    payload = {
        "ts_utc": "2026-03-05T00:00:00+00:00",
        "runtime": {
            "collector": {"alive": True, "trades_per_sec_60s": 12.3},
            "database": {"size_bytes": 1234},
            "data_freshness": {"status": "LIVE", "seconds_since_last_trade": 1.2},
            "system": {},
        },
        "scoreboard": {
            "paper_trading": True,
            "orders_total": 10,
            "fills_total": 4,
            "blocked_total": 2,
            "blocked_by_reason": {"no_match": 2},
        },
        "pnl_strip": {"today": 1.0, "h24": 2.0, "d7": 3.0, "sample": 4},
        "fill_quality": {"avg_delay_ms": 12.5, "avg_adverse_bps": 0.3, "with_delay": 4, "with_adverse": 4},
        "tail_kpis": {"window_lines": 80, "order_count": 20, "fill_count": 8, "blocked_count": 3, "fill_per_order_pct": 40.0},
        "blocked_reasons": [{"reason": "no_match", "count": 2}],
        "last_fills": [],
        "alerts": {
            "any_alert": False,
            "trade_age_alert": False,
            "fill_flatline_alert": False,
            "trade_age_sec": 1.2,
            "fill_age_min": 0.5,
            "config": {"trade_age_alert_sec": 10, "fill_flatline_alert_min": 15},
        },
        "trends": {"trades_per_sec": [1.0, 2.0], "fills_tail": [0.0, 1.0]},
        "paper_file": "paper_trades.jsonl",
    }
    monkeypatch.setattr("dashboard.backend.app.read_live_metrics", lambda: payload)
    client = TestClient(app)
    resp = client.get("/api/live/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ts_utc"] == "2026-03-05T00:00:00+00:00"
    assert body["runtime"]["collector"]["alive"] is True
    assert body["tail_kpis"]["fill_count"] == 8
    assert body["alerts"]["config"]["trade_age_alert_sec"] == 10


def test_read_live_metrics_basic_sanity(monkeypatch) -> None:
    monkeypatch.setattr(ds, "_live_metrics_cache", {})
    monkeypatch.setattr(ds, "_live_metrics_cache_ts", 0.0)
    ds._live_trades_series.clear()
    ds._live_fills_series.clear()

    monkeypatch.setattr(ds, "list_log_files", lambda: [{"name": "paper_trades.jsonl"}])
    monkeypatch.setattr(
        ds,
        "read_runtime_status",
        lambda: {
            "collector": {"alive": True, "trades_per_sec_60s": 25.0},
            "database": {"size_bytes": 1},
            "data_freshness": {"status": "LIVE", "seconds_since_last_trade": 2.0},
            "system": {},
        },
    )
    monkeypatch.setattr(
        ds,
        "read_scoreboard",
        lambda: {
            "paper_trading": True,
            "orders_total": 10,
            "fills_total": 5,
            "blocked_total": 3,
            "blocked_by_reason": {"no_match": 2, "regime_mismatch": 1},
            "last_fill_ts": "2026-03-05T00:00:00+00:00",
        },
    )

    short_tail = [
        "ORDER SUBMITTED symbol=ETHUSDT",
        "ENTRY_DECISION reason=no_match blocked=1",
        "fill_price=2100.1 qty=0.01 pnl_bps=0.5",
    ]
    long_tail = [
        '{"type":"filled","ts_utc":"2026-03-05T00:00:00+00:00","symbol":"ETHUSDT","side":"buy","fill_price":2100.1,"qty":0.01,"pnl_bps":0.5,"fill_delay_ms":120}',
        '{"type":"filled","ts_utc":"2026-03-05T00:01:00+00:00","symbol":"ETHUSDT","side":"sell","fill_price":2101.2,"qty":0.02,"pnl_bps":-0.2,"fill_delay_ms":80}',
    ]

    monkeypatch.setattr(
        ds,
        "tail_log_file",
        lambda filename, limit=200: long_tail if limit >= 2000 else short_tail,
    )

    out = ds.read_live_metrics()
    assert "runtime" in out
    assert "scoreboard" in out
    assert out["tail_kpis"]["order_count"] >= 1
    assert out["tail_kpis"]["fill_count"] >= 1
    assert len(out["blocked_reasons"]) >= 1
    assert out["pnl_strip"]["sample"] >= 1
    assert len(out["trends"]["trades_per_sec"]) >= 1


def test_live_tests_status_endpoint_contract(monkeypatch) -> None:
    payload = {
        "ts_utc": "2026-03-05T01:02:03+00:00",
        "state": "passed",
        "stage": "complete",
        "message": "all checks passed",
        "strict_mode": False,
        "backend_ok": True,
        "frontend_typecheck_ok": True,
        "frontend_smoke_ok": True,
        "frontend_smoke_skipped": False,
        "pid": 12345,
        "run_command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1",
        "log_path": "logs/live_monitor_tests.log",
        "status_path": "runtime/live_monitor_tests_status.json",
        "status_age_sec": 2.5,
        "log_tail": ["line1", "line2"],
    }
    monkeypatch.setattr("dashboard.backend.app.read_live_monitor_tests_status", lambda limit=80: payload)
    client = TestClient(app)
    resp = client.get("/api/live/tests/status?limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "passed"
    assert body["backend_ok"] is True
    assert body["frontend_smoke_ok"] is True
    assert len(body["log_tail"]) == 2


def test_live_tests_run_endpoint_starts_process(monkeypatch) -> None:
    script = Path(__file__).resolve()
    runner_log = Path(__file__).resolve().parents[1] / "runtime" / "live_monitor_tests_runner_test.log"
    runner_log.parent.mkdir(parents=True, exist_ok=True)

    calls: dict[str, object] = {}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["stdout"] = stdout
        calls["stderr"] = stderr
        return SimpleNamespace(pid=43210)

    monkeypatch.setattr("dashboard.backend.app._LIVE_TEST_SCRIPT", script)
    monkeypatch.setattr("dashboard.backend.app._LIVE_TEST_RUNNER_LOG", runner_log)
    monkeypatch.setattr("dashboard.backend.app.subprocess.Popen", fake_popen)

    client = TestClient(app)
    resp = client.post("/api/live/tests/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["started"] is True
    assert body["pid"] == 43210
    assert "run_live_monitor_tests.ps1" in body["command"]
    assert isinstance(calls.get("cmd"), list)
