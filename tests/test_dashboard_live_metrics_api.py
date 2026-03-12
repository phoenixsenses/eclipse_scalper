from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import time

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


def test_paper_run_endpoint_contract(monkeypatch) -> None:
    payload = {
        "ts_utc": "2026-03-10T00:00:00+00:00",
        "session": {
            "status": "running",
            "started_ts": "2026-03-10T00:00:00+00:00",
            "uptime_sec": 300.0,
            "active_symbols": ["ETHUSDT"],
            "telemetry_age_sec": 1.0,
            "telemetry_present": True,
        },
        "startup_manifest": {
            "env_profile": "paper",
            "paper_profile_active": True,
            "paper_execution_mode": "router_blocked",
            "paper_execution_label": "No-fill rehearsal",
            "paper_fill_model": "blocked_no_fill",
            "binance_testnet": True,
            "paper_allow_live_private_api": False,
            "private_api_key_present": False,
            "private_api_secret_present": False,
            "dotenv_source": ".env.paper",
            "_meta": {"path": "logs/paper_startup_manifest.json", "exists": True, "age_sec": 1.0},
        },
        "process_chain": {
            "launcher_present": True,
            "watchdog_present": True,
            "bootstrap_present": True,
            "launcher_pid": 1234,
            "watchdog_pids": [2222],
            "bootstrap_pids": [3333],
            "launcher_started_ts": "2026-03-10T00:00:00+00:00",
            "summary": "launcher pid=1234, watchdog=1, bootstrap=1",
        },
        "entry_state": {
            "allow_entries": True,
            "guard_mode": "GREEN",
            "runtime_gate_degraded": False,
            "runtime_gate_reason": "missing",
            "data_state": "ok",
            "risk_state": "ok",
            "regime_state": "ok",
        },
        "trade_state": {
            "trade_count": 0,
            "last_trade_ts": None,
            "no_trades_yet": True,
            "db_present": True,
            "db_path": "data/paper_trades.db",
            "execution_note": "no fills are expected in router_blocked mode; this run rehearses entry and guard flow only",
        },
        "diagnosis": {
            "code": "signal_not_present",
            "summary": "paper run is healthy but no signal is present",
            "detail": "entries are allowed; recent blockers show signal not present; no fills are expected in router_blocked mode; this run rehearses entry and guard flow only",
            "severity": "info",
            "execution_context": "no fills are expected in router_blocked mode; this run rehearses entry and guard flow only",
        },
        "reason_breakdown": {
            "signal_not_present": 3,
            "gate_blocked": 0,
            "data_degraded": 0,
            "risk_blocked": 0,
            "regime_blocked": 0,
            "unknown": 0,
        },
        "symbols": [
            {
                "symbol": "ETHUSDT",
                "last_blocker_reason": "signal not present",
                "last_blocker_ts": "2026-03-10T00:01:00+00:00",
                "last_signal_ts": None,
                "last_belief_ts": "2026-03-10T00:01:05+00:00",
                "recent_blocked_count": 3,
            }
        ],
    }
    monkeypatch.setattr("dashboard.backend.app.read_paper_run_status", lambda: payload)
    client = TestClient(app)
    resp = client.get("/api/live/paper-run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["status"] == "running"
    assert body["startup_manifest"]["env_profile"] == "paper"
    assert body["startup_manifest"]["paper_execution_label"] == "No-fill rehearsal"
    assert "router_blocked mode" in body["trade_state"]["execution_note"]
    assert body["diagnosis"]["code"] == "signal_not_present"
    assert body["symbols"][0]["symbol"] == "ETHUSDT"


def test_read_paper_run_status_signal_not_present(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {
        "collector": {"alive": True},
        "data_freshness": {"status": "LIVE"},
        "database": {},
        "system": {},
    })
    monkeypatch.setattr(
        ds,
        "_startup_manifest_snapshot",
        lambda: {
            "env_profile": "paper",
            "paper_profile_active": True,
            "paper_execution_mode": "router_blocked",
            "binance_testnet": True,
            "paper_allow_live_private_api": False,
            "_meta": {"exists": True, "age_sec": 1.0, "path": "logs/paper_startup_manifest.json"},
        },
    )
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {"orders_by_symbol": {"ETHUSDT": 1}, "fills_by_symbol": {}})
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {"collector_connected": True})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {
        "launcher_present": True,
        "watchdog_present": True,
        "bootstrap_present": True,
        "launcher_pid": 1,
        "watchdog_pids": [2],
        "bootstrap_pids": [3],
        "launcher_started_ts": "2026-03-10T00:00:00+00:00",
        "summary": "launcher pid=1, watchdog=1, bootstrap=1",
    })
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {
        "trade_count": 0,
        "last_trade_ts": None,
        "no_trades_yet": True,
        "db_present": True,
        "db_path": "data/paper_trades.db",
    })
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: ["ETHUSDT"])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [
        {"ts": now - 5, "event": "execution.belief_state", "data": {"allow_entries": True, "guard_mode": "GREEN", "runtime_gate_degraded": False, "runtime_gate_reason": "missing"}},
        {"ts": now - 1, "event": "entry.blocked", "symbol": "ETHUSDT", "data": {"reason": "signal not present"}},
    ])
    out = ds.read_paper_run_status()
    assert out["session"]["status"] == "running"
    assert out["diagnosis"]["code"] == "signal_not_present"
    assert out["diagnosis"]["execution_context"].startswith("no fills are expected")
    assert out["trade_state"]["execution_note"].startswith("no fills are expected")
    assert out["symbols"][0]["last_blocker_reason"] == "signal not present"


def test_read_paper_run_status_gate_blocked_precedes_signal(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {
        "collector": {"alive": True},
        "data_freshness": {"status": "LIVE"},
        "database": {},
        "system": {},
    })
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {})
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {"collector_connected": True})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {"launcher_present": True, "watchdog_present": True, "bootstrap_present": True, "launcher_pid": 1, "watchdog_pids": [], "bootstrap_pids": [], "launcher_started_ts": None, "summary": "up"})
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {"trade_count": 0, "last_trade_ts": None, "no_trades_yet": True, "db_present": True, "db_path": "data/paper_trades.db"})
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: ["ETHUSDT"])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [
        {"ts": now - 5, "event": "execution.belief_state", "data": {"allow_entries": False, "guard_mode": "YELLOW", "runtime_gate_degraded": True, "runtime_gate_reason": "coverage_gap"}},
        {"ts": now - 1, "event": "entry.blocked", "symbol": "ETHUSDT", "data": {"reason": "signal not present"}},
    ])
    out = ds.read_paper_run_status()
    assert out["diagnosis"]["code"] == "gate_blocked"
    assert out["session"]["status"] == "degraded"


def test_read_paper_run_status_flags_unsafe_startup_contract(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {
        "collector": {"alive": True},
        "data_freshness": {"status": "LIVE"},
        "database": {},
        "system": {},
    })
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {})
    monkeypatch.setattr(
        ds,
        "_startup_manifest_snapshot",
        lambda: {
            "env_profile": "paper",
            "paper_profile_active": True,
            "paper_execution_mode": "router_blocked",
            "paper_execution_label": "No-fill rehearsal",
            "paper_fill_model": "blocked_no_fill",
            "binance_testnet": False,
            "paper_allow_live_private_api": True,
            "private_api_key_present": True,
            "private_api_secret_present": True,
            "_meta": {"exists": True, "age_sec": 1.0},
        },
    )
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {"collector_connected": True})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {
        "launcher_present": True,
        "watchdog_present": True,
        "bootstrap_present": True,
        "launcher_pid": 1,
        "watchdog_pids": [2],
        "bootstrap_pids": [3],
        "launcher_started_ts": "2026-03-10T00:00:00+00:00",
        "summary": "launcher pid=1, watchdog=1, bootstrap=1",
    })
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {
        "trade_count": 0,
        "last_trade_ts": None,
        "no_trades_yet": True,
        "db_present": True,
        "db_path": "data/paper_trades.db",
    })
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: ["ETHUSDT"])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [
        {"ts": now - 2, "event": "execution.belief_state", "data": {"allow_entries": True, "guard_mode": "GREEN", "runtime_gate_degraded": False}},
    ])
    out = ds.read_paper_run_status()
    assert out["session"]["status"] == "degraded"
    assert out["diagnosis"]["code"] == "unsafe_startup_contract"
    assert "live private api" in out["diagnosis"]["summary"].lower()


def test_read_paper_run_status_data_degraded(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {
        "collector": {"alive": False},
        "data_freshness": {"status": "STALE"},
        "database": {},
        "system": {},
    })
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {})
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {"collector_connected": False})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {"launcher_present": True, "watchdog_present": True, "bootstrap_present": True, "launcher_pid": 1, "watchdog_pids": [], "bootstrap_pids": [], "launcher_started_ts": None, "summary": "up"})
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {"trade_count": 0, "last_trade_ts": None, "no_trades_yet": True, "db_present": True, "db_path": "data/paper_trades.db"})
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: ["ETHUSDT"])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [
        {"ts": now - 2, "event": "execution.belief_state", "data": {"allow_entries": True, "guard_mode": "GREEN", "runtime_gate_degraded": False}},
    ])
    out = ds.read_paper_run_status()
    assert out["diagnosis"]["code"] == "data_degraded"
    assert out["entry_state"]["data_state"] == "blocked"


def test_read_paper_run_status_session_down(monkeypatch) -> None:
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {"collector": {}, "data_freshness": {}, "database": {}, "system": {}})
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {})
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {"launcher_present": False, "watchdog_present": False, "bootstrap_present": False, "launcher_pid": None, "watchdog_pids": [], "bootstrap_pids": [], "launcher_started_ts": None, "summary": "process chain unknown"})
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {"trade_count": 0, "last_trade_ts": None, "no_trades_yet": True, "db_present": False, "db_path": "data/paper_trades.db"})
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: [])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [])
    out = ds.read_paper_run_status()
    assert out["session"]["status"] == "down"
    assert out["diagnosis"]["code"] == "session_down"


def test_read_paper_run_status_multi_symbol_prefers_risk_reason(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(ds, "_paper_run_cache", {})
    monkeypatch.setattr(ds, "_paper_run_cache_ts", 0.0)
    monkeypatch.setattr(ds, "read_runtime_status", lambda: {"collector": {"alive": True}, "data_freshness": {"status": "LIVE"}, "database": {}, "system": {}})
    monkeypatch.setattr(ds, "read_scoreboard", lambda: {})
    monkeypatch.setattr(ds, "_health_overall_stats", lambda: {"collector_connected": True})
    monkeypatch.setattr(ds, "_detect_paper_process_chain", lambda: {"launcher_present": True, "watchdog_present": True, "bootstrap_present": True, "launcher_pid": 1, "watchdog_pids": [], "bootstrap_pids": [], "launcher_started_ts": None, "summary": "up"})
    monkeypatch.setattr(ds, "_paper_trade_snapshot", lambda: {"trade_count": 0, "last_trade_ts": None, "no_trades_yet": True, "db_present": True, "db_path": "data/paper_trades.db"})
    monkeypatch.setattr(ds, "_active_symbols_from_env", lambda: ["ETHUSDT", "BTCUSDT"])
    monkeypatch.setattr(ds, "_telemetry_tail", lambda limit=400: [
        {"ts": now - 5, "event": "execution.belief_state", "data": {"allow_entries": True, "guard_mode": "GREEN", "runtime_gate_degraded": False}},
        {"ts": now - 2, "event": "entry.blocked", "symbol": "ETHUSDT", "data": {"reason": "signal not present"}},
        {"ts": now - 1, "event": "entry.blocked", "symbol": "BTCUSDT", "data": {"reason": "risk blocked by guard"}},
    ])
    out = ds.read_paper_run_status()
    assert out["diagnosis"]["code"] == "risk_blocked"
    assert len(out["symbols"]) == 2


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
            "runtime_mode": "paper",
            "paper_execution_mode": "router_blocked",
            "binance_testnet": True,
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


def test_read_scoreboard_uses_startup_manifest(monkeypatch) -> None:
    monkeypatch.setattr(ds, "_safe_json", lambda path: {"orders_total": 1, "paper_trading": False} if "paper_scoreboard.json" in str(path) else {})
    monkeypatch.setattr(
        ds,
        "_startup_manifest_snapshot",
        lambda: {
            "env_profile": "paper",
            "paper_profile_active": True,
            "paper_execution_mode": "router_blocked",
            "paper_execution_label": "No-fill rehearsal",
            "paper_fill_model": "blocked_no_fill",
            "binance_testnet": True,
            "paper_allow_live_private_api": False,
            "_meta": {"exists": True, "age_sec": 1.0, "path": "logs/paper_startup_manifest.json"},
        },
    )
    out = ds.read_scoreboard()
    assert out["paper_trading"] is True
    assert out["runtime_mode"] == "paper"
    assert out["paper_execution_mode"] == "router_blocked"
    assert out["paper_execution_label"] == "No-fill rehearsal"
    assert out["paper_fill_model"] == "blocked_no_fill"
    assert out["binance_testnet"] is True


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
