"""Proves tools/heartbeat_watchdog.py's canonical logs/health/overall.json
payload is actually understood by execution/health_gate.py -- the real
live-safety consumer -- not just internally self-consistent.

Does not modify execution/health_gate.py; only exercises its existing,
unmodified evaluate_health_gate() against payloads built by
build_canonical_overall().
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.health_gate import GateState, evaluate_health_gate
from tools.heartbeat_watchdog import build_canonical_overall
from tools.native_ws_health_policy import evaluate_policy

NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _iso(seconds_ago: float) -> str:
    return (NOW - timedelta(seconds=seconds_ago)).isoformat()


def _fresh_sources():
    return {
        "agg_trades": {"last_ts_ms": 1, "age_sec": 1.0, "error": None},
        "mark_prices": {"last_ts_ms": 1, "age_sec": 1.0, "error": None},
        "liquidations": {"last_ts_ms": 1, "age_sec": 5.0, "error": None},
    }


def _canonical_for(*, native_status_scenario: str, tmp_path: Path) -> dict:
    if native_status_scenario == "green":
        hb = {
            "connected": True, "last_message_ts_utc": _iso(1.0), "rest_fallback_active": False,
            "rest_last_progress_ts_utc": _iso(1.0), "current_backoff_seconds": 1.0, "last_error": "",
        }
        comp = {"status": "ok", "connected": True, "transport_connected": True}
        sources = _fresh_sources()
        overall = "GREEN"
    elif native_status_scenario == "degraded":
        hb = {
            "connected": False, "last_message_ts_utc": None, "rest_fallback_active": True,
            "rest_last_progress_ts_utc": _iso(1.0), "current_backoff_seconds": 1.0, "last_error": "",
        }
        comp = {"status": "degraded", "connected": False, "transport_connected": False}
        sources = _fresh_sources()
        overall = "YELLOW"
    else:  # red
        hb = {
            "connected": False, "last_message_ts_utc": None, "rest_fallback_active": False,
            "rest_last_progress_ts_utc": None, "current_backoff_seconds": 1.0, "last_error": "",
        }
        comp = {"status": "degraded", "connected": False, "transport_connected": False}
        sources = {
            "agg_trades": {"last_ts_ms": 1, "age_sec": 5000.0, "error": None},
            "mark_prices": {"last_ts_ms": 1, "age_sec": 5000.0, "error": None},
            "liquidations": {"last_ts_ms": 1, "age_sec": 5000.0, "error": None},
        }
        overall = "RED"

    policy = evaluate_policy(
        collector_heartbeat=hb, collector_component=comp, collector_process_alive=True,
        source_freshness=sources, now=NOW,
    )
    # collector.json-shaped component, as read by heartbeat_watchdog from disk
    collector_component = {
        "status": comp["status"], "connected": comp["connected"], "transport_connected": comp["transport_connected"],
        "progress_lag_sec": 5, "reconnects_last_5m": 0, "errors_last_5m": 0,
    }
    return build_canonical_overall(
        overall=overall,
        issues=policy["reasons"],
        collector_component=collector_component,
        bookticker_component={"status": "ok", "connected": True},
        native_ws_policy=policy,
        runtime_mode="paper",
        now_iso=NOW.isoformat(),
        log_health=tmp_path,
    )


def test_health_gate_blocks_on_canonical_native_red(tmp_path):
    health_obj = _canonical_for(native_status_scenario="red", tmp_path=tmp_path)
    gate_state = GateState()
    dec = evaluate_health_gate(health_obj, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is False
    assert health_obj["state"] == "halted"


def test_health_gate_blocks_on_canonical_native_degraded(tmp_path):
    health_obj = _canonical_for(native_status_scenario="degraded", tmp_path=tmp_path)
    gate_state = GateState()
    dec = evaluate_health_gate(health_obj, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is False
    assert health_obj["state"] == "degraded"


def test_health_gate_allows_on_canonical_native_green(tmp_path):
    health_obj = _canonical_for(native_status_scenario="green", tmp_path=tmp_path)
    gate_state = GateState()
    dec = evaluate_health_gate(health_obj, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is True
    assert health_obj["state"] == "ok"


def test_health_gate_does_not_accept_collector_only_healthy_when_state_is_halted(tmp_path):
    """Even if components.collector reports connected=True (a stale/partial
    view), a canonical RED (state=halted) must still block -- health_gate.py
    checks top-level state first, not just the collector sub-component."""
    health_obj = _canonical_for(native_status_scenario="red", tmp_path=tmp_path)
    # Simulate a stale collector.json snapshot that still says "connected: True"
    health_obj["components"]["collector"]["connected"] = True
    health_obj["components"]["collector"]["status"] = "ok"
    gate_state = GateState()
    dec = evaluate_health_gate(health_obj, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is False, "top-level state=halted must block regardless of a stale-looking collector component"


def test_health_gate_rejects_missing_health_object():
    gate_state = GateState()
    dec = evaluate_health_gate(None, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is False
    assert dec.reason == "health_missing"


def test_health_gate_rejects_stale_canonical_payload(tmp_path):
    health_obj = _canonical_for(native_status_scenario="green", tmp_path=tmp_path)
    # ts_utc from 20s ago -- beyond the default 15s max_health_staleness_sec.
    health_obj["ts_utc"] = _iso(20.0)
    gate_state = GateState()
    dec = evaluate_health_gate(health_obj, gate_state, now_ts=NOW.timestamp())
    assert dec.allow is False
    assert dec.reason == "health_stale"
