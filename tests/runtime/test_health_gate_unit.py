"""Unit tests for execution/health_gate.py

Covers:
- Degradation tracking (overall_not_ok, collector_disconnected, data_stale)
- Escalation paths (reconnects, errors, prolonged degradation)
- Halted state with cooldown
- Recovery to ok
- Ingestion probe integration
- write_paper_trader_health (when available)
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from execution.health_gate import (
    GateDecision,
    GateState,
    evaluate_health_gate,
    load_overall_health,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _health_obj(
    state: str = "ok",
    connected: bool = True,
    lag_sec: int = 5,
    reconnects: int = 0,
    errors: int = 0,
    ts_offset: float = 0.0,
) -> dict:
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(time.time() - ts_offset, tz=timezone.utc).isoformat()
    return {
        "state": state,
        "ts_utc": ts,
        "components": {
            "collector": {
                "connected": connected,
                "progress_lag_sec": lag_sec,
                "reconnects_last_5m": reconnects,
                "errors_last_5m": errors,
            }
        },
    }


def _local_tmp_dir(name: str) -> Path:
    base = (Path("localtests") / "runtime" / "health_gate_unit" / f"{name}_{uuid.uuid4().hex[:8]}").resolve()
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# 1. Normal (ok) state
# ---------------------------------------------------------------------------

def test_ok_health():
    gs = GateState()
    d = evaluate_health_gate(_health_obj(), gs)
    assert d.allow is True
    assert d.state == "ok"
    assert d.reason == ""


# ---------------------------------------------------------------------------
# 2. Missing health data
# ---------------------------------------------------------------------------

def test_health_missing_none():
    gs = GateState()
    d = evaluate_health_gate(None, gs)
    assert d.allow is False
    assert d.reason == "health_missing"
    assert gs.halted is True


def test_health_missing_non_dict():
    gs = GateState()
    d = evaluate_health_gate("bad", gs)
    assert d.allow is False
    assert d.reason == "health_missing"


# ---------------------------------------------------------------------------
# 3. Stale health data
# ---------------------------------------------------------------------------

def test_health_stale():
    gs = GateState()
    d = evaluate_health_gate(_health_obj(ts_offset=30), gs, max_health_staleness_sec=15)
    assert d.allow is False
    assert d.reason == "health_stale"


# ---------------------------------------------------------------------------
# 4. Degradation: overall_not_ok
# ---------------------------------------------------------------------------

def test_degraded_overall_not_ok():
    gs = GateState()
    d = evaluate_health_gate(_health_obj(state="degraded"), gs)
    assert d.allow is False
    assert d.reason == "overall_not_ok"
    assert d.state == "degraded"


# ---------------------------------------------------------------------------
# 5. Degradation: collector disconnected
# ---------------------------------------------------------------------------

def test_degraded_collector_disconnected():
    gs = GateState()
    d = evaluate_health_gate(_health_obj(connected=False), gs)
    assert d.allow is False
    assert d.reason == "collector_disconnected"


# ---------------------------------------------------------------------------
# 6. Degradation: data stale (lag)
# ---------------------------------------------------------------------------

def test_degraded_data_stale_lag():
    gs = GateState()
    d = evaluate_health_gate(_health_obj(lag_sec=60), gs, max_collector_lag_sec=30)
    assert d.allow is False
    assert d.reason == "data_stale"


# ---------------------------------------------------------------------------
# 7. Escalation after prolonged degradation
# ---------------------------------------------------------------------------

def test_escalation_after_max_degraded():
    gs = GateState()
    now = time.time()
    # First check starts degradation timer — use fresh timestamp
    evaluate_health_gate(_health_obj(connected=False), gs, now_ts=now)
    assert gs.degraded_since > 0

    # After max_degraded_sec, should escalate to halted.
    # IMPORTANT: health_obj timestamp must be fresh relative to now+130,
    # otherwise health_stale fires first.
    from datetime import datetime, timezone
    future_ts = datetime.fromtimestamp(now + 130, tz=timezone.utc).isoformat()
    h = _health_obj(connected=False)
    h["ts_utc"] = future_ts

    d = evaluate_health_gate(
        h, gs,
        now_ts=now + 130,
        max_degraded_sec=120,
        halt_cooldown_sec=60,
    )
    assert d.allow is False
    assert d.state == "halted"
    assert d.reason == "reconnect_escalation"
    assert gs.halted is True


# ---------------------------------------------------------------------------
# 8. Reconnect escalation
# ---------------------------------------------------------------------------

def test_reconnect_escalation():
    gs = GateState()
    d = evaluate_health_gate(
        _health_obj(reconnects=15), gs,
        max_reconnects_5m=10,
        halt_cooldown_sec=60,
    )
    assert d.allow is False
    assert d.reason == "reconnect_escalation"
    assert gs.halted is True
    assert gs.halt_until_ts > 0


def test_error_escalation():
    gs = GateState()
    d = evaluate_health_gate(
        _health_obj(errors=20), gs,
        max_errors_5m=10,
        halt_cooldown_sec=30,
    )
    assert d.allow is False
    assert d.reason == "reconnect_escalation"


# ---------------------------------------------------------------------------
# 9. Halted state with cooldown
# ---------------------------------------------------------------------------

def test_halted_cooldown_blocks():
    gs = GateState()
    now = time.time()
    gs.halted = True
    gs.halt_until_ts = now + 60
    d = evaluate_health_gate(_health_obj(), gs, now_ts=now)
    assert d.allow is False
    assert d.reason == "reconnect_escalation_cooldown"


def test_halted_cooldown_expires():
    gs = GateState()
    now = time.time()
    gs.halted = True
    gs.halt_until_ts = now - 1  # expired
    d = evaluate_health_gate(_health_obj(), gs, now_ts=now)
    assert d.allow is True
    assert gs.halted is False


# ---------------------------------------------------------------------------
# 10. Recovery resets degradation
# ---------------------------------------------------------------------------

def test_recovery_resets_degraded_since():
    gs = GateState()
    now = time.time()
    evaluate_health_gate(_health_obj(connected=False), gs, now_ts=now)
    assert gs.degraded_since > 0

    # Health is ok now
    d = evaluate_health_gate(_health_obj(), gs, now_ts=now + 10)
    assert d.allow is True
    assert gs.degraded_since == 0.0


# ---------------------------------------------------------------------------
# 11. Ingestion probe
# ---------------------------------------------------------------------------

def test_ingestion_probe_ok():
    gs = GateState()
    d = evaluate_health_gate(
        _health_obj(), gs,
        use_ingestion_check=True,
        ingestion_probe=lambda: (True, "all good"),
    )
    assert d.allow is True
    assert gs.last_ingestion_ok is True


def test_ingestion_probe_stalled():
    gs = GateState()
    d = evaluate_health_gate(
        _health_obj(), gs,
        use_ingestion_check=True,
        ingestion_probe=lambda: (False, "no new rows"),
        ingestion_check_cooldown_sec=0,
    )
    assert d.allow is False
    assert d.reason == "ingestion_stalled"


def test_ingestion_probe_exception():
    gs = GateState()
    d = evaluate_health_gate(
        _health_obj(), gs,
        use_ingestion_check=True,
        ingestion_probe=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        ingestion_check_cooldown_sec=0,
    )
    assert d.allow is False
    assert "probe_error" in gs.last_ingestion_reason


def test_ingestion_probe_cooldown_skips():
    gs = GateState()
    gs.last_ingestion_check_ts = time.time()  # just checked
    gs.last_ingestion_ok = True
    calls = []
    d = evaluate_health_gate(
        _health_obj(), gs,
        use_ingestion_check=True,
        ingestion_probe=lambda: (calls.append(1), (True, "ok"))[1],
        ingestion_check_cooldown_sec=60,
    )
    assert d.allow is True
    assert len(calls) == 0  # probe not called


# ---------------------------------------------------------------------------
# 12. load_overall_health
# ---------------------------------------------------------------------------

def test_load_overall_health_missing():
    base = _local_tmp_dir("missing")
    result = load_overall_health(base / "nonexistent.json")
    assert result is None


def test_load_overall_health_valid():
    base = _local_tmp_dir("valid")
    p = base / "overall.json"
    p.write_text(json.dumps({"state": "ok"}), encoding="utf-8")
    result = load_overall_health(p)
    assert result == {"state": "ok"}


def test_load_overall_health_corrupt():
    base = _local_tmp_dir("corrupt")
    p = base / "overall.json"
    p.write_text("NOT JSON", encoding="utf-8")
    result = load_overall_health(p)
    assert result is None


def test_write_paper_trader_health_includes_paper_contract(monkeypatch):
    from execution.health_gate import GateDecision, write_paper_trader_health
    base = _local_tmp_dir("paper_contract")

    monkeypatch.setenv("SCALPER_ENV_PROFILE", "paper")
    monkeypatch.setenv("SCALPER_DRY_RUN", "1")
    monkeypatch.setenv("PAPER_EXECUTION_MODE", "router_blocked")
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setenv("PAPER_ALLOW_LIVE_PRIVATE_API", "0")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    decision = GateDecision(
        allow=True,
        reason="",
        state="ok",
        collector_connected=True,
        collector_lag_sec=2,
        reconnects_last_5m=0,
        errors_last_5m=0,
    )
    write_paper_trader_health(decision, "", root=base)

    payload = json.loads((base / "paper_trader.json").read_text(encoding="utf-8"))
    assert payload["paper_profile_active"] is True
    assert payload["paper_execution_mode"] == "router_blocked"
    assert payload["binance_testnet"] is True
    assert payload["paper_allow_live_private_api"] is False
    assert payload["startup_contract_safe"] is True

    overall = json.loads((base / "overall.json").read_text(encoding="utf-8"))
    assert overall["components"]["paper_trader"]["paper_execution_mode"] == "router_blocked"
    assert overall["components"]["paper_trader"]["startup_contract_safe"] is True
