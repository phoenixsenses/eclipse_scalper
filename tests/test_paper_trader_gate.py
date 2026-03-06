from __future__ import annotations

from pathlib import Path

try:
    from execution.health_gate import GateState, evaluate_health_gate
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.health_gate import GateState, evaluate_health_gate


def _health(state: str = "ok", connected: bool = True, lag: int = 0, r5: int = 0, e5: int = 0, ts: str = "2026-03-01T00:00:10Z"):
    return {
        "ts_utc": ts,
        "mode": "paper",
        "state": state,
        "components": {
            "collector": {
                "status": state,
                "connected": connected,
                "progress_lag_sec": lag,
                "reconnects_last_5m": r5,
                "errors_last_5m": e5,
            }
        },
    }


def test_gate_blocks_missing_or_stale_health() -> None:
    gs = GateState()
    d = evaluate_health_gate(None, gs, now_ts=10.0)
    assert d.allow is False
    assert d.reason == "health_missing"

    gs2 = GateState()
    d2 = evaluate_health_gate(_health(ts="2000-01-01T00:00:00Z"), gs2, now_ts=1_900_000_000.0)
    assert d2.allow is False
    assert d2.reason == "health_stale"


def test_gate_blocks_disconnected_and_resumes() -> None:
    gs = GateState()
    d1 = evaluate_health_gate(_health(state="degraded", connected=False, lag=2), gs, now_ts=100.0)
    assert not d1.allow
    assert d1.reason in ("overall_not_ok", "collector_disconnected")
    d2 = evaluate_health_gate(_health(state="ok", connected=True, lag=1), gs, now_ts=101.0)
    assert d2.allow


def test_gate_escalates_on_reconnect_threshold() -> None:
    gs = GateState()
    d = evaluate_health_gate(_health(state="degraded", connected=False, lag=5, r5=11, e5=0), gs, now_ts=100.0)
    assert d.allow is False
    assert d.reason == "reconnect_escalation"
    assert d.state == "halted"


def test_gate_reconnect_escalation_cooldown() -> None:
    gs = GateState()
    d1 = evaluate_health_gate(
        _health(state="degraded", connected=False, lag=5, r5=11, e5=0),
        gs,
        now_ts=100.0,
        halt_cooldown_sec=30,
    )
    assert d1.reason == "reconnect_escalation"
    assert d1.halt_until_ts >= 130.0
    d2 = evaluate_health_gate(_health(state="ok", connected=True, lag=0, r5=0, e5=0), gs, now_ts=110.0, halt_cooldown_sec=30)
    assert d2.allow is False
    assert d2.reason == "reconnect_escalation_cooldown"
    d3 = evaluate_health_gate(_health(state="ok", connected=True, lag=0, r5=0, e5=0), gs, now_ts=131.0, halt_cooldown_sec=30)
    assert d3.allow is True


def test_gate_ingestion_crosscheck_blocks_on_stall() -> None:
    gs = GateState()
    d = evaluate_health_gate(
        _health(state="ok", connected=True, lag=0, r5=0, e5=0),
        gs,
        now_ts=100.0,
        use_ingestion_check=True,
        ingestion_probe=lambda: (False, "rows_delta_zero"),
        ingestion_check_cooldown_sec=10,
    )
    assert d.allow is False
    assert d.reason == "ingestion_stalled"

