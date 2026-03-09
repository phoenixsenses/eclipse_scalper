"""tests/test_watch_regime_recovery.py — Tests for tools/watch_regime_recovery.py"""
import pytest
import tools.watch_regime_recovery as wr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate(gate: str) -> dict:
    return {
        "gate": gate,
        "lanes": {
            "book_proxy_pressure": {"active": gate == "BLOCKED", "age_sec": 5},
            "volatility_burst": {"active": False, "age_sec": None},
        },
    }


# ---------------------------------------------------------------------------
# 1. ALLOWED gate marks ready (consecutive=1)
# ---------------------------------------------------------------------------

def test_allowed_gate_marks_ready(monkeypatch):
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate("ALLOWED"))
    state = [0]
    status = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=1, consecutive_state=state)
    assert status["allowed"] is True
    assert status["consecutive_allowed"] == 1
    assert status["ready_to_revalidate"] is True
    assert wr.REVALIDATE_CMD in status["revalidate_cmd"]


# ---------------------------------------------------------------------------
# 2. BLOCKED gate not ready
# ---------------------------------------------------------------------------

def test_blocked_gate_not_ready(monkeypatch):
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate("BLOCKED"))
    state = [0]
    status = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=1, consecutive_state=state)
    assert status["allowed"] is False
    assert status["consecutive_allowed"] == 0
    assert status["ready_to_revalidate"] is False


# ---------------------------------------------------------------------------
# 3. UNKNOWN gate (DB error) not ready
# ---------------------------------------------------------------------------

def test_unknown_gate_not_ready(monkeypatch):
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate("UNKNOWN"))
    state = [0]
    status = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=1, consecutive_state=state)
    assert status["allowed"] is False
    assert status["ready_to_revalidate"] is False


# ---------------------------------------------------------------------------
# 4. Consecutive threshold: needs N ALLOWED before READY
# ---------------------------------------------------------------------------

def test_consecutive_threshold_requires_multiple_allowed(monkeypatch):
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate("ALLOWED"))
    state = [0]

    # First check: consecutive=1, required=2 -> not ready yet
    s1 = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=2, consecutive_state=state)
    assert s1["consecutive_allowed"] == 1
    assert s1["ready_to_revalidate"] is False

    # Second check: consecutive=2, required=2 -> now ready
    s2 = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=2, consecutive_state=state)
    assert s2["consecutive_allowed"] == 2
    assert s2["ready_to_revalidate"] is True


# ---------------------------------------------------------------------------
# 5. Consecutive counter resets after a BLOCKED check
# ---------------------------------------------------------------------------

def test_consecutive_resets_on_blocked(monkeypatch):
    calls = iter(["ALLOWED", "BLOCKED", "ALLOWED"])
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate(next(calls)))
    state = [0]

    wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=2, consecutive_state=state)
    assert state[0] == 1

    wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=2, consecutive_state=state)
    assert state[0] == 0  # reset

    s = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=2, consecutive_state=state)
    assert state[0] == 1
    assert s["ready_to_revalidate"] is False


# ---------------------------------------------------------------------------
# 6. Status payload contains required fields
# ---------------------------------------------------------------------------

def test_status_payload_schema(monkeypatch):
    monkeypatch.setattr(wr, "_call_check_gate", lambda db, symbol: _make_gate("ALLOWED"))
    state = [0]
    status = wr.evaluate_status("/fake/db", "ETHUSDT", consecutive_required=1, consecutive_state=state)
    for key in ("timestamp_utc", "symbol", "gate_result", "allowed",
                "consecutive_allowed", "consecutive_required", "ready_to_revalidate", "revalidate_cmd"):
        assert key in status, f"missing key: {key}"
