"""tests/test_event_lane_gate.py — Tests for execution/event_lane_gate.py"""
import os
import pytest
from execution import event_lane_gate


# ---------------------------------------------------------------------------
# 1. applies_to_live_event_gate — match
# ---------------------------------------------------------------------------

def test_applies_to_live_event_gate_matches():
    assert event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT", "micro_edge_v3_passive_alpha", 60
    ) is True


def test_applies_to_live_event_gate_matches_lowercase_symbol():
    assert event_lane_gate.applies_to_live_event_gate(
        "ethusdt", "micro_edge_v3_passive_alpha", 60
    ) is True


# ---------------------------------------------------------------------------
# 2. applies_to_live_event_gate — no match
# ---------------------------------------------------------------------------

def test_applies_to_live_event_gate_wrong_symbol():
    assert event_lane_gate.applies_to_live_event_gate(
        "BTCUSDT", "micro_edge_v3_passive_alpha", 60
    ) is False


def test_applies_to_live_event_gate_wrong_horizon():
    assert event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT", "micro_edge_v3_passive_alpha", 120
    ) is False


def test_applies_to_live_event_gate_wrong_rule():
    assert event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT", "some_other_rule", 60
    ) is False


# ---------------------------------------------------------------------------
# 3. should_block_event_gate — not applicable
# ---------------------------------------------------------------------------

def test_should_block_returns_false_when_not_applicable():
    payload = {"gate": "blocked", "allow_trade": False, "blocked_lanes": ["book_proxy_pressure"]}
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        payload, symbol="BTCUSDT", rule_name="micro_edge_v3_passive_alpha", horizon_sec=60
    )
    assert blocked is False
    assert reason == "gate_not_applicable"
    assert details == {}


def test_should_block_returns_false_for_inactive_gate():
    payload = {"gate": "inactive", "allow_trade": True, "blocked_lanes": []}
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        payload, symbol="ETHUSDT", rule_name="micro_edge_v3_passive_alpha", horizon_sec=60
    )
    assert blocked is False
    assert reason == "inactive"


def test_should_block_returns_false_for_no_data_gate():
    payload = {"gate": "no_data", "allow_trade": True, "blocked_lanes": []}
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        payload, symbol="ETHUSDT", rule_name="micro_edge_v3_passive_alpha", horizon_sec=60
    )
    assert blocked is False
    assert reason == "no_data"


def test_should_block_returns_true_when_blocked():
    payload = {
        "gate": "blocked",
        "allow_trade": False,
        "blocked_lanes": ["book_proxy_pressure"],
        "latest_ts_ms": 1000000,
        "latest_abs_imbalance": 0.91,
        "lanes": {
            "book_proxy_pressure": {"rule_fired": True, "severity": "high"},
            "volatility_burst": {"rule_fired": False, "severity": "none"},
        },
    }
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        payload, symbol="ETHUSDT", rule_name="micro_edge_v3_passive_alpha", horizon_sec=60
    )
    assert blocked is True
    assert reason == "event_lane_gate_blocked"
    assert details["blocking_lanes"] == ["book_proxy_pressure"]


def test_should_block_returns_false_when_allowed():
    payload = {"gate": "allowed", "allow_trade": True, "blocked_lanes": []}
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        payload, symbol="ETHUSDT", rule_name="micro_edge_v3_passive_alpha", horizon_sec=60
    )
    assert blocked is False
    assert reason == "allowed"


# ---------------------------------------------------------------------------
# 4. load_current_event_gate — no data (non-existent DB path), never raises
# ---------------------------------------------------------------------------

def test_load_current_event_gate_no_data_returns_safe_default(monkeypatch):
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
    result = event_lane_gate.load_current_event_gate(
        db="/nonexistent/path/microstructure.db",
        symbol="ETHUSDT",
    )
    assert result["gate"] == "no_data"
    assert result["allow_trade"] is True
    assert "symbol" in result


def test_load_current_event_gate_inactive_when_disabled(monkeypatch):
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "0")
    result = event_lane_gate.load_current_event_gate(
        db="/nonexistent/path/microstructure.db",
        symbol="ETHUSDT",
    )
    assert result["gate"] == "inactive"
    assert result["allow_trade"] is True


def test_load_current_event_gate_never_raises(monkeypatch):
    """Must not raise on any error — returns safe default."""
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
    # Should not raise even with garbage path
    result = event_lane_gate.load_current_event_gate(
        db="\x00invalid\x00path",
        symbol="ETHUSDT",
    )
    assert isinstance(result, dict)
    assert result.get("allow_trade") is True


def test_load_current_event_gate_payload_schema(monkeypatch):
    """Payload must contain all required schema keys."""
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
    result = event_lane_gate.load_current_event_gate(
        db="/nonexistent/path/microstructure.db",
        symbol="ETHUSDT",
    )
    for key in ("symbol", "gate", "allow_trade", "blocked_lanes", "reason", "latest_ts_ms",
                "latest_abs_imbalance", "lanes"):
        assert key in result, f"missing key: {key}"
    assert "book_proxy_pressure" in result["lanes"]
    assert "volatility_burst" in result["lanes"]
