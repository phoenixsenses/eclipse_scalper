from execution import event_lane_gate


def test_applies_to_live_event_gate_matches() -> None:
    assert event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "micro_edge_v3_passive_alpha",
        60,
        signal={"source": "micro_signal", "min_imbalance": 0.85},
    )


def test_applies_to_live_event_gate_prefers_pocket_scope_over_current_feature() -> None:
    assert event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "micro_edge_v3_passive_alpha",
        60,
        signal={
            "source": "micro_signal",
            "pocket_name": "imb>=0.85_int>=7000_spr<=0.000200",
            "min_imbalance": 0.42,
        },
    )


def test_applies_to_live_event_gate_requires_narrow_scope() -> None:
    assert not event_lane_gate.applies_to_live_event_gate(
        "BTCUSDT",
        "micro_edge_v3_passive_alpha",
        60,
        signal={"source": "micro_signal", "min_imbalance": 0.91},
    )
    assert not event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "micro_edge_v3_passive_alpha",
        120,
        signal={"source": "micro_signal", "min_imbalance": 0.91},
    )
    assert not event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "some_other_rule",
        60,
        signal={"source": "micro_signal", "min_imbalance": 0.91},
    )
    assert not event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "micro_edge_v3_passive_alpha",
        60,
        signal={"source": "strategy_signal", "min_imbalance": 0.91},
    )
    assert not event_lane_gate.applies_to_live_event_gate(
        "ETHUSDT",
        "micro_edge_v3_passive_alpha",
        60,
        signal={"source": "micro_signal", "min_imbalance": 0.50},
    )


def test_should_block_returns_false_when_not_applicable() -> None:
    blocked, reason, details = event_lane_gate.should_block_event_gate(
        {"gate": "blocked", "allow_trade": False, "blocked_lanes": ["book_proxy_pressure"]},
        symbol="BTCUSDT",
        rule_name="micro_edge_v3_passive_alpha",
        horizon_sec=60,
        signal={"source": "micro_signal", "min_imbalance": 0.91},
    )
    assert blocked is False
    assert reason == "gate_not_applicable"
    assert details == {}


def test_should_block_returns_false_for_non_blocking_gate_states() -> None:
    signal = {"source": "micro_signal", "min_imbalance": 0.91}
    for gate in ("inactive", "no_data", "inactive_pocket", "allowed"):
        blocked, reason, details = event_lane_gate.should_block_event_gate(
            {"gate": gate, "allow_trade": True, "blocked_lanes": []},
            symbol="ETHUSDT",
            rule_name="micro_edge_v3_passive_alpha",
            horizon_sec=60,
            signal=signal,
        )
        assert blocked is False
        assert reason == gate
        assert details == {}


def test_should_block_returns_true_when_blocked() -> None:
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
        payload,
        symbol="ETHUSDT",
        rule_name="micro_edge_v3_passive_alpha",
        horizon_sec=60,
        signal={"source": "micro_signal", "min_imbalance": 0.91},
    )
    assert blocked is True
    assert reason == "event_lane_gate_blocked"
    assert details["blocking_lanes"] == ["book_proxy_pressure"]


def test_load_current_event_gate_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "0")
    result = event_lane_gate.load_current_event_gate(db="/nonexistent/path.db", symbol="ETHUSDT")
    assert result["gate"] == "inactive"
    assert result["allow_trade"] is True
    assert result["pocket_active"] is False

    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
    result = event_lane_gate.load_current_event_gate(db="/nonexistent/path.db", symbol="ETHUSDT")
    assert result["gate"] == "no_data"
    assert result["allow_trade"] is True


def test_load_current_event_gate_payload_schema(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
    result = event_lane_gate.load_current_event_gate(db="/nonexistent/path.db", symbol="ETHUSDT")
    for key in (
        "symbol",
        "gate",
        "pocket_active",
        "allow_trade",
        "blocked_lanes",
        "reason",
        "latest_ts_ms",
        "latest_abs_imbalance",
        "lanes",
    ):
        assert key in result
    assert "book_proxy_pressure" in result["lanes"]
    assert "volatility_burst" in result["lanes"]
