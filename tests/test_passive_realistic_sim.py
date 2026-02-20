from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.passive_execution_simulator import simulate_passive_fill
from tools import micro_edge_backtest as mb


def test_queue_competition_reduces_touch_probability() -> None:
    params = {
        "seed": 42,
        "base_touch": 0.8,
        "base_full_cond_touch": 0.8,
        "base_adverse_bps": 0.1,
        "maker_fee_bps": 0.5,
        "edges": {
            "spread": [0.00005, 0.00010],
            "trade_intensity": [2000.0, 4000.0],
            "vol_proxy": [0.00005, 0.00010],
            "imbalance_for_fill": [0.4, 0.8],
        },
        "touch_rates": {
            "spread": [0.8, 0.8, 0.8],
            "trade_intensity": [0.8, 0.8, 0.8],
            "vol_proxy": [0.8, 0.8, 0.8],
            "imbalance_for_fill": [0.8, 0.8, 0.8],
        },
        "full_rates": {
            "spread": [0.8, 0.8, 0.8],
            "trade_intensity": [0.8, 0.8, 0.8],
            "vol_proxy": [0.8, 0.8, 0.8],
            "imbalance_for_fill": [0.8, 0.8, 0.8],
        },
        "adverse_means": {
            "spread": [0.1, 0.1, 0.1],
            "trade_intensity": [0.1, 0.1, 0.1],
            "vol_proxy": [0.1, 0.1, 0.1],
            "imbalance_for_fill": [0.1, 0.1, 0.1],
        },
        "queue_competition_strength": 0.5,
    }
    base_event = {"event_id": "E1", "symbol": "BTCUSDT", "side": "LONG", "entry_price": 100.0, "future_mids": [99.8, 99.7]}
    low = simulate_passive_fill(
        base_event,
        horizon_sec=60,
        features={"spread": 0.00003, "trade_intensity": 1000.0, "vol_proxy": 0.00003, "imbalance_for_fill": 0.6},
        params=params,
    )
    high = simulate_passive_fill(
        base_event,
        horizon_sec=60,
        features={"spread": 0.00020, "trade_intensity": 8000.0, "vol_proxy": 0.00003, "imbalance_for_fill": 0.6},
        params=params,
    )
    assert float(high["p_touch"]) < float(low["p_touch"])


def test_toxicity_gate_blocks_attempts(monkeypatch) -> None:
    monkeypatch.setattr(mb, "rule_fires", lambda *args, **kwargs: True)
    monkeypatch.setattr(mb, "rule_predicted_side", lambda *args, **kwargs: "LONG")
    rows = []
    for i in range(10):
        rows.append(
            {
                "ts_ms": float(1000 + i),
                "mid": float(100.0 + i * 0.1),
                "spread": 0.0001,
                "trade_intensity": 5000.0,
                "micro_volatility": 0.01,
                "imbalance": 0.1,
                "ret_1": 0.0,
            }
        )
    sim = mb.simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={},
        labels=None,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=4.0,
        slip_bps=2.0,
        exec_model="passive_realistic",
        passive_params={
            "seed": 7,
            "base_touch": 1.0,
            "base_full_cond_touch": 1.0,
            "base_adverse_bps": 0.1,
            "maker_fee_bps": 0.5,
            "edges": {
                "spread": [0.00005, 0.00010],
                "trade_intensity": [2000.0, 4000.0],
                "vol_proxy": [0.00005, 0.00010],
                "imbalance_for_fill": [0.4, 0.8],
            },
            "touch_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
            "full_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
            "adverse_means": {"spread": [0.1, 0.1, 0.1], "trade_intensity": [0.1, 0.1, 0.1], "vol_proxy": [0.1, 0.1, 0.1], "imbalance_for_fill": [0.1, 0.1, 0.1]},
        },
        toxicity_cfg={"enabled": True, "vol_high_threshold": 0.005, "intensity_high_threshold": 3000.0, "imbalance_min_threshold": 0.3},
    )
    ds = sim["debug_stats"]
    assert int(ds["toxicity_blocked"]) > 0
    assert int(sim["metrics"]["n_trades"]) == 0
    assert int(sim["attempt_level_metrics"]["n_attempts"]) > 0


def test_adverse_bps_scaling_not_amplified() -> None:
    params = {
        "seed": 1,
        "base_touch": 1.0,
        "base_full_cond_touch": 1.0,
        "base_adverse_bps": 0.084,
        "maker_fee_bps": 0.5,
        "edges": {
            "spread": [0.00005, 0.00010],
            "trade_intensity": [2000.0, 4000.0],
            "vol_proxy": [0.00005, 0.00010],
            "imbalance_for_fill": [0.4, 0.8],
        },
        "touch_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
        "full_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
        "adverse_means": {"spread": [0.084, 0.084, 0.084], "trade_intensity": [0.084, 0.084, 0.084], "vol_proxy": [0.084, 0.084, 0.084], "imbalance_for_fill": [0.084, 0.084, 0.084]},
        "queue_competition_strength": 0.0,
        "adverse_toxicity_strength": 0.0,
    }
    out = simulate_passive_fill(
        {"event_id": "E2", "symbol": "BTCUSDT", "side": "LONG", "entry_price": 100.0, "future_mids": [99.0, 98.9]},
        horizon_sec=60,
        features={"spread": 0.00005, "trade_intensity": 1000.0, "vol_proxy": 0.00003, "imbalance_for_fill": 0.5},
        params=params,
    )
    assert bool(out["filled"]) is True
    # 0.084 bps should remain 0.084 bps-scale, not 8.4 bps.
    assert 0.05 <= float(out["adverse_selection_bps"]) <= 0.15
    # Total bps should be near 1.084 bps (2*0.5 + 0.084)
    assert 1.0 <= float(out["effective_cost_bps"]) <= 1.2


def test_attempt_gate_reduces_n_attempts(monkeypatch) -> None:
    """Pre-attempt gate must reduce n_attempts; n_signals_before_gate tracks pre-gate count."""
    monkeypatch.setattr(mb, "rule_fires", lambda *args, **kwargs: True)
    monkeypatch.setattr(mb, "rule_predicted_side", lambda *args, **kwargs: "LONG")
    # Rows: intensity=3000 (strong) at indices 0, 6, 12, 18, 24; 500 (weak) otherwise.
    # With hold=2, cooldown=0, rows 0, 3, 6, 9, 12, 15, 18, 21, 24 are visited (9 total).
    rows = [
        {
            "ts_ms": float(1000 + i),
            "mid": float(100.0 + i * 0.01),
            "spread": 0.0001,
            "trade_intensity": 3000.0 if (i % 6 == 0) else 500.0,
            "micro_volatility": 0.001,
            "imbalance": 0.6,
            "ret_1": 0.001,
        }
        for i in range(30)
    ]
    base_kwargs = dict(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={},
        labels=None,
        hold_buckets=2,
        cooldown_buckets=0,
        fee_bps=4.0,
        slip_bps=2.0,
        exec_model="taker",
    )
    sim_no_gate = mb.simulate_rule_trades(**base_kwargs)
    n_no_gate = int(sim_no_gate["attempt_level_metrics"]["n_attempts"])
    # n_signals_before_gate == n_attempts when no gate (all signals become attempts for taker)
    assert int(sim_no_gate["attempt_level_metrics"]["n_signals_before_gate"]) == n_no_gate

    sim_gated = mb.simulate_rule_trades(**base_kwargs, attempt_gate_bounds={"min_trade_intensity_strong": 2000.0})
    n_gated = int(sim_gated["attempt_level_metrics"]["n_attempts"])
    n_before = int(sim_gated["attempt_level_metrics"]["n_signals_before_gate"])
    assert n_gated < n_no_gate, f"Gate must reduce attempts: {n_gated} < {n_no_gate}"
    assert n_before >= n_gated, "n_signals_before_gate must be >= n_attempts_after_gate"
    assert int(sim_gated["debug_stats"]["attempt_gate_blocked"]) > 0
