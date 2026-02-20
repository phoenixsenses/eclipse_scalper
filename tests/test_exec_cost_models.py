from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import compute_exec_cost
from tools import micro_edge_backtest as mb


def test_exec_cost_models_values():
    taker = compute_exec_cost(
        exec_model="taker",
        fee_bps=4.0,
        slip_bps=2.0,
        maker_fee_bps=0.5,
        maker_penalty_bps=0.5,
        spread_ratio=0.0002,
    )
    maker = compute_exec_cost(
        exec_model="maker",
        fee_bps=4.0,
        slip_bps=2.0,
        maker_fee_bps=0.5,
        maker_penalty_bps=0.5,
        spread_ratio=0.0002,
    )
    mid = compute_exec_cost(
        exec_model="mid",
        fee_bps=4.0,
        slip_bps=2.0,
        maker_fee_bps=0.5,
        maker_penalty_bps=0.5,
        spread_ratio=0.0002,
    )
    halfspread = compute_exec_cost(
        exec_model="halfspread",
        fee_bps=4.0,
        slip_bps=2.0,
        maker_fee_bps=0.5,
        maker_penalty_bps=0.5,
        spread_ratio=0.0002,
    )
    assert abs(float(taker) - 0.0012) < 1e-12
    assert abs(float(maker) - 0.0002) < 1e-12
    assert abs(float(mid) - 0.0008) < 1e-12
    # 2 * (fee_per_side + 0.5*spread)
    assert abs(float(halfspread) - (2.0 * (0.0004 + 0.0001))) < 1e-12


def test_halfspread_requires_spread():
    cost = compute_exec_cost(
        exec_model="halfspread",
        fee_bps=4.0,
        slip_bps=2.0,
        maker_fee_bps=0.5,
        maker_penalty_bps=0.5,
        spread_ratio=None,
    )
    assert cost is None


def test_npa_cost_applied_to_fills_only(monkeypatch) -> None:
    """Unfilled passive attempts must contribute net_return=0.0 (costs not applied). DAT-04."""
    monkeypatch.setattr(mb, "rule_fires", lambda *args, **kwargs: True)
    monkeypatch.setattr(mb, "rule_predicted_side", lambda *args, **kwargs: "LONG")

    call_count = [0]

    def _alternating_fill(**kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return {
                "filled": True,
                "fill_fraction": 1.0,
                "p_touch": 0.8,
                "p_full_cond_touch": 0.8,
                "adverse_selection_bps": 0.1,
                "effective_cost_bps": 1.1,
                "execution_price_adjustment": 0.0,
                "queue_competition_score": 0.0,
                "toxicity_score": 0.0,
                "fill_index_offset": 0,
            }
        else:
            return {
                "filled": False,
                "fill_fraction": 0.0,
                "p_touch": 0.2,
                "p_full_cond_touch": 0.0,
                "adverse_selection_bps": 0.0,
                "effective_cost_bps": 0.0,
                "execution_price_adjustment": 0.0,
                "queue_competition_score": 0.0,
                "toxicity_score": 0.0,
                "fill_index_offset": 0,
            }

    monkeypatch.setattr(mb, "simulate_passive_fill", _alternating_fill)

    rows = [
        {
            "ts_ms": float(1000 + i),
            "mid": float(100.0 + i * 0.01),
            "spread": 0.0001,
            "trade_intensity": 3000.0,
            "micro_volatility": 0.001,
            "imbalance": 0.6,
            "ret_1": 0.001,
        }
        for i in range(20)
    ]
    sim = mb.simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={},
        labels=None,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        exec_model="passive_realistic",
        passive_params={"maker_fee_bps": 0.5},
    )
    attempt_rows = sim["attempt_rows"]
    al = sim["attempt_level_metrics"]
    assert len(attempt_rows) > 0
    unfilled = [a for a in attempt_rows if not bool(a.get("filled", False))]
    assert len(unfilled) > 0, "Need at least one unfilled attempt for this test"
    assert all(float(a.get("net_return", 0.0)) == 0.0 for a in unfilled), (
        "Unfilled passive attempts must have net_return=0.0 (costs not applied)"
    )
    n_att = int(al["n_attempts"])
    if n_att > 0:
        expected_npa = sum(float(a.get("net_return", 0.0)) for a in attempt_rows) / n_att
        assert abs(float(al["net_per_attempt"]) - expected_npa) < 1e-12, (
            "net_per_attempt must equal sum(net_returns)/n_attempts (unfilled contribute 0.0)"
        )

