from __future__ import annotations

from pathlib import Path

try:
    from execution.alpha_gate import evaluate_alpha_gate
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.alpha_gate import evaluate_alpha_gate


def test_alpha_gate_blocks_negative_edge() -> None:
    dec = evaluate_alpha_gate(
        {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": -1.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.1},
        now_ts=100.0,
        min_pnl_net_per_fill=0.0,
    )
    assert dec.blocked is True
    assert dec.reason == "alpha_negative_edge"


def test_alpha_gate_blocks_low_fill_rate() -> None:
    dec = evaluate_alpha_gate(
        {"fills_count": 1, "decisions_count": 20, "pnl_net_sum": 1.0, "fee_dominates_count": 0, "spread_cost_est_sum": 0.0},
        now_ts=100.0,
        min_pnl_net_per_fill=-1.0,
        min_fill_rate=0.2,
    )
    assert dec.blocked is True
    assert dec.reason == "alpha_fill_rate_low"


def test_alpha_gate_allows_ok_metrics() -> None:
    dec = evaluate_alpha_gate(
        {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": 2.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.1},
        now_ts=100.0,
        min_pnl_net_per_fill=0.0,
        min_fill_rate=0.2,
        max_fee_dominates_frac=0.5,
    )
    assert dec.blocked is False
    assert dec.reason == "alpha_ok"

