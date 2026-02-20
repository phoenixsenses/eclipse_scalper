from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import (
    compute_gross_return,
    compute_net_return,
    compute_trade_cost,
)
from tools.micro_edge_lib import rule_predicted_side


def test_long_sign():
    assert abs(compute_gross_return(100.0, 101.0, "LONG") - 0.01) < 1e-12
    assert abs(compute_gross_return(100.0, 99.0, "LONG") - (-0.01)) < 1e-12


def test_short_sign():
    assert abs(compute_gross_return(100.0, 101.0, "SHORT") - (-0.00990099009900991)) < 1e-12
    assert abs(compute_gross_return(100.0, 99.0, "SHORT") - 0.010101010101010166) < 1e-12


def test_net_return_with_cost():
    # fee=4bps, slip=2bps -> round trip cost=12bps
    cost = compute_trade_cost(4.0, 2.0)
    assert abs(cost - 0.0012) < 1e-12
    net, used_cost = compute_net_return(100.0, 101.0, "LONG", 4.0, 2.0)
    assert abs(used_cost - 0.0012) < 1e-12
    assert abs(net - (0.01 - 0.0012)) < 1e-12


def test_dynamic_rule_side_resolution():
    assert rule_predicted_side("intensity_spike_imbalance_cont", {"imbalance": 0.2}, "LONG") == "LONG"
    assert rule_predicted_side("intensity_spike_imbalance_cont", {"imbalance": -0.2}, "LONG") == "SHORT"
    assert rule_predicted_side("spread_spike_reversal", {"ret_1": 0.001}, "LONG") == "SHORT"
    assert rule_predicted_side("spread_spike_reversal", {"ret_1": -0.001}, "SHORT") == "LONG"
