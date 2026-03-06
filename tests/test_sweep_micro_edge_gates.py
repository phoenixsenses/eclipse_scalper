from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sweep_micro_edge_gates import event_passes_config, parse_sweep_clause, summarize


def test_sweep_ranks_best_config():
    rows = [
        {"feature": {"spread": 0.0001, "trade_intensity": 1200}, "net_ret": 0.003, "gross_ret": 0.004, "cost": 0.001, "direction_match": True},
        {"feature": {"spread": 0.00015, "trade_intensity": 1100}, "net_ret": 0.002, "gross_ret": 0.003, "cost": 0.001, "direction_match": True},
        {"feature": {"spread": 0.0006, "trade_intensity": 1500}, "net_ret": -0.002, "gross_ret": -0.001, "cost": 0.001, "direction_match": False},
        {"feature": {"spread": 0.0007, "trade_intensity": 1300}, "net_ret": -0.001, "gross_ret": 0.0, "cost": 0.001, "direction_match": False},
    ]

    c1 = [("spread", "<=", 0.0002), ("trade_intensity", ">=", 1000.0)]
    c2 = [("spread", "<=", 0.0008), ("trade_intensity", ">=", 1000.0)]
    s1 = summarize([r for r in rows if event_passes_config(r, c1)])
    s2 = summarize([r for r in rows if event_passes_config(r, c2)])
    assert int(s1["n"]) == 2
    assert int(s2["n"]) == 4
    assert float(s1["avg_net"]) > float(s2["avg_net"])


def test_parse_sweep_clause():
    feat, op, vals = parse_sweep_clause("spread<=0.0001,0.0005")
    assert feat == "spread"
    assert op == "<="
    assert vals == [0.0001, 0.0005]

