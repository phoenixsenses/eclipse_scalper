from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import sweep_micro_edge_costs as mod


def test_cost_sweep_recomputes_net_and_break_even():
    rows = [
        {"rule_name": "r1", "resolved_side": "LONG", "gross_ret": 0.0020, "direction_match": True},
        {"rule_name": "r1", "resolved_side": "LONG", "gross_ret": 0.0010, "direction_match": True},
        {"rule_name": "r1", "resolved_side": "LONG", "gross_ret": 0.0005, "direction_match": False},
    ]
    s0 = mod.summarize_with_cost(rows, fee_bps=0.0, slip_bps=0.0)
    s6 = mod.summarize_with_cost(rows, fee_bps=2.0, slip_bps=1.0)  # 6 bps roundtrip
    assert float(s0["avg_net_ret"]) > float(s6["avg_net_ret"])
    assert abs(float(s0["avg_gross_ret"]) - 0.0011666666666666668) < 1e-12
    assert abs(float(s0["break_even_cost_bps_total"]) - (0.0011666666666666668 * 10000.0)) < 1e-9


def test_cost_sweep_main_runs():
    root = Path(__file__).resolve().parents[1] / "logs"
    root.mkdir(parents=True, exist_ok=True)
    debug = root / "test_sweep_micro_edge_costs_debug.jsonl"
    data = [
        {
            "rule_name": "r1",
            "resolved_side": "LONG",
            "gross_ret": 0.002,
            "net_ret": 0.001,
            "cost": 0.001,
            "direction_match": True,
        },
        {
            "rule_name": "r1",
            "resolved_side": "SHORT",
            "gross_ret": -0.001,
            "net_ret": -0.002,
            "cost": 0.001,
            "direction_match": False,
        },
    ]
    debug.write_text("\n".join(json.dumps(x) for x in data) + "\n", encoding="utf-8")
    code = mod.main(
        [
            "--debug",
            str(debug),
            "--group-by",
            "overall",
            "--fee-bps",
            "0,2",
            "--slip-bps",
            "0,1",
            "--min-n",
            "1",
            "--top-k",
            "2",
        ]
    )
    assert code == 0
