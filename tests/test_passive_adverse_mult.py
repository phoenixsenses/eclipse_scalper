from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import micro_edge_backtest as mb


def test_passive_adverse_mult_reduces_avg_net(monkeypatch) -> None:
    monkeypatch.setattr(mb, "rule_fires", lambda *args, **kwargs: True)
    monkeypatch.setattr(mb, "rule_predicted_side", lambda *args, **kwargs: "LONG")
    rows = []
    for i in range(200):
        rows.append(
            {
                "ts_ms": float(1000 + i),
                "mid": float(100.0 + (i * 0.01)),
                "spread": 0.0001,
                "trade_intensity": 2500.0,
                "micro_volatility": 0.001,
                "imbalance": 0.6,
                "ret_1": 0.0,
                "volume": 1.0,
                "liquidity_proxy": 1.0,
            }
        )
    base_params = {
        "seed": 42,
        "base_touch": 1.0,
        "base_full_cond_touch": 1.0,
        "base_adverse_bps": 0.2,
        "maker_fee_bps": 0.5,
        "edges": {
            "spread": [0.00005, 0.00010],
            "trade_intensity": [1000.0, 3000.0],
            "vol_proxy": [0.0005, 0.0015],
            "imbalance_for_fill": [0.4, 0.8],
        },
        "touch_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
        "full_rates": {"spread": [1, 1, 1], "trade_intensity": [1, 1, 1], "vol_proxy": [1, 1, 1], "imbalance_for_fill": [1, 1, 1]},
        "adverse_means": {"spread": [0.2, 0.2, 0.2], "trade_intensity": [0.2, 0.2, 0.2], "vol_proxy": [0.2, 0.2, 0.2], "imbalance_for_fill": [0.2, 0.2, 0.2]},
        "queue_competition_strength": 0.0,
        "adverse_toxicity_strength": 0.0,
    }
    sim_a = mb.simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={},
        labels=None,
        hold_buckets=5,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        exec_model="passive_realistic",
        passive_params={**base_params, "passive_adverse_mult": 1.0},
    )
    sim_b = mb.simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={},
        labels=None,
        hold_buckets=5,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        exec_model="passive_realistic",
        passive_params={**base_params, "passive_adverse_mult": 1.2},
    )
    assert float(sim_b["filled_only_metrics"]["avg_net"]) < float(sim_a["filled_only_metrics"]["avg_net"])

