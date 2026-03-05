from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import micro_edge_backtest as mb


def test_scratch_rule_reduces_adverse_loss(monkeypatch) -> None:
    rows = [
        {"ts_ms": 0, "mid": 100.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": 0.0},
        {"ts_ms": 1, "mid": 100.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": 0.0},  # entry
        {"ts_ms": 2, "mid": 99.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": -0.01},  # scratch trigger
        {"ts_ms": 3, "mid": 97.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": -0.02},
        {"ts_ms": 4, "mid": 95.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": -0.02},  # default exit
        {"ts_ms": 5, "mid": 95.0, "imbalance": 0.9, "trade_intensity": 3000.0, "spread": 0.0002, "ret_1": 0.0},
    ]
    thresholds = {"imb_q90": 0.5}

    monkeypatch.setattr(
        mb,
        "simulate_passive_fill",
        lambda **kwargs: {
            "filled": True,
            "fill_fraction": 1.0,
            "effective_cost_bps": 0.0,
            "adverse_selection_bps": 0.0,
            "execution_price_adjustment": 0.0,
            "fill_index_offset": 0,
            "queue_competition_score": 0.0,
            "toxicity_score": 0.0,
        },
    )

    base = mb.simulate_rule_trades(
        rows=rows,
        rule_name="imbalance_gt_q90_up",
        side="LONG",
        thresholds=thresholds,
        labels=None,
        hold_buckets=3,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        exec_model="passive_realistic",
        maker_fee_bps=0.0,
        maker_penalty_bps=0.0,
        passive_params={},
        bucket_sec=1,
    )
    scratched = mb.simulate_rule_trades(
        rows=rows,
        rule_name="imbalance_gt_q90_up",
        side="LONG",
        thresholds=thresholds,
        labels=None,
        hold_buckets=3,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        exec_model="passive_realistic",
        maker_fee_bps=0.0,
        maker_penalty_bps=0.0,
        passive_params={},
        bucket_sec=1,
        scratch_bps=50.0,
        scratch_window_sec=2,
        scratch_taker_fee_bps=0.0,
        scratch_slippage_bps=0.0,
    )
    assert base["trades"], "expected at least one trade in baseline"
    assert scratched["trades"], "expected at least one trade in scratch scenario"
    base_net = float(base["trades"][0]["net_return"])
    scratch_net = float(scratched["trades"][0]["net_return"])
    assert bool(scratched["trades"][0].get("scratch_triggered")) is True
    assert scratch_net > base_net

