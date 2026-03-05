from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.execution.queue_sim import QueueSimParams, simulate_maker_queue_fill


def test_queue_sim_deterministic_fill_timing() -> None:
    n = 20
    df = pd.DataFrame(
        {
            "bid_qty": [10.0] * n,
            "ask_qty": [10.0] * n,
            "qty_sum": [5.0] * n,
            "trade_through_prob": [1.0] * n,
        }
    )
    sim = simulate_maker_queue_fill(df, entry_idx=0, side="buy", params=QueueSimParams(queue_frac=0.5, ttl_bars=10))
    assert sim["filled"] is True
    assert int(sim["fill_delay_bars"]) == 2


def test_queue_sim_adaptive_deep_book_reduces_queue_fraction() -> None:
    n = 20
    df = pd.DataFrame(
        {
            "bid_qty": [1000.0] * n,
            "ask_qty": [1000.0] * n,
            "qty_sum": [300.0] * n,
            "trade_through_prob": [0.2] * n,
        }
    )
    fixed = simulate_maker_queue_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueueSimParams(queue_mode="fixed", queue_frac=0.5, ttl_bars=10),
    )
    adaptive = simulate_maker_queue_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueueSimParams(
            queue_mode="adaptive",
            queue_frac=0.5,
            queue_depth_threshold=200.0,
            queue_frac_shallow=0.4,
            queue_frac_deep=0.1,
            ttl_bars=10,
        ),
    )
    assert float(adaptive["queue_frac_used"]) < float(fixed["queue_frac_used"])
    assert bool(fixed["filled"]) is True
    assert bool(adaptive["filled"]) is True
    assert int(adaptive["fill_delay_bars"]) <= int(fixed["fill_delay_bars"])


def test_queue_sim_adaptive_shallow_book_keeps_higher_queue_fraction() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "bid_qty": [20.0] * n,
            "ask_qty": [20.0] * n,
            "qty_sum": [3.0] * n,
            "trade_through_prob": [0.1] * n,
        }
    )
    sim = simulate_maker_queue_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueueSimParams(
            queue_mode="adaptive",
            queue_depth_threshold=500.0,
            queue_frac_shallow=0.45,
            queue_frac_deep=0.10,
            queue_adaptive_tt_weight=0.0,
            queue_adaptive_intensity_weight=0.0,
            ttl_bars=10,
        ),
    )
    assert float(sim["queue_frac_used"]) > 0.40
