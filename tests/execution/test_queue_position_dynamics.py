from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.microphys.execution.queue_position import QueuePositionParams, simulate_maker_queue_position_fill
from src.microphys.execution.queue_sim import QueueSimParams, simulate_maker_queue_fill


def _frame(n: int = 30, qty_sum: float = 30.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_qty": [100.0] * n,
            "ask_qty": [100.0] * n,
            "qty_sum": [qty_sum] * n,
            "trade_through_prob": [1.0] * n,
        }
    )


def test_queue_position_join_rate_slows_fill() -> None:
    df = _frame()
    fast = simulate_maker_queue_position_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueuePositionParams(
            initial_queue_frac=0.4,
            same_side_join_rate=0.0,
            same_side_cancel_rate=0.10,
            opposite_flow_scale=1.0,
            ttl_bars=15,
        ),
    )
    slow = simulate_maker_queue_position_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueuePositionParams(
            initial_queue_frac=0.4,
            same_side_join_rate=0.30,
            same_side_cancel_rate=0.00,
            opposite_flow_scale=1.0,
            ttl_bars=15,
        ),
    )
    assert bool(fast["filled"]) is True
    assert int(fast["fill_delay_bars"]) <= int(slow["fill_delay_bars"]) if slow["filled"] else True


def test_queue_sim_v2_mode_routes_to_position_model() -> None:
    df = _frame(n=20, qty_sum=40.0)
    out = simulate_maker_queue_fill(
        df,
        entry_idx=0,
        side="buy",
        params=QueueSimParams(queue_mode="position_v2", queue_frac=0.3, ttl_bars=10),
    )
    assert str(out.get("queue_model")) == "position_v2"
    assert "queue_total_consume" in out

