from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class QueuePositionParams:
    initial_queue_frac: float = 0.25
    same_side_join_rate: float = 0.10
    same_side_cancel_rate: float = 0.08
    opposite_flow_scale: float = 1.0
    pressure_floor: float = 0.10
    ttl_bars: int = 10
    min_depth: float = 1.0


def _pressure(side: Literal["buy", "sell"], bid_qty: float, ask_qty: float) -> float:
    den = float(bid_qty + ask_qty)
    if den <= 0:
        return 0.5
    imb = float((bid_qty - ask_qty) / (den + EPS))
    return (1.0 - imb) * 0.5 if side == "buy" else (1.0 + imb) * 0.5


def simulate_maker_queue_position_fill(
    frame: pd.DataFrame,
    *,
    entry_idx: int,
    side: Literal["buy", "sell"],
    params: QueuePositionParams,
) -> dict:
    n = len(frame)
    if n == 0 or entry_idx < 0 or entry_idx >= n:
        return {"filled": False, "fill_idx": None, "fill_delay_bars": None, "ttl_expired": True}

    bid_qty = pd.to_numeric(frame.get("bid_qty"), errors="coerce").fillna(0.0).to_numpy()
    ask_qty = pd.to_numeric(frame.get("ask_qty"), errors="coerce").fillna(0.0).to_numpy()
    qty_sum = pd.to_numeric(frame.get("qty_sum"), errors="coerce").fillna(0.0).to_numpy()
    tt = pd.to_numeric(frame.get("trade_through_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()

    entry_depth = float(bid_qty[entry_idx] if side == "buy" else ask_qty[entry_idx])
    entry_depth = max(float(params.min_depth), entry_depth)
    qfrac = float(np.clip(float(params.initial_queue_frac), 0.001, 0.99))
    queue_ahead = float(entry_depth * qfrac)

    ttl = max(1, int(params.ttl_bars))
    stop = min(n - 1, entry_idx + ttl)
    total_join = 0.0
    total_cancel = 0.0
    total_consume = 0.0

    join_rate = max(0.0, float(params.same_side_join_rate))
    cancel_rate = max(0.0, float(params.same_side_cancel_rate))
    flow_scale = max(0.0, float(params.opposite_flow_scale))
    pressure_floor = max(0.0, float(params.pressure_floor))

    for i in range(entry_idx + 1, stop + 1):
        # Step proxy (1 bar): queue can grow from same-side joins.
        join = float(max(0.0, join_rate * (entry_depth * 0.01)))
        cancel = float(max(0.0, cancel_rate * (entry_depth * 0.01)))

        p = _pressure(side, float(bid_qty[i]), float(ask_qty[i]))
        p = max(pressure_floor, p)
        opposite_flow = float(max(0.0, qty_sum[i] * float(tt[i]) * flow_scale))
        consume = float(max(0.0, opposite_flow * p))

        queue_ahead += join
        queue_ahead -= cancel
        queue_ahead -= consume
        queue_ahead = max(0.0, queue_ahead)

        total_join += join
        total_cancel += cancel
        total_consume += consume

        if queue_ahead <= 0.0:
            return {
                "filled": True,
                "fill_idx": int(i),
                "fill_delay_bars": int(i - entry_idx),
                "ttl_expired": False,
                "queue_frac_used": float(qfrac),
                "queue_depth_entry": float(entry_depth),
                "queue_total_join": float(total_join),
                "queue_total_cancel": float(total_cancel),
                "queue_total_consume": float(total_consume),
                "queue_model": "position_v2",
            }
    return {
        "filled": False,
        "fill_idx": None,
        "fill_delay_bars": None,
        "ttl_expired": True,
        "queue_frac_used": float(qfrac),
        "queue_depth_entry": float(entry_depth),
        "queue_total_join": float(total_join),
        "queue_total_cancel": float(total_cancel),
        "queue_total_consume": float(total_consume),
        "queue_model": "position_v2",
    }

