from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from .queue_position import QueuePositionParams, simulate_maker_queue_position_fill

EPS = 1e-12


@dataclass(frozen=True)
class QueueSimParams:
    queue_frac: float = 0.25
    queue_mode: Literal["fixed", "adaptive", "position_v2", "v2"] = "fixed"
    queue_depth_threshold: float = 500.0
    queue_frac_shallow: float = 0.30
    queue_frac_deep: float = 0.10
    queue_frac_min: float = 0.01
    queue_frac_max: float = 0.95
    queue_adaptive_tt_weight: float = 0.25
    queue_adaptive_intensity_weight: float = 0.20
    queue_adaptive_intensity_norm: float = 100.0
    ttl_bars: int = 10
    min_depth: float = 1.0


def _resolve_queue_frac(
    *,
    params: QueueSimParams,
    depth: float,
    trade_through_prob: float,
    intensity_proxy: float,
) -> float:
    qmin = max(1e-3, float(params.queue_frac_min))
    qmax = max(qmin, float(params.queue_frac_max))
    mode = str(params.queue_mode).lower().strip()
    if mode != "adaptive":
        return float(np.clip(float(params.queue_frac), qmin, qmax))

    dthr = max(1e-6, float(params.queue_depth_threshold))
    deep_ratio = float(np.clip(float(depth) / dthr, 0.0, 1.0))
    q_shallow = float(params.queue_frac_shallow)
    q_deep = float(params.queue_frac_deep)
    qf = q_shallow + (q_deep - q_shallow) * deep_ratio

    tt = float(np.clip(float(trade_through_prob), 0.0, 1.0))
    tt_w = max(0.0, float(params.queue_adaptive_tt_weight))
    qf *= max(0.2, 1.0 - tt_w * tt)

    norm = max(1e-6, float(params.queue_adaptive_intensity_norm))
    intensity = max(0.0, float(intensity_proxy))
    intensity_score = intensity / (intensity + norm)
    iw = max(0.0, float(params.queue_adaptive_intensity_weight))
    qf *= max(0.2, 1.0 - iw * intensity_score)

    return float(np.clip(qf, qmin, qmax))


def simulate_maker_queue_fill(
    frame: pd.DataFrame,
    *,
    entry_idx: int,
    side: Literal["buy", "sell"],
    params: QueueSimParams,
) -> dict:
    n = len(frame)
    if n == 0 or entry_idx < 0 or entry_idx >= n:
        return {"filled": False, "fill_idx": None, "fill_delay_bars": None, "ttl_expired": True}

    mode = str(params.queue_mode).lower().strip()
    if mode in {"position_v2", "v2"}:
        qv2 = QueuePositionParams(
            initial_queue_frac=float(params.queue_frac),
            ttl_bars=int(params.ttl_bars),
            min_depth=float(params.min_depth),
        )
        return simulate_maker_queue_position_fill(frame, entry_idx=entry_idx, side=side, params=qv2)

    bid_qty = pd.to_numeric(frame.get("bid_qty"), errors="coerce").fillna(0.0).to_numpy()
    ask_qty = pd.to_numeric(frame.get("ask_qty"), errors="coerce").fillna(0.0).to_numpy()
    qty_sum = pd.to_numeric(frame.get("qty_sum"), errors="coerce").fillna(0.0).to_numpy()
    tt = pd.to_numeric(frame.get("trade_through_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()

    depth = float(bid_qty[entry_idx] if side == "buy" else ask_qty[entry_idx])
    depth = max(float(params.min_depth), depth)
    qfrac = _resolve_queue_frac(
        params=params,
        depth=depth,
        trade_through_prob=float(tt[entry_idx]),
        intensity_proxy=float(qty_sum[entry_idx]),
    )
    qpos = depth * qfrac
    ttl = max(1, int(params.ttl_bars))
    stop = min(n - 1, entry_idx + ttl)

    for i in range(entry_idx + 1, stop + 1):
        flow = float(qty_sum[i]) * (0.25 + 0.75 * float(tt[i]))
        # If side is buy, fills improve with sell pressure (low imbalance) and vice versa.
        imb = 0.0
        d = float(bid_qty[i] + ask_qty[i])
        if d > 0:
            imb = float((bid_qty[i] - ask_qty[i]) / (d + EPS))
        pressure = (1.0 - imb) * 0.5 if side == "buy" else (1.0 + imb) * 0.5
        consume = max(0.0, flow * max(0.1, pressure))
        qpos -= consume
        if qpos <= 0.0:
            return {
                "filled": True,
                "fill_idx": int(i),
                "fill_delay_bars": int(i - entry_idx),
                "ttl_expired": False,
                "queue_frac_used": float(qfrac),
                "queue_depth_entry": float(depth),
            }
    return {
        "filled": False,
        "fill_idx": None,
        "fill_delay_bars": None,
        "ttl_expired": True,
        "queue_frac_used": float(qfrac),
        "queue_depth_entry": float(depth),
    }
