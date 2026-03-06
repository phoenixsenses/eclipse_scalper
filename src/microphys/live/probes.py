from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Probe:
    name: str
    condition: Dict[str, object]
    min_triggers_per_day: int = 10


@dataclass(frozen=True)
class DirectionalProbe:
    name: str
    condition: Dict[str, object]
    side: str  # buy | sell
    horizon_bars: int = 1


def default_probes() -> List[Probe]:
    return [
        Probe(name="ofi_abs_q95", condition={"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.95}, min_triggers_per_day=10),
        Probe(
            name="ofi_abs_q90_spread_q20",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.90},
                    {"type": "fn", "fn": "q_lt", "col": "spread_z", "q": 0.20},
                ],
            },
            min_triggers_per_day=10,
        ),
        Probe(name="intensity_q90", condition={"type": "fn", "fn": "q_gt", "col": "F_intensity_z", "q": 0.90}, min_triggers_per_day=10),
        Probe(name="compression_flag", condition={"type": "gt", "op": "gt", "left": "compression_flag", "right": 0.5}, min_triggers_per_day=3),
        Probe(name="vacuum_flag", condition={"type": "gt", "op": "gt", "left": "vacuum_flag", "right": 0.5}, min_triggers_per_day=1),
        Probe(name="liq_burst_flag", condition={"type": "gt", "op": "gt", "left": "liq_burst_flag", "right": 0.5}, min_triggers_per_day=1),
        Probe(name="spread_q10", condition={"type": "fn", "fn": "q_lt", "col": "spread_z", "q": 0.10}, min_triggers_per_day=10),
        Probe(
            name="ofi_q85_intensity_q80",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.85},
                    {"type": "fn", "fn": "q_gt", "col": "F_intensity_z", "q": 0.80},
                ],
            },
            min_triggers_per_day=10,
        ),
    ]


def default_directional_probes() -> List[DirectionalProbe]:
    return [
        DirectionalProbe(
            name="buy_ofi_abs_q95_pos",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.95},
                    {"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="buy",
            horizon_bars=1,
        ),
        DirectionalProbe(
            name="sell_ofi_abs_q95_neg",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.95},
                    {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="sell",
            horizon_bars=1,
        ),
        DirectionalProbe(
            name="buy_compression_ofi_pos",
            condition={
                "type": "and",
                "args": [
                    {"type": "gt", "op": "gt", "left": "compression_flag", "right": 0.5},
                    {"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="buy",
            horizon_bars=5,
        ),
        DirectionalProbe(
            name="sell_vacuum_ofi_neg",
            condition={
                "type": "and",
                "args": [
                    {"type": "gt", "op": "gt", "left": "vacuum_flag", "right": 0.5},
                    {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="sell",
            horizon_bars=5,
        ),
        DirectionalProbe(
            name="buy_intensity_q90_pos",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "q_gt", "col": "F_intensity_z", "q": 0.90},
                    {"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="buy",
            horizon_bars=1,
        ),
        DirectionalProbe(
            name="sell_intensity_q90_neg",
            condition={
                "type": "and",
                "args": [
                    {"type": "fn", "fn": "q_gt", "col": "F_intensity_z", "q": 0.90},
                    {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": 0.0},
                ],
            },
            side="sell",
            horizon_bars=1,
        ),
    ]
