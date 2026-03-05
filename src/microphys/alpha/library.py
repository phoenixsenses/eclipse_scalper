from __future__ import annotations

from typing import List

from .spec import SignalSpec


def _spec(
    name: str,
    side: str,
    condition: dict,
    *,
    entry: str = "market",
    horizon_bars: int = 10,
    cooldown_bars: int = 5,
    tags: List[str] | None = None,
) -> SignalSpec:
    return SignalSpec(
        name=name,
        side=side,
        condition=condition,
        entry=entry,
        horizon_bars=horizon_bars,
        cooldown_bars=cooldown_bars,
        meta={"tags": list(tags or [])},
    )


def built_in_signal_specs() -> List[SignalSpec]:
    specs: List[SignalSpec] = []
    for thr in (1.0, 1.5, 2.0):
        specs.append(
            _spec(
                f"ofi_buy_z{thr:.1f}",
                "buy",
                {"type": "fn", "fn": "z_gt", "col": "F_ofi_z", "thr": thr},
                tags=["ofi", "trend"],
            )
        )
        specs.append(
            _spec(
                f"ofi_sell_z{thr:.1f}",
                "sell",
                {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": -thr},
                tags=["ofi", "trend"],
            )
        )
        specs.append(
            _spec(
                f"compression_ofi_buy_z{thr:.1f}",
                "buy",
                {
                    "type": "and",
                    "args": [
                        {"type": "gte", "op": "gte", "left": "compression_flag", "right": 1},
                        {"type": "fn", "fn": "z_gt", "col": "F_ofi_z", "thr": thr},
                    ],
                },
                tags=["compression", "ofi"],
            )
        )
        specs.append(
            _spec(
                f"vacuum_ofi_sell_z{thr:.1f}",
                "sell",
                {
                    "type": "and",
                    "args": [
                        {"type": "gte", "op": "gte", "left": "vacuum_flag", "right": 1},
                        {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": -thr},
                    ],
                },
                tags=["vacuum", "ofi"],
            )
        )
    specs.extend(
        [
            _spec(
                "liq_burst_trend_sell",
                "sell",
                {
                    "type": "and",
                    "args": [
                        {"type": "gte", "op": "gte", "left": "liq_burst_flag", "right": 1},
                        {"type": "gt", "op": "gt", "left": "F_intensity_z", "right": 1.2},
                    ],
                },
                tags=["liquidation", "intensity"],
            ),
            _spec(
                "high_rv_spread_wide_control",
                "buy",
                {
                    "type": "and",
                    "args": [
                        {"type": "gt", "op": "gt", "left": "rv_z", "right": 1.0},
                        {"type": "gt", "op": "gt", "left": "spread_z", "right": 1.0},
                        {"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
                    ],
                },
                tags=["control"],
            ),
            _spec(
                "compression_contrarian_sell",
                "sell",
                {
                    "type": "and",
                    "args": [
                        {"type": "gte", "op": "gte", "left": "compression_flag", "right": 1},
                        {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": -1.0},
                    ],
                },
                tags=["compression", "contrarian"],
            ),
            _spec(
                "vacuum_contrarian_buy",
                "buy",
                {
                    "type": "and",
                    "args": [
                        {"type": "gte", "op": "gte", "left": "vacuum_flag", "right": 1},
                        {"type": "lt", "op": "lt", "left": "F_ofi_z", "right": -1.0},
                    ],
                },
                tags=["vacuum", "contrarian"],
            ),
        ]
    )
    return specs
