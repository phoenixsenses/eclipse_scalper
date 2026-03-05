from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BinanceVipTier:
    tier: int
    volume_30d_usd: float
    maker_bps: float
    taker_bps: float


# Conservative futures-like bps schedule approximation for research simulations.
BINANCE_VIP_SCHEDULE: List[BinanceVipTier] = [
    BinanceVipTier(0, 0.0, 2.0, 5.0),
    BinanceVipTier(1, 15_000_000.0, 1.6, 4.0),
    BinanceVipTier(2, 50_000_000.0, 1.4, 3.5),
    BinanceVipTier(3, 100_000_000.0, 1.2, 3.2),
    BinanceVipTier(4, 250_000_000.0, 1.0, 3.0),
    BinanceVipTier(5, 500_000_000.0, 0.8, 2.7),
    BinanceVipTier(6, 1_000_000_000.0, 0.6, 2.5),
]


def infer_vip_tier_from_30d_volume(volume_30d_usd: float) -> BinanceVipTier:
    v = float(max(0.0, volume_30d_usd))
    best = BINANCE_VIP_SCHEDULE[0]
    for tier in BINANCE_VIP_SCHEDULE:
        if v >= float(tier.volume_30d_usd):
            best = tier
        else:
            break
    return best


def estimate_30d_volume_from_daily(daily_volume_usd: float) -> float:
    return float(max(0.0, daily_volume_usd)) * 30.0


def estimate_fee_bps_for_daily_volume(daily_volume_usd: float) -> dict:
    v30 = estimate_30d_volume_from_daily(daily_volume_usd)
    tier = infer_vip_tier_from_30d_volume(v30)
    return {
        "tier": int(tier.tier),
        "volume_30d_usd": float(v30),
        "maker_bps": float(tier.maker_bps),
        "taker_bps": float(tier.taker_bps),
    }

