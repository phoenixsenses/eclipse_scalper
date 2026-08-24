from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeEvent:
    ts: float
    symbol: str
    price: float
    qty: float
    side: str


@dataclass(frozen=True)
class TopOfBookEvent:
    ts: float
    symbol: str
    bid_px: float
    bid_qty: float
    ask_px: float
    ask_qty: float


@dataclass(frozen=True)
class LiquidationEvent:
    ts: float
    symbol: str
    side: str
    qty: float
    price: float

