"""
Exchange adapter registry.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Type

from .base import ExchangeAdapter
from .binance import BinanceCosmicAdapter
from .coinbase import CoinbaseAdapter
from .mock import MockExchangeAdapter

ADAPTERS: Dict[str, Type[ExchangeAdapter]] = {
    "binance": BinanceCosmicAdapter,
    "coinbase": CoinbaseAdapter,
    "mock": MockExchangeAdapter,
}


def register_adapter(name: str, adapter: Type[ExchangeAdapter]) -> None:
    ADAPTERS[name.lower()] = adapter


def get_exchange(name: str | None = None) -> ExchangeAdapter:
    selected = (name or os.getenv("EXCHANGE_ADAPTER", "binance")).strip().lower()
    cls = ADAPTERS.get(selected)
    if cls is None:
        raise RuntimeError(f"Unknown exchange adapter {selected!r}")
    return cls()
