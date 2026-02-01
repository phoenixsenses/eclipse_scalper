"""
Base adapter interfaces for pluggable exchanges.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class ExchangeAdapter:
    name: str = "base"

    async def fetch_markets(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def fetch_balance(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def fetch_positions(self, symbols=None) -> Any:
        raise NotImplementedError

    async def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 20) -> Any:
        raise NotImplementedError

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> Any:
        raise NotImplementedError

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: Any,
        price: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError

    async def cancel_order(self, order_id: str, symbol: str) -> Any:
        raise NotImplementedError

    async def fetch_ticker(self, symbol: str) -> Any:
        raise NotImplementedError

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since=None,
        limit=None,
        params: Optional[dict] = None,
    ) -> Any:
        raise NotImplementedError

    async def fetch_funding_rate(self, symbol: str) -> float:
        raise NotImplementedError

    async def set_position_mode(self, hedged: bool = True) -> Any:
        raise NotImplementedError

    def price_to_precision(self, symbol: str, price: float) -> str:
        raise NotImplementedError

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        raise NotImplementedError

    async def health_check(self) -> None:
        if not self._ready.is_set():
            self._ready.set()
