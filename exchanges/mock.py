"""
Simple mock exchange adapter for testing/troubleshooting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ExchangeAdapter


class MockExchangeAdapter(ExchangeAdapter):
    name = "mock"

    async def fetch_markets(self) -> Dict[str, Any]:
        return {}

    async def fetch_balance(self) -> Dict[str, Any]:
        return {"total": {"USDT": 0.0}}

    async def fetch_positions(self, symbols=None) -> Any:
        return []

    async def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 20) -> Any:
        return []

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> Any:
        return []

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: Any,
        price: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return {"id": "MOCK", "symbol": symbol, "status": "open", "type": type, "side": side, "amount": amount, "price": price}

    async def cancel_order(self, order_id: str, symbol: str) -> Any:
        return {"id": order_id, "symbol": symbol, "status": "canceled"}

    async def fetch_ticker(self, symbol: str) -> Any:
        return {"symbol": symbol, "last": 0.0}

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since=None,
        limit=None,
        params: Optional[dict] = None,
    ) -> Any:
        return []

    async def fetch_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def set_position_mode(self, hedged: bool = True) -> Any:
        return {"code": 0, "msg": "mock"}

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{float(price):.8f}"

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{float(amount):.8f}"

    async def close(self) -> None:
        return None

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        return symbol
