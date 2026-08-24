"""
Coinbase Pro adapter for the pluggable exchange layer.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import ccxt.async_support as ccxt

from utils.logging import log_core

from .base import ExchangeAdapter


def _env_lookup(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else None


class CoinbaseAdapter(ExchangeAdapter):
    name = "coinbase"

    def __init__(self):
        super().__init__()
        self.exchange = self._create_exchange()
        self._markets_loaded = False
        self._closing = asyncio.Event()
        self._health_lock = asyncio.Lock()

    def _create_exchange(self):
        return ccxt.coinbasepro(
            {
                "apiKey": _env_lookup("COINBASE_API_KEY"),
                "secret": _env_lookup("COINBASE_API_SECRET"),
                "password": _env_lookup("COINBASE_API_PASSPHRASE"),
                "enableRateLimit": True,
                "timeout": 30000,
            }
        )

    async def _ensure_markets(self):
        if self._markets_loaded:
            return
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True
        except Exception as e:
            log_core.warning(f"COINBASE MARKETS FAILED: {e}")

    async def _safe_request(self, method: str, *args, **kwargs):
        await self._ensure_markets()
        for attempt in range(3):
            try:
                fn = getattr(self.exchange, method)
                return await fn(*args, **kwargs)
            except Exception as e:
                log_core.warning(f"Coinbase request {method} failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Coinbase request {method} failed after retries")

    async def fetch_markets(self) -> Dict[str, Any]:
        return await self._safe_request("load_markets")

    async def fetch_balance(self) -> Dict[str, Any]:
        return await self._safe_request("fetch_balance")

    async def fetch_positions(self, symbols=None) -> Any:
        return await self._safe_request("fetch_positions", symbols or [])

    async def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 20) -> Any:
        if not symbol:
            return []
        return await self._safe_request("fetch_closed_orders", symbol, limit)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> Any:
        if not symbol:
            return []
        return await self._safe_request("fetch_open_orders", symbol)

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: Any,
        price: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kwargs = params or {}
        if type.lower() == "limit" and price is not None:
            return await self._safe_request("create_order", symbol, type.lower(), side.lower(), amount, price, kwargs)
        return await self._safe_request("create_order", symbol, type.lower(), side.lower(), amount, kwargs)

    async def cancel_order(self, order_id: str, symbol: str) -> Any:
        return await self._safe_request("cancel_order", order_id, {"symbol": symbol})

    async def fetch_ticker(self, symbol: str) -> Any:
        return await self._safe_request("fetch_ticker", symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since=None,
        limit=None,
        params: Optional[dict] = None,
    ) -> Any:
        return await self._safe_request("fetch_ohlcv", symbol, timeframe, since, limit, params or {})

    async def fetch_funding_rate(self, symbol: str) -> float:
        result = await self._safe_request("fetch_ticker", symbol)
        return float(result.get("info", {}).get("fundingRate", 0.0))

    async def set_position_mode(self, hedged: bool = True) -> Any:
        return {"code": 0, "msg": "coinbase does not support hedge mode"}

    def price_to_precision(self, symbol: str, price: float) -> str:
        try:
            return self.exchange.price_to_precision(symbol, price)
        except Exception:
            return f"{float(price):.8f}"

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        try:
            return self.exchange.amount_to_precision(symbol, amount)
        except Exception:
            return f"{float(amount):.8f}"

    async def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        try:
            await self.exchange.close()
        except Exception:
            pass

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        return symbol
