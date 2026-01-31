# exchanges/binance.py — SCALPER ETERNAL — COSMIC EXCHANGE ORACLE ASCENDANT ABSOLUTE — 2026 v4.5 (FULL ENHANCED)
# Enhancements (vs v4.4):
# - Canonical<->Raw symbol resolver built from load_markets (BTCUSDT -> BTC/USDT:USDT)
# - All symbol-taking methods accept canonical OR raw, and route through resolver
# - price_to_precision / amount_to_precision also resolve symbols safely
# - fetch_closed_orders(None) / fetch_open_orders(None) safely return []
# - Time sync after markets load + on timestamp/nonce errors
# - Failover resets symbol maps + caches (prevents cross-account mixing)
# - Dry-run stubs shaped to not surprise downstream logic

import ccxt.async_support as ccxt
import asyncio
import time
import os
import random
from dotenv import load_dotenv
from typing import Dict, Optional, Any, Tuple

from utils.logging import log_core

load_dotenv()


def _env_truthy(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _symkey(sym: str) -> str:
    """
    Canonical key normalizer: 'BTC/USDT:USDT' -> 'BTCUSDT', 'BTC/USDT' -> 'BTCUSDT', 'BTCUSDT' -> 'BTCUSDT'
    """
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    return s


class CosmicExchangeOracle:
    TICKER_TTL = 2.0
    FUNDING_TTL = 30.0

    def __init__(self):
        self.api_keys = [{"key": os.getenv("BINANCE_API_KEY"), "secret": os.getenv("BINANCE_API_SECRET")}]
        self.current_account_idx = 0
        self.proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

        self.exchange = self._create_exchange()

        self.time_offset = 0
        self.last_health_check = 0.0
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()

        # caches keyed by canonical symkey
        self.funding_cache: Dict[str, Tuple[float, float]] = {}
        self.ticker_cache: Dict[str, Tuple[dict, float]] = {}

        # symbol resolver maps (built after load_markets)
        self._sym_to_raw: Dict[str, str] = {}   # canonical -> raw market symbol (e.g. BTCUSDT -> BTC/USDT:USDT)
        self._raw_to_sym: Dict[str, str] = {}   # raw -> canonical

        self.ws_task = None
        self.ws_connected = False

        self._closing = asyncio.Event()
        self._health_lock = asyncio.Lock()
        self._time_synced = False
        self._markets_loaded = False

    def _create_exchange(self):
        account = self.api_keys[self.current_account_idx]
        return ccxt.binanceusdm(
            {
                "apiKey": account.get("key"),
                "secret": account.get("secret"),
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                },
                "timeout": 30000,
                "proxies": {"http": self.proxy, "https": self.proxy} if self.proxy else None,
            }
        )

    def _is_dry_run(self) -> bool:
        return _env_truthy("SCALPER_DRY_RUN")

    # ---------- symbol resolution ----------

    def _build_symbol_maps_from_markets(self):
        """
        Build canonical<->raw resolver using exchange.markets keys.
        For Binance futures, keys are often like 'BTC/USDT:USDT'.
        """
        self._sym_to_raw.clear()
        self._raw_to_sym.clear()

        markets = getattr(self.exchange, "markets", None) or {}
        if not isinstance(markets, dict) or not markets:
            return

        for raw_sym, info in markets.items():
            try:
                # Only map USDT futures swaps if available in info
                # (still safe if info missing)
                k = _symkey(raw_sym)
                if not k:
                    continue
                self._sym_to_raw[k] = raw_sym
                self._raw_to_sym[raw_sym] = k
            except Exception:
                continue

    def _resolve_symbol(self, sym: Optional[str]) -> Optional[str]:
        """
        Accept canonical or raw.
        Returns a raw ccxt market symbol if possible, else returns input sym.
        """
        if sym is None:
            return None
        s = str(sym).strip()
        if not s:
            return sym

        # If it's already a known raw symbol, keep it
        if s in self._raw_to_sym:
            return s

        # Convert canonical -> raw if we can
        k = _symkey(s)
        raw = self._sym_to_raw.get(k)
        return raw or sym

    def _cache_key(self, sym: str) -> str:
        """
        Always canonicalize cache keys.
        """
        return _symkey(sym)

    async def _ensure_markets_loaded(self):
        if self._markets_loaded:
            return
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True
            self._build_symbol_maps_from_markets()
        except Exception as e:
            log_core.warning(f"Markets load failed (resolver may be weaker): {e}")

    # ---------- time sync / health ----------

    async def _sync_time(self):
        try:
            server_time = await self.exchange.fetch_time()
            self.time_offset = int(server_time) - int(time.time() * 1000)
            self._time_synced = True
            log_core.info(f"COSMIC TIME SYNCHRONIZED — offset {self.time_offset}ms")
        except Exception as e:
            log_core.warning(f"Time sync failed: {e}")

    async def _ensure_time_synced(self):
        if not self._time_synced:
            await self._sync_time()

    async def _health_check(self):
        await self._ensure_time_synced()

        now = time.time()
        if now - self.last_health_check <= 180:
            return

        async with self._health_lock:
            now = time.time()
            if now - self.last_health_check <= 180:
                return
            try:
                await self.exchange.fapiPublicGetPing()
                self.last_health_check = now
                log_core.debug("Health check passed — oracle eternal")
            except Exception as e:
                log_core.critical(f"HEALTH CHECK FAILED — {e}")
                await self._failover_account()

    async def _failover_account(self):
        if len(self.api_keys) <= 1:
            return

        self.current_account_idx = (self.current_account_idx + 1) % len(self.api_keys)
        log_core.critical(f"ACCOUNT FAILOVER — switching to account {self.current_account_idx + 1}")

        try:
            await self.exchange.close()
        except Exception:
            pass

        # Reset caches + maps so we don't mix accounts
        self.funding_cache.clear()
        self.ticker_cache.clear()
        self._sym_to_raw.clear()
        self._raw_to_sym.clear()
        self._markets_loaded = False
        self.last_health_check = 0.0
        self._time_synced = False

        self.exchange = self._create_exchange()
        await self._ensure_markets_loaded()
        await self._sync_time()

    # ---------- request wrapper ----------

    async def safe_request(self, method: str, *args, **kwargs):
        # Hard safety net
        if method == "create_order" and self._is_dry_run():
            symbol = args[0] if len(args) > 0 else kwargs.get("symbol", "UNKNOWN")
            type_ = args[1] if len(args) > 1 else kwargs.get("type", "UNKNOWN")
            side = args[2] if len(args) > 2 else kwargs.get("side", "UNKNOWN")
            amount = args[3] if len(args) > 3 else kwargs.get("amount", 0.0)
            price = args[4] if len(args) > 4 else kwargs.get("price", None)
            params = kwargs.get("params") or {}
            log_core.critical(
                f"DRY_RUN ORDER BLOCKED (SAFE_REQUEST) → {symbol} {type_} {side} amount={amount} price={price} params={params}"
            )
            return {
                "id": "DRY_RUN_ORDER",
                "symbol": symbol,
                "type": type_,
                "side": side,
                "amount": float(amount or 0.0),
                "filled": 0.0,
                "status": "canceled",
                "average": None,
                "reduceOnly": bool((params or {}).get("reduceOnly", False)),
                "info": {"dry_run": True, "blocked_in": "safe_request"},
            }

        await self._ensure_markets_loaded()
        await self._health_check()
        self.request_count += 1

        signed_methods = {
            "create_order",
            "fetch_balance",
            "fetch_positions",
            "fetch_closed_orders",
            "fetch_open_orders",
            "cancel_order",
        }
        if method in signed_methods:
            params = kwargs.setdefault("params", {})
            params.setdefault("timestamp", int(time.time() * 1000) + int(self.time_offset))

        for attempt in range(6):
            try:
                fn = getattr(self.exchange, method)
                return await fn(*args, **kwargs)

            except ccxt.AuthenticationError as e:
                self.error_count += 1
                log_core.critical(f"AUTHENTICATION FAILED: {e}")
                if len(self.api_keys) > 1:
                    await self._failover_account()
                else:
                    raise

            except ccxt.RateLimitExceeded:
                self.error_count += 1
                delay = 1.0 * (2**attempt) + random.uniform(0, 0.5)
                log_core.warning(f"Rate limit — cosmic pause {delay:.2f}s")
                await asyncio.sleep(delay)

            except ccxt.RequestTimeout:
                self.error_count += 1
                delay = 0.5 * (2**attempt) + random.uniform(0, 0.3)
                log_core.warning(f"Timeout — retry in {delay:.2f}s")
                await asyncio.sleep(delay)

            except ccxt.ExchangeError as e:
                self.error_count += 1
                error_str = str(e)
                if "Nonce" in error_str or "Timestamp" in error_str:
                    await self._sync_time()
                elif "Invalid Api-Key" in error_str:
                    raise
                delay = 0.3 * (2**attempt) + random.uniform(0, 0.2)
                log_core.warning(f"Exchange error: {e} — retry in {delay:.2f}s")
                await asyncio.sleep(delay)

            except Exception as e:
                self.error_count += 1
                delay = 0.5 * (2**attempt) + random.uniform(0, 0.3)
                log_core.error(f"Unknown error: {e} — retry in {delay:.2f}s")
                await asyncio.sleep(delay)

        raise Exception(f"COSMIC REQUEST FAILED AFTER 6 ATTEMPTS: {method}")

    # === Public API (stable) ===

    async def fetch_markets(self):
        """
        Loads markets and builds canonical<->raw resolver.
        Returns ccxt markets dict.
        """
        mkts = await self.safe_request("load_markets")
        self._markets_loaded = True
        self._build_symbol_maps_from_markets()
        await self._ensure_time_synced()
        return mkts

    async def fetch_balance(self):
        return await self.safe_request("fetch_balance")

    async def fetch_positions(self, symbols=None):
        await self._ensure_markets_loaded()
        # Resolve each symbol if list passed
        if isinstance(symbols, (list, tuple)):
            symbols = [self._resolve_symbol(s) for s in symbols if s]
        return await self.safe_request("fetch_positions", symbols)

    async def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 20):
        # Binance often requires a symbol; None returns []
        if symbol is None:
            return []
        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        return await self.safe_request("fetch_closed_orders", raw, limit=limit)

    async def fetch_open_orders(self, symbol: Optional[str] = None):
        # Binance often requires a symbol; None returns []
        if symbol is None:
            return []
        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        return await self.safe_request("fetch_open_orders", raw)

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: Any,
        price: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        p = params or {}

        if self._is_dry_run():
            log_core.critical(f"DRY_RUN ORDER BLOCKED → {symbol} {type} {side} amount={amount} price={price} params={p}")
            return {
                "id": "DRY_RUN_ORDER",
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": float(amount or 0.0) if str(amount).replace(".", "", 1).isdigit() else 0.0,
                "filled": 0.0,
                "status": "canceled",
                "average": None,
                "reduceOnly": bool(p.get("reduceOnly", False)),
                "info": {"dry_run": True},
            }

        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        return await self.safe_request("create_order", raw, type, side, amount, price, params=p)

    async def cancel_order(self, id_: str, symbol: str):
        if self._is_dry_run():
            log_core.info(f"DRY_RUN: would cancel_order id={id_} symbol={symbol}")
            return {"id": id_, "symbol": symbol, "status": "canceled", "info": {"dry_run": True}}
        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        return await self.safe_request("cancel_order", id_, raw)

    async def fetch_ticker(self, symbol: str):
        now = time.time()
        key = self._cache_key(symbol)
        cached = self.ticker_cache.get(key)
        if cached:
            val, ts = cached
            if now - ts <= self.TICKER_TTL:
                return dict(val)

        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        result = await self.safe_request("fetch_ticker", raw)
        if isinstance(result, dict):
            self.ticker_cache[key] = (dict(result), now)
        return result

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since=None,
        limit=None,
        params: Optional[dict] = None,
    ):
        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        return await self.safe_request("fetch_ohlcv", raw, timeframe, since, limit, params=params or {})

    async def fetch_funding_rate(self, symbol: str):
        now = time.time()
        key = self._cache_key(symbol)
        cached = self.funding_cache.get(key)
        if cached:
            rate, ts = cached
            if now - ts <= self.FUNDING_TTL:
                return float(rate)

        await self._ensure_markets_loaded()
        raw = self._resolve_symbol(symbol)
        result = await self.safe_request("fetch_funding_rate", raw)
        rate = float(result.get("fundingRate", 0.0) or 0.0) if isinstance(result, dict) else 0.0
        self.funding_cache[key] = (rate, now)
        return rate

    async def set_position_mode(self, hedged: bool = True):
        params = {"dualSidePosition": "true" if hedged else "false"}
        try:
            result = await self.exchange.fapiPrivatePostPositionSideDual(params)
            log_core.info("Hedge mode ascended")
            return result
        except ccxt.ExchangeError as e:
            error_str = str(e)
            if "-4059" in error_str or "No need to change" in error_str:
                log_core.info("Hedge mode already active — ascension complete")
                return {"code": 0, "msg": "Already set"}
            log_core.error(f"Hedge mode activation failed: {e}")
            raise

    def price_to_precision(self, symbol: str, price: float) -> str:
        raw = self._resolve_symbol(symbol) or symbol
        try:
            return self.exchange.price_to_precision(raw, price)
        except Exception:
            # last-resort fallback
            return f"{float(price):.8f}"

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        raw = self._resolve_symbol(symbol) or symbol
        try:
            return self.exchange.amount_to_precision(raw, amount)
        except Exception:
            return f"{float(amount):.8f}"

    async def close(self):
        if self._closing.is_set():
            return
        self._closing.set()

        if self.ws_task:
            try:
                self.ws_task.cancel()
            except Exception:
                pass

        try:
            await self.exchange.close()
        except Exception:
            pass

    async def _start_websocket(self):
        log_core.info("COSMIC WEBSOCKET COMMUNION — READY FOR ASCENSION (placeholder)")


def get_exchange() -> CosmicExchangeOracle:
    return CosmicExchangeOracle()
