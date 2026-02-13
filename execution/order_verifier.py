# execution/order_verifier.py — SCALPER ETERNAL — ORDER VERIFIER — 2026 v1.0
# Verifies order state after network timeouts/unknown responses.
#
# Features:
# - Async order status verification
# - Retry with exponential backoff
# - Reconciles SUBMITTED_UNKNOWN states
# - Updates intent ledger with verified state
#
# Design principles:
# - Never assume order state after timeout
# - Exchange is source of truth
# - Update all tracking systems on verification

from __future__ import annotations

import asyncio
import time
import os
from typing import Any, Dict, List, Optional, Tuple

from utils.logging import log_core, log_entry

try:
    from execution import intent_ledger as _intent_ledger
except Exception:
    _intent_ledger = None


# Configuration
_VERIFY_RETRIES = int(os.getenv("ORDER_VERIFY_RETRIES", "3"))
_VERIFY_TIMEOUT_SEC = float(os.getenv("ORDER_VERIFY_TIMEOUT_SEC", "10.0"))
_VERIFY_BACKOFF_BASE = float(os.getenv("ORDER_VERIFY_BACKOFF_BASE", "1.0"))
_VERIFY_ENABLED = os.getenv("ORDER_VERIFIER_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        cfg = getattr(bot, "cfg", None)
        return getattr(cfg, name, default) if cfg is not None else default
    except Exception:
        return default


def _normalize_symbol(sym: str) -> str:
    """Normalize symbol to canonical form."""
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "").replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _map_ccxt_status(status: str) -> str:
    """
    Map CCXT order status to intent ledger stage.

    CCXT statuses: open, closed, canceled, expired, rejected
    """
    status = str(status or "").lower().strip()
    mapping = {
        "open": "OPEN",
        "closed": "DONE",
        "canceled": "CANCELED",
        "cancelled": "CANCELED",
        "expired": "EXPIRED",
        "rejected": "REJECTED",
        "filled": "DONE",
        "partially_filled": "PARTIAL",
    }
    return mapping.get(status, "UNKNOWN")


class OrderVerifier:
    """
    Verifies order state from exchange.

    Usage:
        verifier = OrderVerifier(bot)

        # Verify single order
        result = await verifier.verify_order(
            order_id="123456",
            client_order_id="SE_ABC123",
            symbol="BTCUSDT",
            intent_id="intent_xyz",
        )

        # Verify all unknown orders
        results = await verifier.verify_unknown_orders()
    """

    def __init__(self, bot):
        self.bot = bot
        self._verification_lock = asyncio.Lock()

    async def verify_order(
        self,
        *,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: str,
        intent_id: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify order state from exchange.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            symbol: Trading symbol
            intent_id: Intent ID for ledger update

        Returns:
            Verification result dict:
            - verified: bool
            - status: str (CCXT status)
            - stage: str (intent ledger stage)
            - filled: float
            - remaining: float
            - order: dict (full order data)
            - error: str (if verification failed)
        """
        if not order_id and not client_order_id:
            return {"verified": False, "error": "No order ID provided"}

        ex = getattr(self.bot, "ex", None) or getattr(self.bot, "exchange", None)
        if ex is None:
            return {"verified": False, "error": "No exchange connection"}

        max_retries = retries if retries is not None else int(_safe_float(_cfg(self.bot, "ORDER_VERIFY_RETRIES", _VERIFY_RETRIES), _VERIFY_RETRIES))
        backoff_base = float(_safe_float(_cfg(self.bot, "ORDER_VERIFY_BACKOFF_BASE", _VERIFY_BACKOFF_BASE), _VERIFY_BACKOFF_BASE))

        # Resolve raw symbol
        sym_raw = self._resolve_raw_symbol(symbol)

        for attempt in range(max_retries):
            try:
                order = None

                # Try fetch by order ID first
                if order_id:
                    try:
                        fetch_fn = getattr(ex, "fetch_order", None)
                        if callable(fetch_fn):
                            order = await fetch_fn(order_id, sym_raw)
                    except Exception as e:
                        if "order does not exist" in str(e).lower():
                            # Order not found - might be already canceled/filled
                            pass
                        elif attempt == max_retries - 1:
                            raise

                # Try fetch by client order ID if no result
                if order is None and client_order_id:
                    try:
                        # Binance: fetch_orders with origClientOrderId
                        fetch_orders_fn = getattr(ex, "fetch_orders", None)
                        if callable(fetch_orders_fn):
                            orders = await fetch_orders_fn(sym_raw, limit=50)
                            for o in (orders or []):
                                if isinstance(o, dict):
                                    coid = o.get("clientOrderId") or o.get("info", {}).get("clientOrderId")
                                    if coid == client_order_id:
                                        order = o
                                        break
                    except Exception:
                        pass

                if order is None:
                    if attempt == max_retries - 1:
                        return {
                            "verified": False,
                            "error": "Order not found after retries",
                            "order_id": order_id,
                            "client_order_id": client_order_id,
                        }
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                    continue

                # Extract order info
                status = str(order.get("status") or "").lower()
                stage = _map_ccxt_status(status)
                filled = _safe_float(order.get("filled"), 0.0)
                remaining = _safe_float(order.get("remaining"), 0.0)
                average = _safe_float(order.get("average"), 0.0)

                result = {
                    "verified": True,
                    "status": status,
                    "stage": stage,
                    "filled": filled,
                    "remaining": remaining,
                    "average_price": average,
                    "order_id": str(order.get("id") or order_id or ""),
                    "client_order_id": str(order.get("clientOrderId") or client_order_id or ""),
                    "order": order,
                }

                # Update intent ledger if available
                if intent_id and _intent_ledger is not None:
                    try:
                        _intent_ledger.record(
                            self.bot,
                            intent_id=intent_id,
                            stage=stage,
                            symbol=symbol,
                            order_id=result["order_id"],
                            client_order_id=result["client_order_id"],
                            status=status,
                            reason="verified_from_exchange",
                            meta={
                                "filled": filled,
                                "remaining": remaining,
                                "average": average,
                                "verified_at": time.time(),
                            },
                        )
                    except Exception:
                        pass

                log_entry.info(
                    f"[order_verifier] VERIFIED: {_normalize_symbol(symbol)} | "
                    f"order_id={result['order_id']} | status={status} | "
                    f"filled={filled:.6f} | remaining={remaining:.6f}"
                )

                return result

            except Exception as e:
                if attempt == max_retries - 1:
                    log_entry.error(f"[order_verifier] FAILED: {symbol} | error={e}")
                    return {
                        "verified": False,
                        "error": str(e),
                        "order_id": order_id,
                        "client_order_id": client_order_id,
                    }

                await asyncio.sleep(backoff_base * (2 ** attempt))

        return {"verified": False, "error": "Max retries exceeded"}

    async def verify_unknown_orders(self) -> List[Dict[str, Any]]:
        """
        Verify all orders in SUBMITTED_UNKNOWN state.

        Returns:
            List of verification results
        """
        if _intent_ledger is None:
            return []

        results = []

        try:
            store = getattr(self.bot.state, "intent_ledger", None)
            if not isinstance(store, dict):
                return []

            intents = store.get("intents", {})
            unknown_stages = {"SUBMITTED_UNKNOWN", "CANCEL_SENT_UNKNOWN", "REPLACE_RACE", "OPEN_UNKNOWN"}

            for intent_id, rec in list(intents.items()):
                if not isinstance(rec, dict):
                    continue

                stage = str(rec.get("stage") or "").upper()
                if stage not in unknown_stages:
                    continue

                order_id = str(rec.get("order_id") or "").strip()
                client_order_id = str(rec.get("client_order_id") or "").strip()
                symbol = str(rec.get("symbol") or "").strip()

                if not symbol:
                    continue

                result = await self.verify_order(
                    order_id=order_id or None,
                    client_order_id=client_order_id or None,
                    symbol=symbol,
                    intent_id=intent_id,
                )
                result["intent_id"] = intent_id
                result["previous_stage"] = stage
                results.append(result)

                # Small delay between verifications
                await asyncio.sleep(0.1)

        except Exception as e:
            log_entry.error(f"[order_verifier] verify_unknown_orders failed: {e}")

        if results:
            verified_count = sum(1 for r in results if r.get("verified"))
            log_core.info(
                f"[order_verifier] Verified {verified_count}/{len(results)} unknown orders"
            )

        return results

    def _resolve_raw_symbol(self, symbol: str) -> str:
        """Resolve raw CCXT symbol from canonical."""
        k = _normalize_symbol(symbol)
        try:
            raw_map = getattr(getattr(self.bot, "data", None), "raw_symbol", {}) or {}
            if isinstance(raw_map, dict) and raw_map.get(k):
                return str(raw_map[k])
        except Exception:
            pass

        # Fallback: try CCXT format
        if "/" not in symbol and "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT:USDT"

        return symbol


# Module-level convenience functions
async def verify_order(
    bot,
    *,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    symbol: str,
    intent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify single order state from exchange."""
    if not _VERIFY_ENABLED:
        return {"verified": False, "error": "Verifier disabled"}
    return await OrderVerifier(bot).verify_order(
        order_id=order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        intent_id=intent_id,
    )


async def verify_unknown_orders(bot) -> List[Dict[str, Any]]:
    """Verify all orders in unknown state."""
    if not _VERIFY_ENABLED:
        return []
    return await OrderVerifier(bot).verify_unknown_orders()


async def verification_tick(bot) -> None:
    """
    Periodic verification tick - call from guardian/reconcile loop.

    Verifies unknown orders every N seconds.
    """
    if not _VERIFY_ENABLED:
        return

    try:
        # Check cooldown
        state = getattr(bot, "state", None)
        if state is None:
            return

        rc = getattr(state, "run_context", None)
        if not isinstance(rc, dict):
            try:
                state.run_context = {}
                rc = state.run_context
            except Exception:
                rc = {}

        now = time.time()
        last_verify = _safe_float(rc.get("_last_order_verify_ts"), 0.0)
        verify_interval = float(_safe_float(_cfg(bot, "ORDER_VERIFY_INTERVAL_SEC", 60.0), 60.0))

        if now - last_verify < verify_interval:
            return

        rc["_last_order_verify_ts"] = now

        # Run verification
        await verify_unknown_orders(bot)

    except Exception as e:
        log_entry.error(f"[order_verifier] verification_tick failed: {e}")
