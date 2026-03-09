"""EXE-01: Order router idempotency invariant tests.

Verifies that:
- Submitting the same intent_id twice does NOT create duplicate orders
- Retry with same clientOrderId is idempotent
- correlation_id links related intents
- All tests use mocks; no live exchange required.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal bot/exchange stubs
# ---------------------------------------------------------------------------

def _make_exchange(create_order_side_effect=None):
    """Return a mock exchange with an async create_order."""
    ex = MagicMock()
    ex.id = "binance"
    ex.markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "contract": True,
            "swap": True,
            "future": False,
            "limits": {"amount": {"min": 0.001, "max": 1000}, "price": {"min": 1, "max": 999999}},
            "precision": {"amount": 3, "price": 2},
        }
    }
    ex.market = MagicMock(return_value=ex.markets["BTC/USDT:USDT"])
    ex.amount_to_precision = MagicMock(side_effect=lambda s, a: round(float(a), 3))
    ex.price_to_precision = MagicMock(side_effect=lambda s, p: round(float(p), 2))
    ex.fetch_ticker = AsyncMock(return_value={"bid": 50000.0, "ask": 50001.0, "last": 50000.5})

    order_result = {
        "id": "ORD-123",
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "buy",
        "amount": 0.01,
        "filled": 0.01,
        "status": "closed",
        "info": {},
    }
    if create_order_side_effect is not None:
        ex.create_order = AsyncMock(side_effect=create_order_side_effect)
    else:
        ex.create_order = AsyncMock(return_value=order_result)

    # Not dry run
    ex._is_dry_run = MagicMock(return_value=False)
    ex._dry_run_guard_installed = False

    # futures mode detection helpers
    ex.fapiPrivateGetPositionSideDual = AsyncMock(return_value={"dualSidePosition": False})
    ex.fapiPrivateGetMultiAssetsMargin = AsyncMock(return_value={"multiAssetsMargin": False})
    ex.set_leverage = AsyncMock(return_value=None)
    ex.set_margin_mode = AsyncMock(return_value=None)
    return ex


def _make_bot(ex=None):
    """Return a minimal bot stub suitable for order_router."""
    bot = types.SimpleNamespace()
    bot.ex = ex or _make_exchange()
    bot.exchange = bot.ex
    bot.cfg = types.SimpleNamespace()
    bot.cfg.LEVERAGE = 1
    bot.cfg.MARGIN_MODE = "cross"
    bot.cfg.FIRST_LIVE_SAFE = False
    bot.cfg.INTENT_LEDGER_ENABLED = True
    bot.cfg.INTENT_LEDGER_REUSE_ENABLED = True
    bot.cfg.INTENT_LEDGER_PATH = ""
    bot.cfg.EVENT_JOURNAL_PATH = ""
    bot.cfg.ROUTER_AUTO_CLIENT_ID = "1"
    bot.cfg.ROUTER_ATTEMPT_TIMEOUT_SEC = 5.0
    bot.cfg.ROUTER_RETRY_MAX_ELAPSED_SEC = 5.0
    bot.data = types.SimpleNamespace()
    bot.data.raw_symbol = {"BTCUSDT": "BTC/USDT:USDT"}
    bot.state = types.SimpleNamespace()
    bot.state.positions = {}
    bot.state.current_equity = 1000.0
    bot.state.run_context = {}
    bot.state.guard_knobs = {}
    return bot


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_duplicate_intent_id_reuses_existing_order():
    """Submitting the same correlation_id twice should skip the second exchange call
    when INTENT_LEDGER_REUSE_ENABLED is True and the first intent was acknowledged."""
    from execution import order_router, intent_ledger

    ex = _make_exchange()
    bot = _make_bot(ex)

    corr = "DEDUP-TEST-001"

    # First call creates the order
    r1 = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01, correlation_id=corr,
    ))
    assert r1 is not None
    first_call_count = ex.create_order.call_count

    # Manually record the intent as ACKED so reuse kicks in
    intent_ledger.record(
        bot, intent_id=corr, stage="ACKED", symbol="BTCUSDT",
        side="buy", order_type="MARKET", order_id="ORD-123",
    )

    # Second call with same correlation_id should reuse
    r2 = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01, correlation_id=corr,
    ))
    assert r2 is not None
    assert r2.get("info", {}).get("intent_ledger_reused") is True
    # Exchange was NOT called again
    assert ex.create_order.call_count == first_call_count


def test_same_client_order_id_reuses():
    """When the same clientOrderId is in the ledger at a reusable stage,
    the router should skip submission."""
    from execution import order_router, intent_ledger

    ex = _make_exchange()
    bot = _make_bot(ex)

    coid = "SE_BTC_M_BUY_X1"
    corr = "CORR-COID-001"

    # Seed the ledger with a reusable entry
    intent_ledger.record(
        bot, intent_id=corr, stage="OPEN", symbol="BTCUSDT",
        side="buy", order_type="MARKET", client_order_id=coid,
        order_id="ORD-456",
    )

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01, client_order_id=coid, correlation_id=corr,
    ))
    assert r is not None
    assert r.get("info", {}).get("intent_ledger_reused") is True
    assert ex.create_order.call_count == 0


def test_correlation_id_propagated_to_ledger():
    """correlation_id passed to create_order must appear in the intent ledger record."""
    from execution import order_router, intent_ledger

    ex = _make_exchange()
    bot = _make_bot(ex)
    # Disable reuse so the order actually goes through
    bot.cfg.INTENT_LEDGER_REUSE_ENABLED = False

    corr = "CORR-PROP-001"
    _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01, correlation_id=corr,
    ))

    rec = intent_ledger.get_intent(bot, corr)
    assert rec is not None, "Intent should exist in ledger"
    assert rec["intent_id"] == corr


def test_auto_correlation_id_generated_when_not_provided():
    """When no correlation_id is given, the router generates one deterministically."""
    from execution import order_router, intent_ledger

    ex = _make_exchange()
    bot = _make_bot(ex)
    bot.cfg.INTENT_LEDGER_REUSE_ENABLED = False

    _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01,
    ))

    # Check that at least one intent was recorded
    store = (bot.state.run_context.get("intent_ledger") or {}).get("intents", {})
    assert len(store) >= 1, "At least one intent should exist"
    # The auto-generated ID should be non-empty
    first_id = list(store.keys())[0]
    assert first_id and len(first_id) > 0


def test_binance_duplicate_error_does_not_create_second_order():
    """When Binance returns -4116 (duplicate clientOrderId), the router should
    retry with a freshened ID, but the total exchange calls should be bounded."""
    from execution import order_router

    call_count = 0
    order_result = {
        "id": "ORD-789",
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "buy",
        "amount": 0.01,
        "filled": 0.01,
        "status": "closed",
        "info": {},
    }

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("binance {\"code\":-4116,\"msg\":\"ClientOrderId is duplicated.\"}")
        return order_result

    ex = _make_exchange(create_order_side_effect=_side_effect)
    bot = _make_bot(ex)
    bot.cfg.INTENT_LEDGER_REUSE_ENABLED = False

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01, client_order_id="SE_DUP_TEST",
    ))
    # Should eventually succeed or at least be bounded
    assert call_count <= 20, f"Retry storm: {call_count} calls"


def test_idempotent_cancel_unknown_order():
    """cancel_order on an already-gone order should return True (idempotent)."""
    from execution import order_router

    ex = _make_exchange()
    ex.cancel_order = AsyncMock(side_effect=Exception(
        "binance {\"code\":-2011,\"msg\":\"Unknown order sent.\"}"
    ))
    bot = _make_bot(ex)

    result = _run(order_router.cancel_order(bot, "FAKE-ORDER-ID", "BTC/USDT:USDT"))
    assert result is True, "cancel of unknown order should be treated as success"


def test_correlation_id_links_entry_and_exit_intents():
    """Using the same correlation_id for entry and exit links them in the ledger."""
    from execution import intent_ledger

    bot = _make_bot()
    corr = "LIFECYCLE-001"

    intent_ledger.record(
        bot, intent_id=corr, stage="FILLED", symbol="BTCUSDT",
        side="buy", order_type="MARKET", order_id="ORD-E1",
    )
    intent_ledger.record(
        bot, intent_id=corr, stage="DONE", symbol="BTCUSDT",
        side="sell", order_type="MARKET", order_id="ORD-X1",
        is_exit=True, reason="take_profit",
    )

    rec = intent_ledger.get_intent(bot, corr)
    assert rec is not None
    assert rec["stage"] == "DONE"
    assert rec["order_id"] == "ORD-X1"


def test_resolve_intent_by_order_id():
    """resolve_intent_id should find an intent by exchange order_id."""
    from execution import intent_ledger

    bot = _make_bot()
    corr = "RESOLVE-001"

    intent_ledger.record(
        bot, intent_id=corr, stage="OPEN", symbol="BTCUSDT",
        side="buy", order_type="MARKET", order_id="EX-ORD-42",
    )

    found = intent_ledger.resolve_intent_id(bot, order_id="EX-ORD-42")
    assert found == corr


def test_resolve_intent_by_client_order_id():
    """resolve_intent_id should find an intent by client_order_id."""
    from execution import intent_ledger

    bot = _make_bot()
    corr = "RESOLVE-002"

    intent_ledger.record(
        bot, intent_id=corr, stage="OPEN", symbol="BTCUSDT",
        side="buy", order_type="MARKET", client_order_id="SE_BTC_XYZ",
    )

    found = intent_ledger.resolve_intent_id(bot, client_order_id="SE_BTC_XYZ")
    assert found == corr
