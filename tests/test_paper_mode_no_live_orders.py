"""SAF-02: Paper/dry-run mode safety invariant tests.

Verifies that:
- When dry_run=True or SCALPER_DRY_RUN=1, no actual exchange API calls are made
- The exchange adapter's create_order is never called in paper mode
- The dry-run guard installed by the runner blocks all order creation
- All tests use mocks; no live exchange required.
"""
from __future__ import annotations

import asyncio
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_exchange():
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
    ex._dry_run_guard_installed = False
    ex.fapiPrivateGetPositionSideDual = AsyncMock(return_value={"dualSidePosition": False})
    ex.fapiPrivateGetMultiAssetsMargin = AsyncMock(return_value={"multiAssetsMargin": False})
    ex.set_leverage = AsyncMock(return_value=None)
    ex.set_margin_mode = AsyncMock(return_value=None)

    # The REAL create_order that should NEVER be called in dry-run
    ex.create_order = AsyncMock(return_value={
        "id": "SHOULD-NEVER-APPEAR",
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "buy",
        "amount": 0.01,
        "filled": 0.01,
        "status": "closed",
        "info": {},
    })
    return ex


def _make_bot(ex=None):
    bot = types.SimpleNamespace()
    bot.ex = ex or _make_exchange()
    bot.exchange = bot.ex
    bot.cfg = types.SimpleNamespace()
    bot.cfg.LEVERAGE = 1
    bot.cfg.MARGIN_MODE = "cross"
    bot.cfg.FIRST_LIVE_SAFE = False
    bot.cfg.INTENT_LEDGER_ENABLED = True
    bot.cfg.INTENT_LEDGER_REUSE_ENABLED = False
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
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dry_run_env_blocks_exchange_call():
    """When SCALPER_DRY_RUN=1, create_order should return a stub
    without calling exchange.create_order."""
    from execution import order_router

    ex = _make_exchange()
    # Signal dry_run via the exchange method
    ex._is_dry_run = MagicMock(return_value=True)
    bot = _make_bot(ex)

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.01,
    ))
    assert r is not None
    assert r.get("info", {}).get("dry_run") is True
    assert ex.create_order.call_count == 0, "Exchange create_order must NOT be called in dry_run"


def test_dry_run_env_var_blocks_exchange_call():
    """When SCALPER_DRY_RUN env var is set to '1', the router should block."""
    from execution import order_router

    ex = _make_exchange()
    # No _is_dry_run method on exchange — rely on env var
    del ex._is_dry_run
    bot = _make_bot(ex)

    old_val = os.environ.get("SCALPER_DRY_RUN")
    os.environ["SCALPER_DRY_RUN"] = "1"
    try:
        r = _run(order_router.create_order(
            bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
            amount=0.01,
        ))
        assert r is not None
        assert r.get("info", {}).get("dry_run") is True
        assert ex.create_order.call_count == 0
    finally:
        if old_val is None:
            os.environ.pop("SCALPER_DRY_RUN", None)
        else:
            os.environ["SCALPER_DRY_RUN"] = old_val


def test_dry_run_stub_has_correct_fields():
    """The dry-run stub should contain symbol, type, side, amount, and info.dry_run."""
    from execution import order_router

    ex = _make_exchange()
    ex._is_dry_run = MagicMock(return_value=True)
    bot = _make_bot(ex)

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
        amount=0.05,
    ))
    assert r is not None
    assert r["side"] == "buy"
    assert r["type"] == "market"
    assert r["info"]["dry_run"] is True
    assert r["id"] is None, "Dry-run orders must NOT have a real order ID"
    assert r["status"] == "canceled"


def test_dry_run_stub_for_limit_order():
    """Limit orders should also be blocked in dry-run mode."""
    from execution import order_router

    ex = _make_exchange()
    ex._is_dry_run = MagicMock(return_value=True)
    bot = _make_bot(ex)

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="LIMIT", side="sell",
        amount=0.02, price=55000.0,
    ))
    assert r is not None
    assert r["info"]["dry_run"] is True
    assert r["side"] == "sell"
    assert ex.create_order.call_count == 0


def test_dry_run_stub_for_stop_market_order():
    """Stop-market orders should also be blocked in dry-run mode."""
    from execution import order_router

    ex = _make_exchange()
    ex._is_dry_run = MagicMock(return_value=True)
    bot = _make_bot(ex)

    r = _run(order_router.create_order(
        bot, symbol="BTC/USDT:USDT", type="STOP_MARKET", side="sell",
        amount=0.01, stop_price=48000.0,
        intent_reduce_only=True,
    ))
    assert r is not None
    assert r["info"]["dry_run"] is True
    assert ex.create_order.call_count == 0


def test_runner_dry_run_guard_blocks_create_order():
    """The _install_dry_run_guard function from bot/runner.py should replace
    exchange.create_order with a blocking stub."""
    from bot.runner import _install_dry_run_guard

    ex = _make_exchange()
    original_fn = ex.create_order
    bot = _make_bot(ex)

    _install_dry_run_guard(bot)

    assert ex._dry_run_guard_installed is True
    # The original should be preserved
    assert hasattr(ex, "_original_create_order")

    # Calling the guarded create_order should NOT call the original
    r = _run(ex.create_order(symbol="BTC/USDT:USDT", type="market", side="buy", amount=0.01))
    assert r is not None
    assert r.get("info", {}).get("dry_run") is True
    assert r.get("info", {}).get("runner_guard") is True
    # The ORIGINAL create_order should not have been invoked
    assert original_fn.call_count == 0


def test_runner_dry_run_guard_idempotent():
    """Installing the dry-run guard twice should NOT double-wrap."""
    from bot.runner import _install_dry_run_guard

    ex = _make_exchange()
    bot = _make_bot(ex)

    _install_dry_run_guard(bot)
    guard1 = ex.create_order
    _install_dry_run_guard(bot)
    guard2 = ex.create_order

    # Should be the same function — no double wrapping
    assert guard1 is guard2


def test_non_dry_run_calls_exchange():
    """When dry_run is NOT active, exchange.create_order MUST be called."""
    from execution import order_router

    ex = _make_exchange()
    ex._is_dry_run = MagicMock(return_value=False)
    bot = _make_bot(ex)

    old_val = os.environ.pop("SCALPER_DRY_RUN", None)
    try:
        r = _run(order_router.create_order(
            bot, symbol="BTC/USDT:USDT", type="MARKET", side="buy",
            amount=0.01,
        ))
        assert r is not None
        assert ex.create_order.call_count >= 1, "Exchange create_order MUST be called when not dry_run"
        assert r.get("info", {}).get("dry_run") is not True
    finally:
        if old_val is not None:
            os.environ["SCALPER_DRY_RUN"] = old_val


def test_dry_run_multiple_order_types_all_blocked():
    """Multiple order types in sequence should ALL be blocked in dry-run mode."""
    from execution import order_router

    ex = _make_exchange()
    ex._is_dry_run = MagicMock(return_value=True)
    bot = _make_bot(ex)

    order_specs = [
        {"type": "MARKET", "side": "buy", "amount": 0.01},
        {"type": "MARKET", "side": "sell", "amount": 0.02},
        {"type": "LIMIT", "side": "buy", "amount": 0.03, "price": 49000.0},
    ]

    for spec in order_specs:
        r = _run(order_router.create_order(
            bot, symbol="BTC/USDT:USDT",
            type=spec["type"], side=spec["side"],
            amount=spec["amount"], price=spec.get("price"),
        ))
        assert r is not None
        assert r["info"]["dry_run"] is True

    assert ex.create_order.call_count == 0, \
        f"Exchange called {ex.create_order.call_count} times in dry-run mode"
