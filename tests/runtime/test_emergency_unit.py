"""Unit tests for execution/emergency.py

Covers:
- emergency_flat: flatten logic, position closing, order cancellation
- emergency_tick: flag-driven invocation
- Forced escalation (second wave without reduceOnly)
- Partial flatten (some symbols fail)
- _extract helpers
- _cancel_open_orders_best_effort
- _verify_remaining
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.emergency import (
    _extract_symbol,
    _extract_position_side,
    _extract_ccxt_position_amt,
    _signed_to_close_order,
    _safe_float,
    emergency_flat,
    emergency_tick,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_exchange(positions=None, open_orders=None):
    ex = AsyncMock()
    ex.fetch_positions = AsyncMock(return_value=positions or [])
    ex.fetch_open_orders = AsyncMock(return_value=open_orders or [])
    return ex


def _mock_bot(positions=None, ex=None):
    state_positions = {}
    if positions:
        for sym, side, size in positions:
            state_positions[sym] = SimpleNamespace(
                side=side, size=size, entry_price=100.0,
            )
    state = SimpleNamespace(
        positions=state_positions,
        current_equity=1000.0,
        last_exit_time={},
        emergency_flag=False,
        emergency_reason="",
        emergency_forced=False,
        run_context={},
    )
    bot = SimpleNamespace(
        state=state,
        cfg=SimpleNamespace(
            EMERGENCY_ALWAYS_FLAT=False,
        ),
        ex=ex or _mock_exchange(),
        notify=None,
        session_peak_equity=1000.0,
        data=SimpleNamespace(raw_symbol={}),
    )
    return bot


# ---------------------------------------------------------------------------
# 1. _extract helpers
# ---------------------------------------------------------------------------

def test_extract_symbol_from_dict():
    assert _extract_symbol({"symbol": "BTCUSDT"}) == "BTCUSDT"


def test_extract_symbol_from_info():
    assert _extract_symbol({"info": {"symbol": "BTCUSDT"}}) == "BTCUSDT"


def test_extract_symbol_empty():
    assert _extract_symbol({}) == ""


def test_extract_position_side_long():
    assert _extract_position_side({"info": {"positionSide": "LONG"}}) == "LONG"


def test_extract_position_side_from_side():
    assert _extract_position_side({"side": "long"}) == "LONG"
    assert _extract_position_side({"side": "short"}) == "SHORT"


def test_extract_position_side_empty():
    assert _extract_position_side({}) == ""


def test_extract_ccxt_position_amt():
    assert _extract_ccxt_position_amt({"info": {"positionAmt": "0.5"}}) == 0.5
    assert _extract_ccxt_position_amt({"contracts": 1.5}) == 1.5
    assert _extract_ccxt_position_amt({}) == 0.0


def test_signed_to_close_order():
    assert _signed_to_close_order(0.5) == "sell"
    assert _signed_to_close_order(-0.5) == "buy"
    assert _signed_to_close_order(0.0) is None


# ---------------------------------------------------------------------------
# 2. _safe_float
# ---------------------------------------------------------------------------

def test_safe_float():
    assert _safe_float("3.14") == 3.14
    assert _safe_float("bad", 0.0) == 0.0
    assert _safe_float(None, 5.0) == 5.0
    assert _safe_float(float("nan"), 1.0) == 1.0


# ---------------------------------------------------------------------------
# 3. emergency_tick — no flag
# ---------------------------------------------------------------------------

def test_emergency_tick_noop():
    bot = _mock_bot()
    _run(emergency_tick(bot))
    # No emergency actions taken
    bot.ex.fetch_positions.assert_not_called()


# ---------------------------------------------------------------------------
# 4. emergency_tick — flag set
# ---------------------------------------------------------------------------

@patch("execution.emergency.emergency_flat", new_callable=AsyncMock)
def test_emergency_tick_with_flag(mock_flat):
    bot = _mock_bot()
    bot.state.emergency_flag = True
    bot.state.emergency_reason = "manual trigger"
    _run(emergency_tick(bot))
    mock_flat.assert_called_once()
    # Flag should be cleared
    assert bot.state.emergency_flag is False


# ---------------------------------------------------------------------------
# 5. emergency_tick — EMERGENCY_ALWAYS_FLAT
# ---------------------------------------------------------------------------

@patch("execution.emergency.emergency_flat", new_callable=AsyncMock)
def test_emergency_tick_always_flat(mock_flat):
    bot = _mock_bot()
    bot.cfg.EMERGENCY_ALWAYS_FLAT = True
    _run(emergency_tick(bot))
    mock_flat.assert_called_once()


# ---------------------------------------------------------------------------
# 6. emergency_flat — no positions
# ---------------------------------------------------------------------------

@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_no_positions(mock_save):
    bot = _mock_bot()
    _run(emergency_flat(bot, reason="test"))
    # No orders to cancel or close
    bot.ex.fetch_positions.assert_called()
    mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# 7. emergency_flat — closes live positions
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock)
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_closes_positions(mock_save, mock_cancel, mock_create):
    live_positions = [
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "info": {"positionAmt": "0.01", "positionSide": "LONG"},
        }
    ]
    ex = _mock_exchange(positions=live_positions)
    # After flatten, verify returns empty
    ex.fetch_positions.side_effect = [live_positions, []]

    bot = _mock_bot(
        positions=[("BTCUSDT", "long", 0.01)],
        ex=ex,
    )
    _run(emergency_flat(bot, reason="test close"))
    # create_order should be called to close position
    assert mock_create.call_count >= 1
    call_kwargs = mock_create.call_args
    assert call_kwargs[1]["side"] == "sell"  # close long = sell


# ---------------------------------------------------------------------------
# 8. emergency_flat — handles failed close gracefully
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock, side_effect=Exception("exchange down"))
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_handles_failure(mock_save, mock_cancel, mock_create):
    live_positions = [
        {
            "symbol": "ETHUSDT",
            "side": "short",
            "info": {"positionAmt": "-0.5", "positionSide": "SHORT"},
        }
    ]
    ex = _mock_exchange(positions=live_positions)
    # fetch_positions called multiple times
    ex.fetch_positions.return_value = live_positions

    bot = _mock_bot(
        positions=[("ETHUSDT", "short", 0.5)],
        ex=ex,
    )
    # Should NOT raise even though create_order fails
    _run(emergency_flat(bot, reason="test failure"))
    mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# 9. emergency_flat — forced escalation
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock)
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_forced(mock_save, mock_cancel, mock_create):
    remaining_pos = {
        "symbol": "BTCUSDT",
        "side": "long",
        "info": {"positionAmt": "0.01", "positionSide": "LONG"},
    }
    ex = _mock_exchange()
    # 1st call: live positions, 2nd: verification (still has), 3rd: forced re-fetch, 4th: final verify
    ex.fetch_positions.side_effect = [
        [remaining_pos],  # initial
        [remaining_pos],  # verify after first wave
        [remaining_pos],  # forced re-fetch
        [],               # final verify
    ]
    bot = _mock_bot(ex=ex)
    _run(emergency_flat(bot, reason="forced test", forced=True))
    # Should have created orders for both regular and forced passes
    assert mock_create.call_count >= 1


# ---------------------------------------------------------------------------
# 10. emergency_flat — state positions fallback
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock)
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_state_fallback(mock_save, mock_cancel, mock_create):
    """When live positions are empty but state has positions, use state as hint."""
    ex = _mock_exchange(positions=[])
    # fetch_positions returns empty
    ex.fetch_positions.return_value = []

    bot = _mock_bot(
        positions=[("SOLUSDT", "long", 1.0)],
        ex=ex,
    )
    _run(emergency_flat(bot, reason="state fallback test"))
    # Should try to close based on state hint
    assert mock_create.call_count >= 1


# ---------------------------------------------------------------------------
# 11. emergency_flat — clears state on success
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock)
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_clears_state(mock_save, mock_cancel, mock_create):
    ex = _mock_exchange(positions=[])
    ex.fetch_positions.return_value = []  # all clear

    bot = _mock_bot(
        positions=[("BTCUSDT", "long", 0.01)],
        ex=ex,
    )
    _run(emergency_flat(bot, reason="clear test"))
    # Positions should be cleared
    assert len(bot.state.positions) == 0


# ---------------------------------------------------------------------------
# 12. emergency_flat — sets cooldown lockout
# ---------------------------------------------------------------------------

@patch("execution.emergency.create_order", new_callable=AsyncMock)
@patch("execution.emergency.cancel_order", new_callable=AsyncMock)
@patch("execution.emergency.save_brain", new_callable=AsyncMock)
def test_emergency_flat_sets_lockout(mock_save, mock_cancel, mock_create):
    ex = _mock_exchange(positions=[])
    ex.fetch_positions.return_value = []

    bot = _mock_bot(
        positions=[("BTCUSDT", "long", 0.01)],
        ex=ex,
    )
    _run(emergency_flat(bot, reason="lockout test"))
    # last_exit_time should have entries
    assert "BTCUSDT" in bot.state.last_exit_time
    assert bot.state.last_exit_time["BTCUSDT"] > time.time()
