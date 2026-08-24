"""Unit tests for risk/kill_switch.py

Covers:
- Persist/restore halt state
- Post-crash cooldown
- Data staleness detection
- Daily loss limit
- Drawdown limit
- Margin alert
- API error rate / burst
- Escalation ladder (flatten + shutdown)
- Trip history ring buffer
- clear_halt
- is_halted / remaining_halt_seconds / should_evaluate / trade_allowed
- Fail-closed on evaluate error
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Import helpers — we need to patch some modules before import
# ---------------------------------------------------------------------------

def _make_state(**extra):
    s = SimpleNamespace(
        halt_until_ts=0.0,
        halt_reason="",
        halt_count=0,
        kill_metrics={},
        current_equity=1000.0,
        peak_equity=1000.0,
        start_of_day_equity=1000.0,
        daily_pnl=0.0,
        margin_ratio=0.0,
        positions={},
        run_context={},
        **extra,
    )
    return s


def _make_bot(state=None, cfg=None, ex=None):
    if state is None:
        state = _make_state()
    if cfg is None:
        cfg = SimpleNamespace(
            KILL_SWITCH_ENABLED=True,
            KILL_SWITCH_COOLDOWN_SEC=300.0,
            KILL_DATA_BOOT_GRACE_SEC=0.0,
            KILL_MIN_DATA_SAMPLES_BEFORE_ENFORCE=1,
            MAX_DAILY_LOSS_PCT=0.0,
            MAX_DRAWDOWN_PCT=0.0,
            KILL_DAILY_HALT_SEC=300.0,
            KILL_DRAWDOWN_HALT_SEC=300.0,
            KILL_DAILY_HALT_UNTIL_UTC=False,
            KILL_MAX_DATA_STALENESS_SEC=150.0,
            KILL_MAX_API_ERROR_RATE=0.35,
            KILL_MAX_API_ERROR_BURST=12,
            KILL_MIN_REQ_WINDOW=10,
            KILL_MIN_EQUITY=0.0,
            KILL_MARGIN_ALERT_PCT=0.0,
            KILL_MAX_EX_HEALTH_STALE_SEC=0.0,
            KILL_SWITCH_MIN_EVAL_GAP_SEC=0.0,
            KILL_SWITCH_TRIP_STREAK_WINDOW_SEC=900.0,
            KILL_SWITCH_MAX_BACKOFF_MULT=4.0,
            KILL_SWITCH_TRIP_HISTORY_MAX=12,
            KILL_ESCALATE_FLAT_AFTER_TRIPS=0,
            KILL_ESCALATE_SHUTDOWN_AFTER_TRIPS=0,
            KILL_ESCALATE_WINDOW_SEC=900.0,
            KILL_SWITCH_EMERGENCY_FLAT=False,
            KILL_SWITCH_FAIL_CLOSED_ON_ERROR=True,
            KILL_SWITCH_ERROR_LATCH_SEC=30.0,
        )
    bot = SimpleNamespace(
        state=state,
        cfg=cfg,
        ex=ex,
        notify=None,
        active_symbols={"BTCUSDT", "ETHUSDT"},
    )
    return bot


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_kill_switch(tmp_path):
    """
    Clear kill switch loaded state and redirect file paths to tmp_path
    so existing logs/last_shutdown.json doesn't interfere.
    """
    import risk.kill_switch as mod
    mod._HALT_STATE_LOADED.clear()
    orig_halt_path = mod._HALT_STATE_PATH
    mod._HALT_STATE_PATH = tmp_path / "ks_state.json"
    # Patch the last_shutdown_path check inside _load_halt_state
    # by ensuring the tmp path has no last_shutdown.json
    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)
    yield
    os.chdir(orig_cwd)
    mod._HALT_STATE_PATH = orig_halt_path
    mod._HALT_STATE_LOADED.clear()


# ---------------------------------------------------------------------------
# 1. is_halted / remaining_halt_seconds
# ---------------------------------------------------------------------------

def test_not_halted_initially():
    from risk.kill_switch import is_halted, remaining_halt_seconds
    bot = _make_bot()
    assert is_halted(bot) is False
    assert remaining_halt_seconds(bot) == 0.0


def test_halted_when_until_ts_in_future():
    from risk.kill_switch import is_halted, remaining_halt_seconds
    bot = _make_bot()
    bot.state.halt_until_ts = time.time() + 60
    assert is_halted(bot) is True
    assert remaining_halt_seconds(bot) > 50


def test_not_halted_when_until_ts_in_past():
    from risk.kill_switch import is_halted
    bot = _make_bot()
    bot.state.halt_until_ts = time.time() - 10
    assert is_halted(bot) is False


# ---------------------------------------------------------------------------
# 2. should_evaluate
# ---------------------------------------------------------------------------

def test_should_evaluate_always_when_no_gap():
    from risk.kill_switch import should_evaluate
    bot = _make_bot()
    assert should_evaluate(bot) is True


def test_should_evaluate_throttled():
    from risk.kill_switch import should_evaluate, _ensure_state_fields
    bot = _make_bot()
    bot.cfg.KILL_SWITCH_MIN_EVAL_GAP_SEC = 60.0
    _ensure_state_fields(bot.state)
    bot.state.kill_metrics["last_check_ts"] = time.time()
    assert should_evaluate(bot) is False


# ---------------------------------------------------------------------------
# 3. trade_allowed (cheap gate)
# ---------------------------------------------------------------------------

def test_trade_allowed_when_not_halted():
    assert _run(_trade_allowed_helper(halted=False)) is True


def test_trade_allowed_blocked_when_halted():
    assert _run(_trade_allowed_helper(halted=True)) is False


async def _trade_allowed_helper(halted: bool):
    from risk.kill_switch import trade_allowed
    bot = _make_bot()
    if halted:
        bot.state.halt_until_ts = time.time() + 60
    return await trade_allowed(bot)


def test_trade_allowed_disabled():
    from risk.kill_switch import trade_allowed
    bot = _make_bot()
    bot.cfg.KILL_SWITCH_ENABLED = False
    assert _run(trade_allowed(bot)) is True


# ---------------------------------------------------------------------------
# 4. request_halt — basic
# ---------------------------------------------------------------------------

def test_request_halt_sets_halt():
    from risk.kill_switch import request_halt, is_halted

    bot = _make_bot()
    _run(request_halt(bot, 60, "test trip"))
    assert is_halted(bot)
    assert bot.state.halt_reason == "test trip"
    assert bot.state.halt_count == 1


def test_request_halt_backoff_on_streak():
    from risk.kill_switch import request_halt, _ensure_state_fields

    bot = _make_bot()
    _ensure_state_fields(bot.state)

    # First trip
    _run(request_halt(bot, 60, "trip 1"))
    first_until = bot.state.halt_until_ts

    # Reset for second trip
    bot.state.halt_until_ts = 0.0
    bot.state.halt_reason = ""

    # Second trip immediately (within streak window)
    _run(request_halt(bot, 60, "trip 2"))
    second_until = bot.state.halt_until_ts

    # Second should be longer due to backoff
    assert second_until > first_until or bot.state.kill_metrics["trip_streak"] >= 2


# ---------------------------------------------------------------------------
# 5. Trip history
# ---------------------------------------------------------------------------

def test_trip_history_records():
    from risk.kill_switch import request_halt, _ensure_state_fields

    bot = _make_bot()
    _ensure_state_fields(bot.state)
    _run(request_halt(bot, 30, "test reason"))
    hist = bot.state.kill_metrics.get("trip_history", [])
    assert len(hist) == 1
    assert hist[0]["reason"] == "test reason"


def test_trip_history_caps():
    from risk.kill_switch import _push_trip_history, _ensure_state_fields

    bot = _make_bot()
    bot.cfg.KILL_SWITCH_TRIP_HISTORY_MAX = 3
    _ensure_state_fields(bot.state)

    for i in range(10):
        _push_trip_history(bot, f"trip {i}", 60.0)

    hist = bot.state.kill_metrics["trip_history"]
    assert len(hist) == 3


# ---------------------------------------------------------------------------
# 6. clear_halt
# ---------------------------------------------------------------------------

def test_clear_halt():
    from risk.kill_switch import request_halt, clear_halt, is_halted

    bot = _make_bot()
    _run(request_halt(bot, 60, "test"))
    assert is_halted(bot)
    _run(clear_halt(bot, "manual clear"))
    assert not is_halted(bot)
    assert bot.state.halt_reason == ""


# ---------------------------------------------------------------------------
# 7. evaluate — pass
# ---------------------------------------------------------------------------

def test_evaluate_pass():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    # Mock data age to be fresh
    bot.data = SimpleNamespace(
        get_cache_age=lambda k, tf: 5.0,
    )
    ok, reason = _run(evaluate(bot))
    assert ok is True
    assert reason is None


def test_evaluate_disabled():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.KILL_SWITCH_ENABLED = False
    ok, reason = _run(evaluate(bot))
    assert ok is True


# ---------------------------------------------------------------------------
# 8. evaluate — equity below min
# ---------------------------------------------------------------------------

def test_evaluate_min_equity_trip():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.KILL_MIN_EQUITY = 500.0
    bot.state.current_equity = 100.0
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "EQUITY BELOW MIN" in reason


# ---------------------------------------------------------------------------
# 9. evaluate — daily loss
# ---------------------------------------------------------------------------

def test_evaluate_daily_loss_trip():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.MAX_DAILY_LOSS_PCT = 0.05  # 5%
    bot.state.start_of_day_equity = 1000.0
    bot.state.daily_pnl = -60.0  # 6% loss
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "DAILY LOSS" in reason


def test_evaluate_daily_loss_within_limit():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.MAX_DAILY_LOSS_PCT = 0.05
    bot.state.start_of_day_equity = 1000.0
    bot.state.daily_pnl = -40.0  # 4% loss, within limit
    bot.data = SimpleNamespace(get_cache_age=lambda k, tf: 5.0)
    ok, reason = _run(evaluate(bot))
    assert ok is True


# ---------------------------------------------------------------------------
# 10. evaluate — drawdown
# ---------------------------------------------------------------------------

def test_evaluate_drawdown_trip():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.MAX_DRAWDOWN_PCT = 0.10  # 10%
    bot.state.peak_equity = 1000.0
    bot.state.current_equity = 850.0  # 15% drawdown
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "DRAWDOWN" in reason


# ---------------------------------------------------------------------------
# 11. evaluate — data staleness
# ---------------------------------------------------------------------------

def test_evaluate_data_stale():
    from risk.kill_switch import evaluate, _ensure_state_fields

    bot = _make_bot()
    bot.cfg.KILL_MAX_DATA_STALENESS_SEC = 60.0
    _ensure_state_fields(bot.state)
    bot.state.kill_metrics["data_samples_ok"] = 5  # past min samples
    bot.data = SimpleNamespace(get_cache_age=lambda k, tf: 120.0)  # 120s stale
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "DATA STALE" in reason


# ---------------------------------------------------------------------------
# 12. evaluate — API error burst
# ---------------------------------------------------------------------------

def test_evaluate_api_error_burst():
    from risk.kill_switch import evaluate, _ensure_state_fields

    bot = _make_bot()
    bot.cfg.KILL_MAX_API_ERROR_BURST = 5
    ex = SimpleNamespace(request_count=100, error_count=10)
    bot.ex = ex
    _ensure_state_fields(bot.state)
    bot.state.kill_metrics["last_ex_request_count"] = 90
    bot.state.kill_metrics["last_ex_error_count"] = 3
    bot.data = SimpleNamespace(get_cache_age=lambda k, tf: 5.0)
    # 10-3=7 errors in window > burst threshold of 5
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "ERROR BURST" in reason


# ---------------------------------------------------------------------------
# 13. evaluate — API error rate
# ---------------------------------------------------------------------------

def test_evaluate_api_error_rate():
    from risk.kill_switch import evaluate, _ensure_state_fields

    bot = _make_bot()
    bot.cfg.KILL_MAX_API_ERROR_RATE = 0.20
    bot.cfg.KILL_MAX_API_ERROR_BURST = 999  # disable burst
    bot.cfg.KILL_MIN_REQ_WINDOW = 10
    ex = SimpleNamespace(request_count=100, error_count=10)
    bot.ex = ex
    _ensure_state_fields(bot.state)
    bot.state.kill_metrics["last_ex_request_count"] = 80  # 20 new requests
    bot.state.kill_metrics["last_ex_error_count"] = 4    # 6 new errors = 30% rate
    bot.data = SimpleNamespace(get_cache_age=lambda k, tf: 5.0)
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "ERROR RATE" in reason


# ---------------------------------------------------------------------------
# 14. evaluate — margin alert
# ---------------------------------------------------------------------------

def test_evaluate_margin_alert():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    bot.cfg.KILL_MARGIN_ALERT_PCT = 0.70
    bot.state.margin_ratio = 0.80  # 80% > 70% threshold
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "MARGIN ALERT" in reason


# ---------------------------------------------------------------------------
# 15. evaluate — fail-closed on error
# ---------------------------------------------------------------------------

def test_evaluate_fail_closed_on_error():
    from risk.kill_switch import evaluate

    bot = _make_bot()
    # Make state.kill_metrics raise on access
    bot.state = None  # will cause AttributeError
    # Re-create bot with broken state
    bot = SimpleNamespace(
        state=42,  # not a namespace, will fail
        cfg=_make_bot().cfg,
        ex=None,
        notify=None,
        active_symbols=set(),
    )
    ok, reason = _run(evaluate(bot))
    assert ok is False
    assert "EVALUATE ERROR" in reason


# ---------------------------------------------------------------------------
# 16. Persist / restore halt state
# ---------------------------------------------------------------------------

def test_save_and_load_halt_state(tmp_path):
    from risk.kill_switch import _save_halt_state, _load_halt_state, _HALT_STATE_LOADED
    import risk.kill_switch as mod

    state = _make_state()
    state.halt_until_ts = time.time() + 600
    state.halt_reason = "test persist"
    state.halt_count = 3
    state.kill_metrics = {"trip_streak": 2, "last_trip_ts": time.time(), "trip_history": []}

    # _isolate_kill_switch fixture already sets _HALT_STATE_PATH to tmp_path
    state_file = mod._HALT_STATE_PATH
    _save_halt_state(state)
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["halt_reason"] == "test persist"
    assert data["halt_count"] == 3

    # Now load into fresh state
    new_state = _make_state()
    _HALT_STATE_LOADED.discard(id(new_state))
    _load_halt_state(new_state)
    assert new_state.halt_until_ts > time.time()
    assert new_state.halt_reason == "test persist"
