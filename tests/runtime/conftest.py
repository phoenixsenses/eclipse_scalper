"""Shared fixtures for runtime tests."""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ---------------------------------------------------------------------------
# Event loop fixture (for older asyncio.get_event_loop usage)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Bot & state factories
# ---------------------------------------------------------------------------

@pytest.fixture
def make_state():
    """Factory fixture for bot state objects."""
    def _factory(**overrides):
        defaults = dict(
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
            emergency_flag=False,
            emergency_reason="",
            emergency_forced=False,
            last_exit_time={},
            total_trades=0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)
    return _factory


@pytest.fixture
def make_bot(make_state):
    """Factory fixture for bot objects."""
    def _factory(state=None, cfg=None, ex=None, **state_overrides):
        if state is None:
            state = make_state(**state_overrides)
        if cfg is None:
            cfg = SimpleNamespace(
                KILL_SWITCH_ENABLED=True,
                KILL_SWITCH_COOLDOWN_SEC=300.0,
                CIRCUIT_BREAKER_ENABLED=True,
                EVENT_JOURNAL_ENABLED=True,
            )
        return SimpleNamespace(
            state=state,
            cfg=cfg,
            ex=ex,
            notify=None,
            active_symbols={"BTCUSDT", "ETHUSDT"},
            data=SimpleNamespace(raw_symbol={}, ohlcv={}),
            session_peak_equity=1000.0,
        )
    return _factory
