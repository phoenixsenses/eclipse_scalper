from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

try:
    from core.trade_logger import TradeLogger
    from notifications.daily_summary import compose_daily_summary
    from notifications.events import NotificationEvent, NotificationSeverity
    from notifications.manager import NotificationConfig, NotificationManager
    from notifications.risk_alerts import build_entry_blocked_event
    from notifications.trade_alerts import build_entry_event, build_exit_event
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.trade_logger import TradeLogger
    from notifications.daily_summary import compose_daily_summary
    from notifications.events import NotificationEvent, NotificationSeverity
    from notifications.manager import NotificationConfig, NotificationManager
    from notifications.risk_alerts import build_entry_blocked_event
    from notifications.trade_alerts import build_entry_event, build_exit_event


class _DummyNotifier:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def speak(self, text: str, priority: str = "critical", silent: bool = False):
        self.sent.append({"text": text, "priority": priority, "silent": bool(silent)})


def test_event_formatting() -> None:
    e1 = build_entry_event(
        symbol="ETHUSDT",
        side="sell",
        entry_price=2451.3,
        regime="UP",
        regime_age_sec=120.0,
        rolling_return=0.0008,
    )
    e2 = build_exit_event(
        symbol="ETHUSDT",
        side="sell",
        pnl_bps=1.8,
        exit_type="HORIZON_EXIT",
        hold_sec=120.0,
        peak_bps=3.1,
        adverse_bps=-1.2,
    )
    assert "ENTRY" in e1.render()
    assert "EXIT" in e2.render()
    assert "ETHUSDT" in e1.render()


def test_throttle_logic() -> None:
    async def _run() -> None:
        dm = _DummyNotifier()
        nm = NotificationManager(dm, NotificationConfig(enabled=True, max_messages_per_minute=50))
        evt = build_entry_blocked_event("daily_loss_limit", {"daily_pnl_bps": -48.2, "daily_trades": 3, "config": {"max_daily_loss_bps": 50.0, "max_daily_trades": 100}})
        ok1 = await nm.send(evt)
        ok2 = await nm.send(evt)
        assert ok1 is True
        assert ok2 is False
        assert len(dm.sent) == 1

    asyncio.run(_run())


def test_rate_limiter() -> None:
    async def _run() -> None:
        dm = _DummyNotifier()
        nm = NotificationManager(dm, NotificationConfig(enabled=True, max_messages_per_minute=2))
        a = NotificationEvent(NotificationSeverity.INFO, "x", "A", "a")
        b = NotificationEvent(NotificationSeverity.INFO, "x", "B", "b")
        c = NotificationEvent(NotificationSeverity.INFO, "x", "C", "c")
        assert await nm.send(a)
        assert await nm.send(b)
        assert (await nm.send(c)) is False
        assert len(dm.sent) == 2
        assert len(nm._pending) == 1

    asyncio.run(_run())


def test_daily_summary_compose() -> None:
    db = Path("data") / f"test_notify_daily_{uuid.uuid4().hex}.db"
    try:
        lg = TradeLogger(str(db))
        lg.log_trade(
            {
                "trade_id": "n1",
                "entry_time": 1_700_000_000.0,
                "exit_time": 1_700_000_120.0,
                "side": "sell",
                "regime": "UP",
                "entry_price": 100.0,
                "exit_price": 99.9,
                "pnl_bps": 10.0,
                "exit_type": "HORIZON_EXIT",
                "exit_reason": "time",
                "elapsed_sec": 120.0,
            }
        )
        evt = compose_daily_summary(trades_db=str(db), micro_db="data/microstructure.db", day_index=1, day_total=60)
        text = evt.render()
        assert "DAILY SUMMARY" in text
        assert "Total trades" in text or "Total trades:" in text
    finally:
        db.unlink(missing_ok=True)


def test_disabled_mode_sends_nothing() -> None:
    async def _run() -> None:
        dm = _DummyNotifier()
        nm = NotificationManager(dm, NotificationConfig(enabled=False))
        ok = await nm.send(NotificationEvent(NotificationSeverity.INFO, "x", "OFF", "off"))
        assert ok is False
        assert dm.sent == []

    asyncio.run(_run())


def test_silent_mode_and_heartbeat_interval() -> None:
    async def _run() -> None:
        dm = _DummyNotifier()
        cfg = NotificationConfig(enabled=True, heartbeat_enabled=True, heartbeat_interval_sec=3600, silent_heartbeat=True)
        nm = NotificationManager(dm, cfg)
        bot = type("B", (), {})()
        bot.state = type("S", (), {"total_trades": 0, "total_wins": 0, "daily_pnl_bps": 0.0})()
        bot._notify_last_regime = "UP"
        bot._notify_last_rolling_return = 0.0
        await nm.maybe_emit_periodics(bot, started_ts=0.0)
        first = len(dm.sent)
        await nm.maybe_emit_periodics(bot, started_ts=0.0)
        second = len(dm.sent)
        assert first >= 1
        assert second == first
        assert dm.sent[0]["silent"] is True

    asyncio.run(_run())
