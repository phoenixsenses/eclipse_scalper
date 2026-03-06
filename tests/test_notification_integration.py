from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

try:
    from notifications.events import NotificationEvent, NotificationSeverity
    from notifications.health_alerts import build_crash_event
    from notifications.manager import NotificationConfig, NotificationManager
    from notifications.risk_alerts import build_entry_blocked_event, build_regime_change_event
    from notifications.trade_alerts import build_entry_event, build_exit_event, build_scratch_event
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from notifications.events import NotificationEvent, NotificationSeverity
    from notifications.health_alerts import build_crash_event
    from notifications.manager import NotificationConfig, NotificationManager
    from notifications.risk_alerts import build_entry_blocked_event, build_regime_change_event
    from notifications.trade_alerts import build_entry_event, build_exit_event, build_scratch_event


class _DummyNotifier:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def speak(self, text: str, priority: str = "critical", silent: bool = False):
        self.sent.append({"text": text, "priority": priority, "silent": bool(silent)})


def test_integration_entry_exit_scratch_risk_regime_startup() -> None:
    async def _run() -> None:
        dn = _DummyNotifier()
        nm = NotificationManager(dn, NotificationConfig(enabled=True, max_messages_per_minute=100))
        events = [
            build_entry_event(
                symbol="ETHUSDT",
                side="sell",
                entry_price=2450.0,
                regime="UP",
                regime_age_sec=60.0,
                rolling_return=0.001,
            ),
            build_exit_event(
                symbol="ETHUSDT",
                side="sell",
                pnl_bps=1.0,
                exit_type="HORIZON_EXIT",
                hold_sec=120.0,
                peak_bps=1.2,
                adverse_bps=-0.4,
            ),
            build_scratch_event(
                symbol="ETHUSDT",
                side="sell",
                pnl_bps=-4.2,
                adverse_limit_bps=5.0,
                elapsed_sec=34.0,
            ),
            build_entry_blocked_event("daily_loss_limit", {"daily_pnl_bps": -48.2, "daily_trades": 34, "config": {"max_daily_loss_bps": 50.0, "max_daily_trades": 100}}),
            build_regime_change_event("UP", "DOWN", 300.0, 0),
            NotificationEvent(NotificationSeverity.SUCCESS, "startup", "STARTED", "paper"),
        ]
        for e in events:
            await nm.send(e)
        assert len(dn.sent) >= 6
        joined = "\n".join(x["text"] for x in dn.sent)
        assert "ENTRY" in joined
        assert "EXIT" in joined
        assert "SCRATCH" in joined
        assert "ENTRY BLOCKED" in joined
        assert "REGIME CHANGE" in joined

    asyncio.run(_run())


def test_crash_event_sends() -> None:
    async def _run() -> None:
        dn = _DummyNotifier()
        nm = NotificationManager(dn, NotificationConfig(enabled=True))
        evt = build_crash_event("ConnectionResetError", 3600.0)
        ok = await nm.send(evt)
        assert ok is True
        assert any("CRASH" in x["text"] for x in dn.sent)

    asyncio.run(_run())
