from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from notifications.daily_summary import compose_daily_summary
from notifications.events import NotificationEvent
from notifications.health_alerts import build_heartbeat_event
from notifications.telegram import Notifier


@dataclass(frozen=True)
class NotificationConfig:
    enabled: bool = True
    trade_alerts: bool = True
    risk_alerts: bool = True
    daily_summary: bool = True
    heartbeat_enabled: bool = True
    heartbeat_interval_sec: float = 14400.0
    daily_summary_utc_hour: int = 0
    daily_summary_utc_minute: int = 5
    max_messages_per_minute: int = 20
    silent_heartbeat: bool = True
    send_timeout_sec: float = 2.0


def load_config_from_env() -> NotificationConfig:
    def b(name: str, default: bool) -> bool:
        v = str(os.getenv(name, "1" if default else "0")).strip().lower()
        return v in ("1", "true", "yes", "on")

    def i(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, str(default))))
        except Exception:
            return int(default)

    def f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    return NotificationConfig(
        enabled=b("NOTIFY_ENABLED", True),
        trade_alerts=b("NOTIFY_TRADE_ALERTS", True),
        risk_alerts=b("NOTIFY_RISK_ALERTS", True),
        daily_summary=b("NOTIFY_DAILY_SUMMARY", True),
        heartbeat_enabled=b("NOTIFY_HEARTBEAT", True),
        heartbeat_interval_sec=f("NOTIFY_HEARTBEAT_INTERVAL_SEC", 14400.0),
        max_messages_per_minute=i("NOTIFY_MAX_MSG_PER_MIN", 20),
        silent_heartbeat=b("NOTIFY_HEARTBEAT_SILENT", True),
        send_timeout_sec=f("NOTIFY_SEND_TIMEOUT_SEC", 2.0),
    )


class NotificationManager:
    def __init__(self, notifier: Notifier, config: NotificationConfig):
        self.notifier = notifier
        self.config = config
        self._last_by_key: Dict[str, float] = {}
        self._sent_ts: Deque[float] = deque()
        self._pending: Deque[NotificationEvent] = deque()
        self._trade_burst_ts: Deque[float] = deque()
        self._batched_trade_count: int = 0
        self._last_trade_batch_flush_ts: float = 0.0
        self._last_heartbeat_ts: float = 0.0
        self._last_daily_key: Optional[Tuple[int, int, int]] = None

    async def send(self, event: NotificationEvent) -> bool:
        if not self.config.enabled:
            return False
        cat = str(event.category or "")
        if cat.startswith("trade_") and not self.config.trade_alerts:
            return False
        if cat.startswith("risk_") and not self.config.risk_alerts:
            return False
        now = time.time()
        if event.throttle_key and float(event.throttle_window_sec or 0.0) > 0:
            last = float(self._last_by_key.get(event.throttle_key, 0.0) or 0.0)
            if now - last < float(event.throttle_window_sec):
                return False
        while self._sent_ts and now - self._sent_ts[0] > 60.0:
            self._sent_ts.popleft()
        while self._pending and len(self._sent_ts) < int(max(1, self.config.max_messages_per_minute)):
            queued = self._pending.popleft()
            await self._send_now(queued, now=time.time())
            now = time.time()
            while self._sent_ts and now - self._sent_ts[0] > 60.0:
                self._sent_ts.popleft()
        if len(self._sent_ts) >= int(max(1, self.config.max_messages_per_minute)):
            self._pending.append(event)
            return False
        return await self._send_now(event, now=now)

    async def _send_now(self, event: NotificationEvent, now: float) -> bool:
        try:
            coro = self.notifier.speak(
                event.render(),
                priority=("critical" if event.category in ("crash", "drawdown_limit", "data_stale") else "normal"),
                silent=bool(event.silent),
            )
            timeout_sec = float(max(0.0, self.config.send_timeout_sec))
            if timeout_sec > 0.0:
                sent_ok = await asyncio.wait_for(coro, timeout=timeout_sec)
            else:
                sent_ok = await coro
            # Accept None as success for in-memory/dummy notifier implementations in tests.
            if sent_ok is False:
                return False
            self._sent_ts.append(now)
            if event.throttle_key:
                self._last_by_key[event.throttle_key] = now
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def send_trade_event(self, event: NotificationEvent) -> bool:
        now = time.time()
        while self._trade_burst_ts and now - self._trade_burst_ts[0] > 300.0:
            self._trade_burst_ts.popleft()
        self._trade_burst_ts.append(now)
        if len(self._trade_burst_ts) > 10:
            self._batched_trade_count += 1
            if now - self._last_trade_batch_flush_ts >= 300.0:
                self._last_trade_batch_flush_ts = now
                return await self.send(
                    NotificationEvent(
                        severity=event.severity,
                        category="trade_batch",
                        title="TRADE BATCH",
                        body=f"Suppressed {self._batched_trade_count} trade alerts in last 5m (rate protection).",
                    )
                )
            return False
        return await self.send(event)

    async def send_daily_summary(self, trades_db: str = "data/paper_trades.db", micro_db: str = "data/microstructure.db") -> bool:
        if not self.config.daily_summary:
            return False
        evt = compose_daily_summary(trades_db=trades_db, micro_db=micro_db)
        return await self.send(evt)

    async def send_heartbeat(self, bot: Any, started_ts: float) -> bool:
        if not self.config.heartbeat_enabled:
            return False
        evt = build_heartbeat_event(bot, started_ts=started_ts)
        if self.config.silent_heartbeat:
            evt = NotificationEvent(
                severity=evt.severity,
                category=evt.category,
                title=evt.title,
                body=evt.body,
                throttle_key=evt.throttle_key,
                throttle_window_sec=evt.throttle_window_sec,
                silent=True,
            )
        return await self.send(evt)

    async def maybe_emit_periodics(self, bot: Any, started_ts: float) -> None:
        now = time.time()
        if self.config.heartbeat_enabled and (now - float(self._last_heartbeat_ts)) >= float(self.config.heartbeat_interval_sec):
            ok = await self.send_heartbeat(bot, started_ts=started_ts)
            if ok:
                self._last_heartbeat_ts = now
        if self.config.daily_summary:
            t = time.gmtime(now)
            key = (t.tm_year, t.tm_mon, t.tm_mday)
            if (
                int(t.tm_hour) == int(self.config.daily_summary_utc_hour)
                and int(t.tm_min) == int(self.config.daily_summary_utc_minute)
                and key != self._last_daily_key
            ):
                ok = await self.send_daily_summary()
                if ok:
                    self._last_daily_key = key


def build_notifier_from_env() -> Optional[Notifier]:
    if os.getenv("PYTEST_CURRENT_TEST") and not _env_bool("NOTIFY_ALLOW_REAL_IN_TESTS"):
        return None
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
    if not token or not chat_id:
        return None
    return Notifier(token=token, chat_id=chat_id)


def get_notification_manager_from_bot(bot: Any) -> Optional[NotificationManager]:
    rc = getattr(getattr(bot, "state", None), "run_context", None)
    if not isinstance(rc, dict):
        return None
    nm = rc.get("notification_manager")
    if isinstance(nm, NotificationManager):
        return nm
    notifier = build_notifier_from_env()
    if notifier is None:
        return None
    nm = NotificationManager(notifier=notifier, config=load_config_from_env())
    rc["notification_manager"] = nm
    return nm
