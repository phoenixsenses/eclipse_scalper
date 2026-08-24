from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from notifications.daily_summary import compose_daily_summary
from notifications.events import NotificationEvent, NotificationSeverity
from notifications.health_alerts import build_heartbeat_event
from notifications.pentest_publisher import PentestPublisher
from notifications.telegram import Notifier
from notifications.webhook import WebhookSender, get_sender as get_webhook_sender

_FALLBACK_LOG = Path(os.environ.get("NOTIFY_FALLBACK_LOG", "logs/notifications_fallback.jsonl"))


def _fallback_log_event(event: NotificationEvent, reason: str) -> None:
    """Write notification to file when Telegram is unavailable."""
    try:
        _FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "category": str(event.category or ""),
            "severity": str(event.severity.value if hasattr(event.severity, "value") else event.severity),
            "title": str(event.title or ""),
            "body": str(event.body or "")[:500],
            "reason": str(reason or ""),
        }
        with open(_FALLBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
    silent_heartbeat: bool = False
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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class NotificationManager:
    def __init__(
        self,
        notifier: Notifier | None,
        config: NotificationConfig,
        pentest_publisher: PentestPublisher | None = None,
        webhook_sender: WebhookSender | None = None,
    ):
        self.notifier = notifier
        self.config = config
        self._pentest_publisher = pentest_publisher if pentest_publisher is not None else PentestPublisher()
        self._webhook_sender = webhook_sender
        self._last_by_key: Dict[str, float] = {}
        self._sent_ts: Deque[float] = deque()
        self._pending: Deque[NotificationEvent] = deque(maxlen=200)
        self._trade_burst_ts: Deque[float] = deque()
        self._batched_trade_count: int = 0
        self._last_trade_batch_flush_ts: float = 0.0
        self._last_heartbeat_ts: float = 0.0
        self._last_daily_key: Optional[Tuple[int, int, int]] = None
        # Alert escalation: same alert repeated → auto-escalate severity
        self._alert_repeat_count: Dict[str, int] = {}
        # Telegram failure circuit breaker
        self._consecutive_failures: int = 0
        self._tg_circuit_open: bool = False
        self._tg_circuit_open_ts: float = 0.0
        self._tg_dead_logged: bool = False

    async def send(self, event: NotificationEvent) -> bool:
        if not self.config.enabled:
            return False
        cat = str(event.category or "")
        if cat.startswith("trade_") and not self.config.trade_alerts:
            return False
        if cat.startswith("risk_") and not self.config.risk_alerts:
            return False
        event = self._maybe_escalate(event)
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

    _TG_CIRCUIT_THRESHOLD = 5  # consecutive failures to open circuit
    _TG_CIRCUIT_COOLDOWN_SEC = 120.0  # wait before retrying after circuit opens
    _ESCALATION_THRESHOLD = 3  # same alert N times → escalate severity
    _ESCALATION_MAX_TRACKED = 500  # cap on tracked alert keys

    _SEVERITY_LADDER = {
        NotificationSeverity.INFO: NotificationSeverity.WARNING,
        NotificationSeverity.WARNING: NotificationSeverity.CRITICAL,
    }

    def _maybe_escalate(self, event: NotificationEvent) -> NotificationEvent:
        """Auto-escalate severity if the same alert fires repeatedly."""
        esc_key = f"{event.category or ''}:{event.throttle_key or event.title or ''}"
        count = self._alert_repeat_count.get(esc_key, 0) + 1
        self._alert_repeat_count[esc_key] = count
        if len(self._alert_repeat_count) > self._ESCALATION_MAX_TRACKED:
            oldest = min(self._alert_repeat_count, key=self._alert_repeat_count.get)  # type: ignore[arg-type]
            self._alert_repeat_count.pop(oldest, None)
        if count >= self._ESCALATION_THRESHOLD:
            new_sev = self._SEVERITY_LADDER.get(event.severity)
            if new_sev is not None:
                return NotificationEvent(
                    severity=new_sev,
                    category=event.category,
                    title=f"[ESCALATED x{count}] {event.title}",
                    body=event.body,
                    symbol=event.symbol,
                    side=event.side,
                    source=event.source,
                    raw_payload=event.raw_payload,
                    throttle_key=event.throttle_key,
                    throttle_window_sec=event.throttle_window_sec,
                    silent=event.silent,
                )
        return event

    async def _send_now(self, event: NotificationEvent, now: float) -> bool:
        webhook_ok = await self._send_webhook(event)

        # Telegram circuit breaker: skip sends during cooldown
        if self._tg_circuit_open:
            if (now - self._tg_circuit_open_ts) < self._TG_CIRCUIT_COOLDOWN_SEC:
                _fallback_log_event(event, "telegram_circuit_open")
                self._publish_to_pentest(event)
                return webhook_ok
            # Cooldown expired, try again (half-open state)
            self._tg_circuit_open = False
            self._tg_dead_logged = False

        if self.notifier is None:
            _fallback_log_event(event, "no_notifier")
            self._publish_to_pentest(event)
            if webhook_ok:
                self._record_delivery_success(event, now)
            return webhook_ok

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
                self._record_tg_failure("telegram_returned_false", event)
                self._publish_to_pentest(event)
                if webhook_ok:
                    self._record_delivery_success(event, now)
                return webhook_ok
            # Success: reset failure counter
            self._consecutive_failures = 0
            self._record_delivery_success(event, now)
            self._publish_to_pentest(event)
            return True
        except asyncio.TimeoutError:
            self._record_tg_failure("telegram_timeout", event)
            self._publish_to_pentest(event)
            if webhook_ok:
                self._record_delivery_success(event, now)
            return webhook_ok
        except Exception as exc:
            self._record_tg_failure(f"telegram_error:{type(exc).__name__}", event)
            self._publish_to_pentest(event)
            if webhook_ok:
                self._record_delivery_success(event, now)
            return webhook_ok

    async def _send_webhook(self, event: NotificationEvent) -> bool:
        sender = self._webhook_sender
        if sender is None:
            return False
        try:
            if not sender.enabled:
                return False
            severity = event.severity.name.lower()
            if severity == "success":
                severity = "info"
            elif severity == "daily":
                severity = "info"
            return bool(await sender.send(event.render(), severity=severity, title=event.title))
        except Exception:
            return False

    def _record_delivery_success(self, event: NotificationEvent, now: float) -> None:
        self._sent_ts.append(now)
        if event.throttle_key:
            self._last_by_key[event.throttle_key] = now
            if len(self._last_by_key) > 500:
                oldest_k = min(self._last_by_key, key=self._last_by_key.get)  # type: ignore[arg-type]
                self._last_by_key.pop(oldest_k, None)

    def _publish_to_pentest(self, event: NotificationEvent) -> None:
        if self._pentest_publisher is None:
            return
        try:
            asyncio.create_task(self._pentest_publisher.publish(event))
        except Exception:
            pass

    def _record_tg_failure(self, reason: str, event: NotificationEvent) -> None:
        _fallback_log_event(event, reason)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._TG_CIRCUIT_THRESHOLD and not self._tg_circuit_open:
            self._tg_circuit_open = True
            self._tg_circuit_open_ts = time.time()
            if not self._tg_dead_logged:
                self._tg_dead_logged = True
                # Log a critical warning that Telegram channel is dead
                try:
                    import logging
                    logging.getLogger("eclipse.notifications").critical(
                        f"TELEGRAM CIRCUIT BREAKER OPEN — {self._consecutive_failures} consecutive failures. "
                        f"Messages falling back to {_FALLBACK_LOG}. Retrying in {self._TG_CIRCUIT_COOLDOWN_SEC}s."
                    )
                except Exception:
                    pass

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
                symbol=evt.symbol,
                side=evt.side,
                source=evt.source,
                raw_payload=evt.raw_payload,
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
    webhook_sender = get_webhook_sender()
    nm = NotificationManager(notifier=notifier, config=load_config_from_env(), webhook_sender=webhook_sender)
    rc["notification_manager"] = nm
    return nm
