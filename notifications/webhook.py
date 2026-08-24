# notifications/webhook.py — SCALPER ETERNAL — WEBHOOK SENDER — 2026 v1.0
# Generic webhook sender for Slack, Discord, and custom endpoints.
#
# Usage:
#   sender = WebhookSender(url="https://hooks.slack.com/services/...", platform="slack")
#   await sender.send("Kill switch triggered", severity="critical")
#
# Design:
# - Never fatal: all sends are best-effort
# - Circuit breaker: backs off after repeated failures
# - Supports Slack and Discord payload formats

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from utils.logging import log_core
except Exception:
    import logging
    log_core = logging.getLogger("webhook")

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
_WEBHOOK_PLATFORM = os.getenv("WEBHOOK_PLATFORM", "slack").lower()  # slack | discord | generic
_WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "0").lower() in ("1", "true", "yes", "on")
_WEBHOOK_TIMEOUT_SEC = float(os.getenv("WEBHOOK_TIMEOUT_SEC", "5.0"))


# ---------------------------------------------------------------------------
# Payload formatters
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "info": "#36a64f",
    "warning": "#ff9900",
    "critical": "#ff0000",
    "error": "#ff0000",
}

_SEVERITY_EMOJI = {
    "info": "ℹ️",
    "warning": "⚠️",
    "critical": "🚨",
    "error": "❌",
}


def _slack_payload(text: str, severity: str = "info", title: str = "") -> Dict[str, Any]:
    color = _SEVERITY_COLORS.get(severity, "#333333")
    return {
        "attachments": [
            {
                "color": color,
                "title": title or "Eclipse Scalper",
                "text": text,
                "footer": "Eclipse Scalper Bot",
                "ts": int(time.time()),
            }
        ]
    }


def _discord_payload(text: str, severity: str = "info", title: str = "") -> Dict[str, Any]:
    color_hex = _SEVERITY_COLORS.get(severity, "#333333")
    # Discord uses decimal color
    color_int = int(color_hex.lstrip("#"), 16)
    emoji = _SEVERITY_EMOJI.get(severity, "")
    return {
        "embeds": [
            {
                "title": f"{emoji} {title or 'Eclipse Scalper'}",
                "description": text,
                "color": color_int,
                "footer": {"text": "Eclipse Scalper Bot"},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ]
    }


def _generic_payload(text: str, severity: str = "info", title: str = "") -> Dict[str, Any]:
    return {
        "title": title or "Eclipse Scalper",
        "text": text,
        "severity": severity,
        "timestamp": time.time(),
    }


_FORMATTERS = {
    "slack": _slack_payload,
    "discord": _discord_payload,
    "generic": _generic_payload,
}


# ---------------------------------------------------------------------------
# Webhook Sender
# ---------------------------------------------------------------------------

class WebhookSender:
    """
    Async webhook sender with circuit breaker.

    Args:
        url: Webhook URL (Slack/Discord/custom)
        platform: "slack", "discord", or "generic"
        timeout: HTTP timeout in seconds
    """

    _CIRCUIT_THRESHOLD = 5
    _CIRCUIT_COOLDOWN_SEC = 120.0

    def __init__(
        self,
        url: str = "",
        platform: str = "slack",
        timeout: float = 5.0,
    ):
        self.url = url or _WEBHOOK_URL
        self.platform = (platform or _WEBHOOK_PLATFORM).lower()
        self.timeout = timeout or _WEBHOOK_TIMEOUT_SEC
        self._formatter = _FORMATTERS.get(self.platform, _generic_payload)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_ts = 0.0
        self._total_sent = 0
        self._total_failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def send(
        self,
        text: str,
        severity: str = "info",
        title: str = "",
    ) -> bool:
        """
        Send a webhook message. Returns True on success, False on failure.
        Never raises.
        """
        if not self.url:
            return False

        if aiohttp is None:
            log_core.warning("[webhook] aiohttp not installed — cannot send")
            return False

        now = time.time()

        # Circuit breaker check
        if self._circuit_open:
            if (now - self._circuit_open_ts) < self._CIRCUIT_COOLDOWN_SEC:
                return False
            self._circuit_open = False

        payload = self._formatter(text, severity, title)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status < 300:
                        self._consecutive_failures = 0
                        self._total_sent += 1
                        return True
                    else:
                        self._record_failure(f"HTTP {resp.status}")
                        return False

        except Exception as e:
            self._record_failure(str(e))
            return False

    def _record_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        self._total_failed += 1
        if self._consecutive_failures >= self._CIRCUIT_THRESHOLD:
            self._circuit_open = True
            self._circuit_open_ts = time.time()
            log_core.warning(f"[webhook] circuit open after {self._consecutive_failures} failures: {reason}")
        else:
            log_core.warning(f"[webhook] send failed ({self._consecutive_failures}): {reason}")

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "url_set": bool(self.url),
            "platform": self.platform,
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
            "circuit_open": self._circuit_open,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_sender: Optional[WebhookSender] = None


def get_sender() -> WebhookSender:
    """Get or create the default webhook sender."""
    global _default_sender
    if _default_sender is None:
        _default_sender = WebhookSender(
            url=_WEBHOOK_URL,
            platform=_WEBHOOK_PLATFORM,
            timeout=_WEBHOOK_TIMEOUT_SEC,
        )
    return _default_sender


async def send_webhook(text: str, severity: str = "info", title: str = "") -> bool:
    """Convenience function to send via default sender."""
    return await get_sender().send(text, severity, title)
