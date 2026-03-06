from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NotificationSeverity(Enum):
    INFO = "ℹ"
    SUCCESS = "✅"
    WARNING = "⚠"
    CRITICAL = "🚨"
    DAILY = "📊"


@dataclass(frozen=True)
class NotificationEvent:
    severity: NotificationSeverity
    category: str
    title: str
    body: str
    throttle_key: Optional[str] = None
    throttle_window_sec: float = 0.0
    silent: bool = False

    def render(self) -> str:
        head = f"{self.severity.value} {self.title}".strip()
        body = str(self.body or "").strip()
        return head if not body else f"{head}\n{body}"

