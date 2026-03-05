from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BaselineStrategy:
    """Deterministic baseline strategy that emits one signal every N events per symbol."""

    period: int = 5
    action: str = "signal"

    def __post_init__(self) -> None:
        self.period = max(1, int(self.period))
        self._count_by_symbol: Dict[str, int] = {}

    def on_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        symbol = str(event.get("symbol") or "ALL")
        cnt = int(self._count_by_symbol.get(symbol, 0)) + 1
        self._count_by_symbol[symbol] = cnt
        if (cnt % self.period) != 0:
            return []
        params = {
            "period": int(self.period),
            "source_table": str(event.get("source_table") or ""),
            "event_index": int(event.get("event_index") or 0),
            "symbol_count": cnt,
        }
        return [
            {
                "action": self.action,
                "params": params,
            }
        ]

    def on_tick(self, ts_utc: str) -> List[Dict[str, Any]]:
        _ = ts_utc
        return []

