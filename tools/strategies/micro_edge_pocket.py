from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _first_float(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in payload:
            val = _f(payload.get(k))
            if val is not None:
                return val
    return None


@dataclass
class MicroEdgePocketStrategy:
    rule: str = "micro_edge_v3_passive_alpha"
    side: str = "buy"  # buy | sell | auto
    symbol_whitelist: tuple[str, ...] = ()
    event_source_table: str = "agg_trades"
    min_trade_count_window: int = 1
    horizon_sec: int = 120
    cooldown_ms: int = 250
    filters: Dict[str, float] | None = None
    action: str = "signal"

    def __post_init__(self) -> None:
        self.side = str(self.side or "buy").lower()
        if self.side not in ("buy", "sell", "auto"):
            self.side = "buy"
        self.symbol_whitelist = tuple(str(x).upper() for x in self.symbol_whitelist)
        self.min_trade_count_window = max(1, int(self.min_trade_count_window))
        self.horizon_sec = max(1, int(self.horizon_sec))
        self.cooldown_ms = max(0, int(self.cooldown_ms))
        self.filters = dict(self.filters or {})
        self._count_by_symbol: Dict[str, int] = {}
        self._last_signal_ms_by_symbol: Dict[str, int] = {}
        self._pocket_id = self._build_pocket_id()

    def _build_pocket_id(self) -> str:
        raw = json.dumps(
            {
                "rule": self.rule,
                "side": self.side,
                "filters": dict(sorted(self.filters.items())),
                "cooldown_ms": self.cooldown_ms,
                "horizon_sec": self.horizon_sec,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _resolve_side(self, payload: Dict[str, Any]) -> str:
        if self.side in ("buy", "sell"):
            return self.side
        # auto mode: simple deterministic fallback to imbalance sign if available
        imb = _first_float(payload, ("imbalance", "imb", "order_imbalance"))
        if imb is None:
            return "buy"
        if imb < 0:
            return "sell"
        return "buy"

    def on_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_table = str(event.get("source_table") or "")
        if self.event_source_table and source_table != self.event_source_table:
            return []
        symbol = str(event.get("symbol") or "ALL").upper()
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
            return []
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        ts_utc = str(event.get("ts_utc") or "")
        ts_ms_raw = _first_float({"ts_ms": payload.get("ts_ms"), "ts": payload.get("ts"), "timestamp": payload.get("timestamp")}, ("ts_ms", "ts", "timestamp"))
        if ts_ms_raw is None:
            return []
        ts_ms = int(ts_ms_raw if ts_ms_raw > 1e12 else (ts_ms_raw * 1000.0))

        cnt = int(self._count_by_symbol.get(symbol, 0)) + 1
        self._count_by_symbol[symbol] = cnt
        if cnt < self.min_trade_count_window:
            return []

        last_ms = int(self._last_signal_ms_by_symbol.get(symbol, -10**18))
        if (ts_ms - last_ms) < self.cooldown_ms:
            return []

        intensity = _first_float(payload, ("trade_intensity", "intensity", "ti"))
        spread = _first_float(payload, ("spread", "spr"))
        imbalance = _first_float(payload, ("imbalance", "imb", "order_imbalance"))
        checks = {
            "filters.intensity_gte": intensity is not None and intensity >= float(self.filters.get("intensity_gte", float("-inf"))),
            "filters.spread_lte": spread is not None and spread <= float(self.filters.get("spread_lte", float("inf"))),
            "filters.imbalance_gte": imbalance is not None and imbalance >= float(self.filters.get("imbalance_gte", float("-inf"))),
        }
        if not all(checks.values()):
            return []

        self._last_signal_ms_by_symbol[symbol] = ts_ms
        resolved_side = self._resolve_side(payload)
        params = {
            "rule": self.rule,
            "side": resolved_side,
            "filters": dict(sorted(self.filters.items())),
            "source_table": source_table,
            "event_keys_used": ["imbalance", "trade_intensity", "spread"],
            "pocket_id": self._pocket_id,
            "horizon_sec": self.horizon_sec,
            "cooldown_ms": self.cooldown_ms,
            "symbol_count": cnt,
        }
        return [{"action": self.action, "params": params}]

    def on_tick(self, ts_utc: str) -> List[Dict[str, Any]]:
        _ = ts_utc
        return []

