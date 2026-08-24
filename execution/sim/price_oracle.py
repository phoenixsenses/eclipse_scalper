from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_iso_utc(text: str) -> float:
    s = str(text or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _extract_price(payload: Dict[str, Any], fields: Sequence[str]) -> Optional[float]:
    for k in fields:
        if k in payload:
            try:
                v = float(payload[k])
                if v > 0:
                    return v
            except Exception:
                continue
    return None


@dataclass(frozen=True)
class PricePoint:
    ts: float
    price: float
    ts_utc: str
    rowid: int
    source_table: str


class PriceOracle:
    def __init__(self, by_symbol: Dict[str, List[PricePoint]]) -> None:
        self._by_symbol = by_symbol
        self._ts = {k: [p.ts for p in v] for k, v in by_symbol.items()}

    @classmethod
    def build(
        cls,
        events: Iterable[Dict[str, Any]],
        price_field_candidates: Sequence[str],
    ) -> "PriceOracle":
        rows: Dict[str, List[PricePoint]] = {}
        for ev in events:
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            px = _extract_price(payload, price_field_candidates)
            if px is None:
                continue
            try:
                ts_utc = str(ev.get("ts_utc") or "")
                ts = _parse_iso_utc(ts_utc)
            except Exception:
                continue
            symbol = str(ev.get("symbol") or "ALL").upper()
            rowid = int(ev.get("rowid") or 0)
            source_table = str(ev.get("source_table") or "")
            rows.setdefault(symbol, []).append(
                PricePoint(ts=ts, price=float(px), ts_utc=ts_utc, rowid=rowid, source_table=source_table)
            )
        by_symbol: Dict[str, List[PricePoint]] = {}
        for sym, points in rows.items():
            points.sort(key=lambda p: (p.ts, p.rowid, p.source_table))
            by_symbol[sym] = points
        return cls(by_symbol)

    def has_symbol(self, symbol: str) -> bool:
        return str(symbol).upper() in self._by_symbol

    def price_at_or_after(self, symbol: str, target_ts: float, max_ts: Optional[float] = None) -> Optional[PricePoint]:
        sym = str(symbol).upper()
        points = self._by_symbol.get(sym)
        if not points:
            return None
        ts_list = self._ts[sym]
        i = bisect_left(ts_list, float(target_ts))
        if i >= len(points):
            return None
        out = points[i]
        if max_ts is not None and out.ts > float(max_ts):
            return None
        return out

    def price_at_or_before(self, symbol: str, target_ts: float) -> Optional[PricePoint]:
        sym = str(symbol).upper()
        points = self._by_symbol.get(sym)
        if not points:
            return None
        ts_list = self._ts[sym]
        i = bisect_right(ts_list, float(target_ts)) - 1
        if i < 0:
            return None
        return points[i]

    def price_window(self, symbol: str, start_ts: float, end_ts: float) -> List[PricePoint]:
        sym = str(symbol).upper()
        points = self._by_symbol.get(sym)
        if not points:
            return []
        ts_list = self._ts[sym]
        i0 = bisect_left(ts_list, float(start_ts))
        i1 = bisect_right(ts_list, float(end_ts))
        if i1 <= i0:
            return []
        return points[i0:i1]

