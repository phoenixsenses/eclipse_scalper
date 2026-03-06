from __future__ import annotations

import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_TS_CANDIDATES = ("ts_ms", "timestamp_ms", "timestamp", "ts", "event_time", "time_ms", "time")
_SYMBOL_CANDIDATES = ("symbol", "s", "pair", "instrument", "ticker")
_BID_CANDIDATES = ("best_bid", "bid", "bid_price", "b")
_ASK_CANDIDATES = ("best_ask", "ask", "ask_price", "a")
_BID_VOL_CANDIDATES = ("bid_vol", "bid_volume", "bq", "bid_qty")
_ASK_VOL_CANDIDATES = ("ask_vol", "ask_volume", "aq", "ask_qty")
_PRICE_CANDIDATES = ("mark_price", "mid", "price", "last_price", "p")
_TRADE_QTY_CANDIDATES = ("quantity", "qty", "size", "q")


@dataclass(frozen=True)
class TableSpec:
    name: str
    ts_col: Optional[str]
    symbol_col: Optional[str]
    bid_col: Optional[str]
    ask_col: Optional[str]
    bid_vol_col: Optional[str]
    ask_vol_col: Optional[str]
    price_col: Optional[str]
    trade_qty_col: Optional[str]
    is_trade_like: bool


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(r[0]) for r in rows if r and str(r[0]) and not str(r[0]).startswith("sqlite_")]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows if len(r) > 1]


def _pick_first(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    colset = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return None


def _build_table_spec(conn: sqlite3.Connection, table: str) -> TableSpec:
    cols = table_columns(conn, table)
    cols_lower = {c.lower() for c in cols}
    is_trade_like = ("agg" in table.lower() or "trade" in table.lower()) and ("price" in cols_lower or "p" in cols_lower)
    return TableSpec(
        name=table,
        ts_col=_pick_first(cols, _TS_CANDIDATES),
        symbol_col=_pick_first(cols, _SYMBOL_CANDIDATES),
        bid_col=_pick_first(cols, _BID_CANDIDATES),
        ask_col=_pick_first(cols, _ASK_CANDIDATES),
        bid_vol_col=_pick_first(cols, _BID_VOL_CANDIDATES),
        ask_vol_col=_pick_first(cols, _ASK_VOL_CANDIDATES),
        price_col=_pick_first(cols, _PRICE_CANDIDATES),
        trade_qty_col=_pick_first(cols, _TRADE_QTY_CANDIDATES),
        is_trade_like=is_trade_like,
    )


def discover_table_specs(conn: sqlite3.Connection) -> List[TableSpec]:
    out: List[TableSpec] = []
    for table in list_tables(conn):
        try:
            out.append(_build_table_spec(conn, table))
        except sqlite3.DatabaseError:
            continue
    return out


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except Exception:
        return None


def _query_rows(
    conn: sqlite3.Connection,
    spec: TableSpec,
    symbol: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if spec.ts_col is None or spec.symbol_col is None:
        return []
    fields = [spec.ts_col, spec.symbol_col]
    for c in (
        spec.bid_col,
        spec.ask_col,
        spec.bid_vol_col,
        spec.ask_vol_col,
        spec.price_col,
        spec.trade_qty_col,
    ):
        if c and c not in fields:
            fields.append(c)
    sql = f"SELECT {', '.join(fields)} FROM {spec.name} WHERE {spec.symbol_col} = ?"
    args: List[Any] = [symbol]
    if start_ms is not None:
        sql += f" AND {spec.ts_col} >= ?"
        args.append(int(start_ms))
    if end_ms is not None:
        sql += f" AND {spec.ts_col} <= ?"
        args.append(int(end_ms))
    sql += f" ORDER BY {spec.ts_col} ASC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        args.append(int(limit))
    rows = conn.execute(sql, args).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = {fields[i]: row[i] for i in range(len(fields))}
        out.append(rec)
    return out


def load_symbol_window(
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit_per_table: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Schema-agnostic window loader.
    Returns canonicalized records sorted by ts_ms.
    """
    symbol_u = str(symbol or "").strip().upper()
    records: List[Dict[str, Any]] = []
    for spec in discover_table_specs(conn):
        if spec.ts_col is None or spec.symbol_col is None:
            continue
        rows = _query_rows(conn, spec, symbol_u, start_ms, end_ms, limit=limit_per_table)
        for row in rows:
            ts_ms = _safe_int(row.get(spec.ts_col))
            if ts_ms is None:
                continue
            bid = _safe_float(row.get(spec.bid_col)) if spec.bid_col else None
            ask = _safe_float(row.get(spec.ask_col)) if spec.ask_col else None
            price = _safe_float(row.get(spec.price_col)) if spec.price_col else None
            bid_vol = _safe_float(row.get(spec.bid_vol_col)) if spec.bid_vol_col else None
            ask_vol = _safe_float(row.get(spec.ask_vol_col)) if spec.ask_vol_col else None
            qty = _safe_float(row.get(spec.trade_qty_col)) if spec.trade_qty_col else None
            mid = None
            spread = None
            if bid is not None and ask is not None:
                mid = (bid + ask) * 0.5
                spread = ask - bid
            elif price is not None:
                mid = price
            records.append(
                {
                    "ts_ms": int(ts_ms),
                    "symbol": symbol_u,
                    "source_table": spec.name,
                    "best_bid": bid,
                    "best_ask": ask,
                    "bid_vol": bid_vol,
                    "ask_vol": ask_vol,
                    "price": price,
                    "trade_qty": qty,
                    "mid": mid,
                    "spread": spread,
                    "trade_count": 1 if spec.is_trade_like else 0,
                }
            )
    records.sort(key=lambda r: int(r.get("ts_ms", 0) or 0))
    return records


def compute_features(records: List[Dict[str, Any]], volatility_window: int = 12) -> List[Dict[str, Any]]:
    """
    Pure feature computation from canonical records.
    """
    out: List[Dict[str, Any]] = []
    logrets: deque[float] = deque(maxlen=max(2, int(volatility_window)))
    trade_window: deque[tuple[int, int]] = deque()
    prev_mid: Optional[float] = None
    for rec in records:
        ts_ms = _safe_int(rec.get("ts_ms")) or 0
        mid = _safe_float(rec.get("mid"))
        if mid is None:
            bid = _safe_float(rec.get("best_bid"))
            ask = _safe_float(rec.get("best_ask"))
            if bid is not None and ask is not None:
                mid = (bid + ask) * 0.5
            else:
                mid = _safe_float(rec.get("price"))
        spread = _safe_float(rec.get("spread"))
        if spread is None:
            bid = _safe_float(rec.get("best_bid"))
            ask = _safe_float(rec.get("best_ask"))
            if bid is not None and ask is not None:
                spread = ask - bid
        bid_vol = _safe_float(rec.get("bid_vol"))
        ask_vol = _safe_float(rec.get("ask_vol"))
        imbalance = None
        if bid_vol is not None and ask_vol is not None:
            den = bid_vol + ask_vol
            if den > 0:
                imbalance = bid_vol / den
        logret = None
        if mid is not None and mid > 0 and prev_mid is not None and prev_mid > 0:
            logret = math.log(mid / prev_mid)
            logrets.append(logret)
        elif mid is not None and mid > 0:
            prev_mid = mid
        if mid is not None and mid > 0:
            prev_mid = mid
        micro_volatility = None
        if len(logrets) >= 2:
            m = sum(logrets) / len(logrets)
            var = sum((x - m) ** 2 for x in logrets) / (len(logrets) - 1)
            micro_volatility = math.sqrt(max(0.0, var))
        trade_count = int(rec.get("trade_count", 0) or 0)
        trade_window.append((ts_ms, trade_count))
        cutoff_ms = ts_ms - 60_000
        while trade_window and trade_window[0][0] < cutoff_ms:
            trade_window.popleft()
        trade_intensity = sum(v for _, v in trade_window)
        out.append(
            {
                "ts_ms": ts_ms,
                "symbol": str(rec.get("symbol") or ""),
                "mid": mid,
                "spread": spread,
                "imbalance": imbalance,
                "trade_intensity": trade_intensity,
                "micro_volatility": micro_volatility,
                "ret_1": logret,
            }
        )
    return out
