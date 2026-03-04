from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.symbols import canonical_symbol


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (str(table),),
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows if len(r) > 1]


def _pick(cols: list[str], *names: str) -> str:
    low = {c.lower(): c for c in cols}
    for n in names:
        c = low.get(str(n).lower())
        if c:
            return c
    return ""


@dataclass(frozen=True)
class PlacementDecision:
    mode: str
    limit_price: float
    queue_ahead: float
    source: str
    reason: str
    best_bid: float = 0.0
    best_ask: float = 0.0


class OrderPlacementEngine:
    """
    Passive order placement helper.
    Does not alter signal logic; only computes a suggested limit price.
    """

    def __init__(
        self,
        *,
        db_path: str = "data/microstructure.db",
        mode: str = "best",
        queue_depth_threshold: float = 500.0,
        default_tick_size: float = 0.01,
    ):
        self.db_path = str(db_path)
        m = str(mode or "best").strip().lower()
        if m not in ("best", "inside_spread", "adaptive"):
            m = "best"
        self.mode = m
        self.queue_depth_threshold = max(0.0, float(queue_depth_threshold))
        self.default_tick_size = max(1e-8, float(default_tick_size))

    @staticmethod
    def from_env() -> "OrderPlacementEngine":
        return OrderPlacementEngine(
            db_path=str(os.getenv("MICRO_SIGNAL_DB", "data/microstructure.db") or "data/microstructure.db"),
            mode=str(os.getenv("MICRO_SIGNAL_ORDER_PLACEMENT_MODE", "best") or "best"),
            queue_depth_threshold=float(os.getenv("MICRO_SIGNAL_QUEUE_DEPTH_THRESHOLD", "500") or 500.0),
            default_tick_size=float(os.getenv("MICRO_SIGNAL_TICK_SIZE", "0.01") or 0.01),
        )

    def _latest_top_of_book(self, symbol: str) -> tuple[float, float, float, float, str]:
        db = Path(self.db_path)
        if not db.exists():
            return 0.0, 0.0, 0.0, 0.0, "db_missing"
        conn = sqlite3.connect(str(db), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            for table in ("orderbook", "top_of_book", "book_snapshots", "book_tob"):
                if not _table_exists(conn, table):
                    continue
                cols = _table_cols(conn, table)
                sym_col = _pick(cols, "symbol", "s", "pair", "instrument")
                bid_px_col = _pick(cols, "bid_price", "best_bid", "bid_px", "b")
                ask_px_col = _pick(cols, "ask_price", "best_ask", "ask_px", "a")
                bid_qty_col = _pick(cols, "bid_size", "bid_qty", "best_bid_qty", "bq")
                ask_qty_col = _pick(cols, "ask_size", "ask_qty", "best_ask_qty", "aq")
                ts_col = _pick(cols, "ts_ms", "ts_utc", "ts", "timestamp", "event_ts", "time")
                if not (sym_col and bid_px_col and ask_px_col):
                    continue
                order_by = ts_col if ts_col else sym_col
                query = (
                    f"SELECT {bid_px_col} as bid_px, {ask_px_col} as ask_px, "
                    f"{(bid_qty_col or '0')} as bid_qty, {(ask_qty_col or '0')} as ask_qty "
                    f"FROM {table} WHERE {sym_col}=? ORDER BY {order_by} DESC LIMIT 1"
                )
                row = conn.execute(query, (canonical_symbol(symbol),)).fetchone()
                if row:
                    bid_px = _safe_float(row["bid_px"], 0.0)
                    ask_px = _safe_float(row["ask_px"], 0.0)
                    bid_qty = _safe_float(row["bid_qty"], 0.0)
                    ask_qty = _safe_float(row["ask_qty"], 0.0)
                    if bid_px > 0 and ask_px > 0:
                        return bid_px, ask_px, bid_qty, ask_qty, f"db:{table}"
            return 0.0, 0.0, 0.0, 0.0, "no_book_rows"
        finally:
            conn.close()

    def decide(
        self,
        *,
        symbol: str,
        side: str,  # long | short
        ref_price: float,
        tick_size: Optional[float] = None,
        fallback_limit: Optional[float] = None,
    ) -> PlacementDecision:
        mode = self.mode
        tick = max(1e-8, float(tick_size or self.default_tick_size))
        ref = _safe_float(ref_price, 0.0)
        fallback_px = _safe_float(fallback_limit, ref)
        bid_px, ask_px, bid_qty, ask_qty, src = self._latest_top_of_book(symbol)
        if bid_px <= 0 or ask_px <= 0:
            if fallback_px <= 0 and ref > 0:
                fallback_px = ref
            return PlacementDecision(
                mode=mode,
                limit_price=float(max(0.0, fallback_px)),
                queue_ahead=0.0,
                source=src,
                reason="fallback_no_book",
                best_bid=bid_px,
                best_ask=ask_px,
            )

        is_long = str(side).strip().lower() == "long"
        best_px = bid_px if is_long else ask_px
        queue_ahead = bid_qty if is_long else ask_qty
        spread = max(0.0, ask_px - bid_px)

        chosen_mode = mode
        if mode == "adaptive":
            if queue_ahead > self.queue_depth_threshold and spread >= (2.0 * tick):
                chosen_mode = "inside_spread"
            else:
                chosen_mode = "best"

        if chosen_mode == "inside_spread":
            if is_long:
                px = min(ask_px - tick, bid_px + tick)
                px = max(bid_px, px)
            else:
                px = max(bid_px + tick, ask_px - tick)
                px = min(ask_px, px)
        else:
            px = best_px

        if px <= 0:
            px = fallback_px if fallback_px > 0 else ref
            return PlacementDecision(
                mode=chosen_mode,
                limit_price=float(max(0.0, px)),
                queue_ahead=float(queue_ahead),
                source=src,
                reason="fallback_invalid_px",
                best_bid=bid_px,
                best_ask=ask_px,
            )

        return PlacementDecision(
            mode=chosen_mode,
            limit_price=float(px),
            queue_ahead=float(queue_ahead),
            source=src,
            reason="ok",
            best_bid=bid_px,
            best_ask=ask_px,
        )

