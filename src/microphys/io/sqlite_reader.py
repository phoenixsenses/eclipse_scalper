from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.check_data_ready import detect_ts_col, list_tables, table_columns
from utils.symbols import canonical_symbol

from src.microphys.schemas import LiquidationEvent, TopOfBookEvent, TradeEvent


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_ts(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e12:
            return x / 1000.0
        return x
    s = str(v).strip()
    if not s:
        return 0.0
    try:
        return _parse_ts(float(s))
    except Exception:
        pass
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def _pick(cols: List[str], candidates: Iterable[str]) -> str:
    lm = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lm:
            return lm[c.lower()]
    return ""


def _is_text_type(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for r in rows:
        if str(r[1]) != col:
            continue
        t = str(r[2] or "").upper()
        return ("CHAR" in t) or ("TEXT" in t) or ("CLOB" in t)
    return False


@dataclass(frozen=True)
class TableMapping:
    kind: str
    table: str
    ts_col: str
    symbol_col: str
    cols: Dict[str, str]
    ts_is_text: bool


def discover_mappings(db_path: Path) -> Dict[str, Optional[TableMapping]]:
    # Read-only: metadata discovery only (PRAGMA table_info), no data
    # mutation. data/microstructure.db in particular must always be opened
    # mode=ro (CLAUDE.md guardrail) -- it is a large, concurrently-written
    # production database.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = list_tables(conn)
        out: Dict[str, Optional[TableMapping]] = {"trades": None, "book": None, "liquidations": None}
        best_score: Dict[str, int] = {"trades": -1, "book": -1, "liquidations": -1}

        for t in tables:
            cols = table_columns(conn, t)
            if not cols:
                continue
            ts_col = detect_ts_col(cols) or _pick(cols, ("ts_ms", "timestamp", "ts", "event_ts", "time"))
            sym_col = _pick(cols, ("symbol", "s", "pair", "instrument", "ticker"))
            if not ts_col or not sym_col:
                continue

            price_col = _pick(cols, ("price", "p", "trade_price", "mark_price", "mid_price", "mid"))
            qty_col = _pick(cols, ("quantity", "qty", "q", "size", "amount"))
            side_col = _pick(cols, ("side", "taker_side"))
            bm_col = _pick(cols, ("is_buyer_maker",))
            bid_px = _pick(cols, ("bid_price", "best_bid", "bid_px", "bid"))
            ask_px = _pick(cols, ("ask_price", "best_ask", "ask_px", "ask"))
            bid_qty = _pick(cols, ("bid_qty", "bid_size", "best_bid_qty"))
            ask_qty = _pick(cols, ("ask_qty", "ask_size", "best_ask_qty"))

            lname = t.lower()
            score_trade = int(bool(price_col)) + int(bool(qty_col)) + int(bool(side_col or bm_col))
            score_book = int(bool(bid_px)) + int(bool(ask_px)) + int(bool(bid_qty)) + int(bool(ask_qty))
            score_liq = int("liq" in lname) + int(bool(side_col)) + int(bool(qty_col))

            # Name-aware tie-breakers prevent liquidation tables from being selected as trades.
            if ("trade" in lname) or ("agg" in lname):
                score_trade += 2
            if ("liq" in lname) or ("liquid" in lname):
                score_trade -= 2
                score_liq += 3
            if ("mark" in lname) or ("book" in lname) or ("depth" in lname):
                score_book += 2

            if score_trade > best_score["trades"]:
                out["trades"] = TableMapping(
                    kind="trades",
                    table=t,
                    ts_col=ts_col,
                    symbol_col=sym_col,
                    cols={"price": price_col, "qty": qty_col, "side": side_col, "is_buyer_maker": bm_col},
                    ts_is_text=_is_text_type(conn, t, ts_col),
                )
                best_score["trades"] = score_trade

            if score_book > best_score["book"] and (score_book >= 2 or "mark" in lname):
                out["book"] = TableMapping(
                    kind="book",
                    table=t,
                    ts_col=ts_col,
                    symbol_col=sym_col,
                    cols={
                        "bid_px": bid_px,
                        "ask_px": ask_px,
                        "bid_qty": bid_qty,
                        "ask_qty": ask_qty,
                        "mid": _pick(cols, ("mid", "mid_price", "mark_price", "price", "p")),
                    },
                    ts_is_text=_is_text_type(conn, t, ts_col),
                )
                best_score["book"] = score_book

            if score_liq > best_score["liquidations"]:
                out["liquidations"] = TableMapping(
                    kind="liquidations",
                    table=t,
                    ts_col=ts_col,
                    symbol_col=sym_col,
                    cols={"side": side_col, "qty": qty_col, "price": price_col},
                    ts_is_text=_is_text_type(conn, t, ts_col),
                )
                best_score["liquidations"] = score_liq

        return out
    finally:
        conn.close()


class SQLiteMicroReader:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.mappings = discover_mappings(self.db_path)

    def _query_rows(self, mp: TableMapping, symbol: str, start_ts: float, end_ts: float) -> List[sqlite3.Row]:
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            select_cols = [mp.ts_col, mp.symbol_col]
            for v in mp.cols.values():
                if v and v not in select_cols:
                    select_cols.append(v)
            if mp.ts_is_text:
                s0 = dt.datetime.utcfromtimestamp(start_ts).isoformat() + "Z"
                s1 = dt.datetime.utcfromtimestamp(end_ts).isoformat() + "Z"
                sql = (
                    f"SELECT {', '.join(select_cols)} FROM {mp.table} "
                    f"WHERE {mp.symbol_col}=? AND {mp.ts_col}>=? AND {mp.ts_col}<=? "
                    f"ORDER BY {mp.ts_col} ASC"
                )
                rows = conn.execute(sql, (symbol, s0, s1)).fetchall()
            else:
                probe = conn.execute(
                    f"SELECT {mp.ts_col} FROM {mp.table} WHERE {mp.symbol_col}=? ORDER BY {mp.ts_col} DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
                scale = 1.0
                if probe is not None:
                    try:
                        raw = float(probe[0])
                        if raw > 1e12:
                            scale = 1000.0
                    except Exception:
                        pass
                sql = (
                    f"SELECT {', '.join(select_cols)} FROM {mp.table} "
                    f"WHERE {mp.symbol_col}=? AND {mp.ts_col}>=? AND {mp.ts_col}<=? "
                    f"ORDER BY {mp.ts_col} ASC"
                )
                rows = conn.execute(sql, (symbol, start_ts * scale, end_ts * scale)).fetchall()
            return rows
        finally:
            conn.close()

    def get_ts_range(self, kind: str, symbol: str) -> tuple[Optional[float], Optional[float]]:
        mp = self.mappings.get(kind)
        if mp is None:
            return None, None
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        try:
            mn, mx = conn.execute(
                f"SELECT MIN({mp.ts_col}), MAX({mp.ts_col}) FROM {mp.table} WHERE {mp.symbol_col}=?",
                (canonical_symbol(symbol),),
            ).fetchone()
            return _parse_ts(mn), _parse_ts(mx)
        finally:
            conn.close()

    def read_trades(self, symbol: str, start_ts: float, end_ts: float) -> List[TradeEvent]:
        mp = self.mappings.get("trades")
        if mp is None:
            return []
        sym = canonical_symbol(symbol)
        out: List[TradeEvent] = []
        for r in self._query_rows(mp, sym, start_ts, end_ts):
            ts = _parse_ts(r[mp.ts_col])
            price = _safe_float(r[mp.cols.get("price")] if mp.cols.get("price") else 0.0, 0.0)
            qty = _safe_float(r[mp.cols.get("qty")] if mp.cols.get("qty") else 0.0, 0.0)
            side = ""
            if mp.cols.get("side"):
                side = str(r[mp.cols["side"]] or "").lower().strip()
            elif mp.cols.get("is_buyer_maker"):
                try:
                    side = "sell" if int(r[mp.cols["is_buyer_maker"]]) == 1 else "buy"
                except Exception:
                    side = ""
            out.append(TradeEvent(ts=ts, symbol=sym, price=price, qty=qty, side=side))
        return out

    def read_top_of_book(self, symbol: str, start_ts: float, end_ts: float) -> List[TopOfBookEvent]:
        mp = self.mappings.get("book")
        if mp is None:
            return []
        sym = canonical_symbol(symbol)
        out: List[TopOfBookEvent] = []
        for r in self._query_rows(mp, sym, start_ts, end_ts):
            ts = _parse_ts(r[mp.ts_col])
            bid_px = _safe_float(r[mp.cols.get("bid_px")] if mp.cols.get("bid_px") else 0.0, 0.0)
            ask_px = _safe_float(r[mp.cols.get("ask_px")] if mp.cols.get("ask_px") else 0.0, 0.0)
            bid_qty = _safe_float(r[mp.cols.get("bid_qty")] if mp.cols.get("bid_qty") else 0.0, 0.0)
            ask_qty = _safe_float(r[mp.cols.get("ask_qty")] if mp.cols.get("ask_qty") else 0.0, 0.0)
            mid = _safe_float(r[mp.cols.get("mid")] if mp.cols.get("mid") else 0.0, 0.0)
            if bid_px <= 0 and ask_px <= 0 and mid > 0:
                bid_px = mid
                ask_px = mid
            out.append(TopOfBookEvent(ts=ts, symbol=sym, bid_px=bid_px, bid_qty=bid_qty, ask_px=ask_px, ask_qty=ask_qty))
        return out

    def read_liquidations(self, symbol: str, start_ts: float, end_ts: float) -> List[LiquidationEvent]:
        mp = self.mappings.get("liquidations")
        if mp is None:
            return []
        sym = canonical_symbol(symbol)
        out: List[LiquidationEvent] = []
        for r in self._query_rows(mp, sym, start_ts, end_ts):
            ts = _parse_ts(r[mp.ts_col])
            side = str(r[mp.cols.get("side")] if mp.cols.get("side") else "" or "").lower().strip()
            qty = _safe_float(r[mp.cols.get("qty")] if mp.cols.get("qty") else 0.0, 0.0)
            price = _safe_float(r[mp.cols.get("price")] if mp.cols.get("price") else 0.0, 0.0)
            out.append(LiquidationEvent(ts=ts, symbol=sym, side=side, qty=qty, price=price))
        return out
