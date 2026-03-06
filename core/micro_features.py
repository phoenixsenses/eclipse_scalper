from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.symbols import canonical_symbol

try:
    from utils.logging import log_core
except Exception:  # pragma: no cover
    class _FallbackLogger:
        @staticmethod
        def info(msg: str) -> None:
            print(msg)

        @staticmethod
        def warning(msg: str) -> None:
            print(msg)

    log_core = _FallbackLogger()  # type: ignore


@dataclass(frozen=True)
class MicroFeatures:
    timestamp: float
    symbol: str
    imbalance: float
    imbalance_signed: float
    trade_intensity: float
    spread: float
    mark_price: float
    age_sec: float


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str:
    low_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low_map:
            return low_map[c]
    return ""


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows if len(r) > 1]


def _parse_ts_any(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e12:
            return x / 1000.0
        if x > 1e9:
            return x
        return x
    s = str(v).strip()
    if not s:
        return 0.0
    try:
        x = float(s)
        return _parse_ts_any(x)
    except Exception:
        pass
    try:
        s2 = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s2).timestamp()
    except Exception:
        return 0.0


def _is_text_col(cols: list[sqlite3.Row], col: str) -> bool:
    for r in cols:
        if str(r[1]) != col:
            continue
        t = str(r[2] or "").upper()
        return ("CHAR" in t) or ("TEXT" in t) or ("CLOB" in t)
    return False


class MicroFeatureEngine:
    def __init__(
        self,
        db_path: str,
        symbol: str | list[str],
        lookback_sec: int = 120,
        update_interval_sec: float = 1.0,
        bucket_sec: int = 5,
    ):
        self.db_path = str(db_path)
        if isinstance(symbol, (list, tuple, set)):
            syms = [canonical_symbol(str(s)) for s in symbol if str(s).strip()]
        else:
            syms = [canonical_symbol(str(symbol))]
        if not syms:
            raise ValueError("symbol list is empty")
        self.symbols = list(dict.fromkeys(syms))
        self.default_symbol = self.symbols[0]
        self.lookback_sec = int(max(30, lookback_sec))
        self.update_interval_sec = float(max(0.1, update_interval_sec))
        self.bucket_sec = int(max(1, bucket_sec))
        self.min_trades_30s = int(max(1, _safe_float(__import__("os").getenv("MICRO_MIN_TRADES_30S", "5"), 5.0)))
        self.min_marks_30s = int(max(1, _safe_float(__import__("os").getenv("MICRO_MIN_MARKS_30S", "1"), 1.0)))
        self.max_age_sec = float(max(1.0, _safe_float(__import__("os").getenv("MICRO_STALE_SEC", "10"), 10.0)))
        self.heartbeat_sec = float(max(5.0, _safe_float(__import__("os").getenv("MICRO_HEARTBEAT_SEC", "30"), 30.0)))
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._stop = False
        self._features_by_symbol: dict[str, MicroFeatures] = {}
        self._readiness_by_symbol: dict[str, tuple[bool, str, str]] = {}
        self._sample_counts_by_symbol: dict[str, dict[str, int]] = {}
        self._last_update_ts_by_symbol: dict[str, float] = {}
        self._last_db_ts_by_symbol: dict[str, float] = {}
        self._last_hb_ts = 0.0

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._task = None

    @property
    def current_features(self) -> Optional[MicroFeatures]:
        return self.get_features(self.default_symbol)

    def get_features(self, symbol: str) -> Optional[MicroFeatures]:
        try:
            k = canonical_symbol(symbol)
        except Exception:
            return None
        with self._lock:
            feat = self._features_by_symbol.get(k)
        if feat is None:
            return None
        now = time.time()
        return MicroFeatures(
            timestamp=float(feat.timestamp),
            symbol=str(feat.symbol),
            imbalance=float(feat.imbalance),
            imbalance_signed=float(feat.imbalance_signed),
            trade_intensity=float(feat.trade_intensity),
            spread=float(feat.spread),
            mark_price=float(feat.mark_price),
            age_sec=max(0.0, now - float(feat.timestamp)),
        )

    def get_readiness(self, symbol: str) -> tuple[bool, str, str]:
        try:
            k = canonical_symbol(symbol)
        except Exception:
            return False, "symbol_mismatch", "invalid_symbol"
        with self._lock:
            r = self._readiness_by_symbol.get(k)
        return r if isinstance(r, tuple) and len(r) == 3 else (False, "no_data", "no_state")

    def get_diag(self, symbol: str) -> dict[str, Any]:
        try:
            k = canonical_symbol(symbol)
        except Exception:
            return {"symbol": str(symbol), "ready": False, "reason": "symbol_mismatch", "detail": "invalid_symbol"}
        with self._lock:
            feat = self._features_by_symbol.get(k)
            ready = self._readiness_by_symbol.get(k, (False, "no_data", "no_state"))
            counts = dict(self._sample_counts_by_symbol.get(k, {}))
            upd = float(self._last_update_ts_by_symbol.get(k, 0.0) or 0.0)
            db_ts = float(self._last_db_ts_by_symbol.get(k, 0.0) or 0.0)
        return {
            "symbol": k,
            "ready": bool(ready[0]),
            "reason": str(ready[1]),
            "detail": str(ready[2]),
            "feature_count": (8 if feat is not None else 0),
            "age_sec": (max(0.0, time.time() - float(feat.timestamp)) if feat is not None else None),
            "last_update_age_sec": (max(0.0, time.time() - upd) if upd > 0 else None),
            "last_db_age_sec": (max(0.0, time.time() - db_ts) if db_ts > 0 else None),
            "sample_counts": counts,
        }

    def _compute_once(self) -> Optional[MicroFeatures]:
        self._refresh_all_once()
        return self.get_features(self.default_symbol)

    async def _run_loop(self) -> None:
        while not self._stop:
            try:
                self._refresh_all_once()
                now = time.time()
                if (now - self._last_hb_ts) >= self.heartbeat_sec:
                    self._last_hb_ts = now
                    self._log_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_core.warning(f"[MICRO] refresh loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(self.update_interval_sec)

    def _log_heartbeat(self) -> None:
        for s in self.symbols:
            d = self.get_diag(s)
            counts = d.get("sample_counts") or {}
            log_core.info(
                "[MICRO_HEARTBEAT] "
                f"{s} ready={int(bool(d.get('ready')))} reason={d.get('reason')} "
                f"features={int(d.get('feature_count') or 0)} age_sec={d.get('age_sec')} "
                f"db_age_sec={d.get('last_db_age_sec')} "
                f"trades_30s={int(counts.get('trades_30s', 0) or 0)} marks_30s={int(counts.get('marks_30s', 0) or 0)}"
            )

    def _refresh_all_once(self) -> None:
        db = Path(self.db_path)
        if not db.exists():
            with self._lock:
                for s in self.symbols:
                    self._readiness_by_symbol[s] = (False, "db_error", "db_missing")
            return

        conn = sqlite3.connect(str(db), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            now_sec = time.time()
            for s in self.symbols:
                feat, ready, counts, db_ts = self._compute_symbol(conn, s, now_sec)
                with self._lock:
                    self._readiness_by_symbol[s] = ready
                    self._sample_counts_by_symbol[s] = counts
                    self._last_update_ts_by_symbol[s] = now_sec
                    self._last_db_ts_by_symbol[s] = db_ts
                    if feat is not None:
                        self._features_by_symbol[s] = feat
        finally:
            conn.close()

    def _compute_symbol(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        now_sec: float,
    ) -> tuple[Optional[MicroFeatures], tuple[bool, str, str], dict[str, int], float]:
        try:
            trades, marks, liqs = self._load_windows(conn, symbol, now_sec)
        except Exception as e:
            return None, (False, "db_error", f"{type(e).__name__}:{e}"), {}, 0.0

        counts = {
            "trades_30s": sum(1 for r in trades if (now_sec - r["ts_sec"]) <= 30.0),
            "trades_120s": len(trades),
            "marks_30s": sum(1 for r in marks if (now_sec - r["ts_sec"]) <= 30.0),
            "marks_120s": len(marks),
            "liqs_30s": sum(1 for r in liqs if (now_sec - r["ts_sec"]) <= 30.0),
            "liqs_120s": len(liqs),
        }

        last_ts = 0.0
        if trades:
            last_ts = max(last_ts, max(float(r["ts_sec"]) for r in trades))
        if marks:
            last_ts = max(last_ts, max(float(r["ts_sec"]) for r in marks))
        if liqs:
            last_ts = max(last_ts, max(float(r["ts_sec"]) for r in liqs))

        if not trades and not marks:
            return None, (False, "no_data", "no_rows_in_window"), counts, last_ts

        age_sec = (now_sec - last_ts) if last_ts > 0 else 1e9
        if age_sec > self.max_age_sec:
            feat = self._build_features(symbol, now_sec, trades, marks, liqs, age_sec)
            return feat, (False, "stale", f"age_sec={age_sec:.1f}"), counts, last_ts

        if counts["trades_30s"] < self.min_trades_30s or counts["marks_30s"] < self.min_marks_30s:
            feat = self._build_features(symbol, now_sec, trades, marks, liqs, age_sec)
            detail = (
                f"trades_30s={counts['trades_30s']}<{self.min_trades_30s} "
                f"marks_30s={counts['marks_30s']}<{self.min_marks_30s}"
            )
            return feat, (False, "min_samples", detail), counts, last_ts

        feat = self._build_features(symbol, now_sec, trades, marks, liqs, age_sec)
        if feat is None:
            return None, (False, "no_data", "feature_build_failed"), counts, last_ts
        return feat, (True, "ok", "ok"), counts, last_ts

    def _build_features(
        self,
        symbol: str,
        now_sec: float,
        trades: list[dict[str, Any]],
        marks: list[dict[str, Any]],
        liqs: list[dict[str, Any]],
        age_sec: float,
    ) -> Optional[MicroFeatures]:
        if not trades and not marks:
            return None

        mark_last = 0.0
        if marks:
            mark_last = float(sorted(marks, key=lambda x: x["ts_sec"])[-1]["mark_price"] or 0.0)
        trade_last_px = 0.0
        if trades:
            trade_last_px = float(sorted(trades, key=lambda x: x["ts_sec"])[-1]["price"] or 0.0)
        if mark_last <= 0 and trade_last_px > 0:
            mark_last = trade_last_px

        window_30 = [r for r in trades if (now_sec - r["ts_sec"]) <= 30.0]
        qty_sum_30 = sum(float(r.get("qty") or 0.0) for r in window_30)
        buy_qty_30 = 0.0
        for r in window_30:
            side = str(r.get("side") or "").lower()
            is_bm = r.get("is_buyer_maker")
            if side in ("buy", "b"):
                buy_qty_30 += float(r.get("qty") or 0.0)
            elif side in ("sell", "s"):
                continue
            elif is_bm is not None:
                # Binance agg_trades: is_buyer_maker=True => taker is sell; False => taker is buy.
                if int(is_bm) == 0:
                    buy_qty_30 += float(r.get("qty") or 0.0)
        buy_ratio = (buy_qty_30 / qty_sum_30) if qty_sum_30 > 0 else 0.5
        imbalance_signed = (2.0 * buy_ratio) - 1.0

        # Keep units aligned with offline/backtest feature builders:
        # trade_intensity is "trades per minute equivalent".
        # With a 30s window this is count * (60 / 30) = count * 2.
        trade_intensity = float(len(window_30)) * (60.0 / 30.0)
        spread_proxy = 0.0
        if mark_last > 0 and trade_last_px > 0:
            spread_proxy = abs(trade_last_px - mark_last) / mark_last

        return MicroFeatures(
            timestamp=float(now_sec),
            symbol=symbol,
            imbalance=abs(float(imbalance_signed)),
            imbalance_signed=float(imbalance_signed),
            trade_intensity=float(trade_intensity),
            spread=float(max(0.0, spread_proxy)),
            mark_price=float(max(0.0, mark_last)),
            age_sec=float(max(0.0, age_sec)),
        )

    def _load_windows(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        now_sec: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        since_sec = now_sec - float(self.lookback_sec)
        return (
            self._load_table_rows(conn, "agg_trades", symbol, since_sec),
            self._load_table_rows(conn, "mark_prices", symbol, since_sec),
            self._load_table_rows(conn, "liquidations", symbol, since_sec),
        )

    def _load_table_rows(
        self,
        conn: sqlite3.Connection,
        table: str,
        symbol: str,
        since_sec: float,
    ) -> list[dict[str, Any]]:
        cols = _table_cols(conn, table)
        if not cols:
            return []

        sym_col = _pick_col(cols, ("symbol", "s"))
        ts_col = _pick_col(cols, ("ts_ms", "timestamp_ms", "ts", "timestamp", "event_ts", "time_ms", "time"))
        if not sym_col or not ts_col:
            return []

        select_cols = [sym_col, ts_col]
        if table == "agg_trades":
            price_col = _pick_col(cols, ("price", "p", "trade_price"))
            qty_col = _pick_col(cols, ("quantity", "qty", "size", "q"))
            bm_col = _pick_col(cols, ("is_buyer_maker",))
            side_col = _pick_col(cols, ("side", "taker_side"))
            for c in (price_col, qty_col, bm_col, side_col):
                if c and c not in select_cols:
                    select_cols.append(c)
        elif table == "mark_prices":
            mark_col = _pick_col(cols, ("mark_price", "price", "p"))
            if mark_col and mark_col not in select_cols:
                select_cols.append(mark_col)
        elif table == "liquidations":
            qty_col = _pick_col(cols, ("quantity", "qty", "size", "q"))
            not_col = _pick_col(cols, ("notional", "quote_qty", "quote_quantity"))
            side_col = _pick_col(cols, ("side", "s"))
            for c in (qty_col, not_col, side_col):
                if c and c not in select_cols:
                    select_cols.append(c)

        pinfo = conn.execute(f"PRAGMA table_info({table})").fetchall()
        text_ts = _is_text_col(pinfo, ts_col)

        if text_ts:
            cutoff = dt.datetime.utcfromtimestamp(since_sec).isoformat() + "Z"
            sql = f"SELECT {', '.join(select_cols)} FROM {table} WHERE {sym_col}=? AND {ts_col}>=? ORDER BY {ts_col} DESC LIMIT 20000"
            rows = conn.execute(sql, (symbol, cutoff)).fetchall()
        else:
            # heuristic for sec/ms column scale
            scale = 1.0
            probe = conn.execute(f"SELECT {ts_col} FROM {table} WHERE {sym_col}=? ORDER BY {ts_col} DESC LIMIT 1", (symbol,)).fetchone()
            if probe is not None:
                t0 = _parse_ts_any(probe[0])
                # if raw ts parses to sec but original value seems ms-like, cutoff in ms
                try:
                    raw_probe = float(probe[0])
                    if raw_probe > 1e12:
                        scale = 1000.0
                except Exception:
                    if t0 > 1e11:
                        scale = 1000.0
            cutoff_num = float(since_sec) * scale
            sql = f"SELECT {', '.join(select_cols)} FROM {table} WHERE {sym_col}=? AND {ts_col}>=? ORDER BY {ts_col} DESC LIMIT 20000"
            rows = conn.execute(sql, (symbol, cutoff_num)).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            rd = dict(r)
            ts_sec = _parse_ts_any(rd.get(ts_col))
            if ts_sec <= 0:
                continue
            item: dict[str, Any] = {"ts_sec": ts_sec, "symbol": symbol}
            if table == "agg_trades":
                item["price"] = _safe_float(rd.get(_pick_col(cols, ("price", "p", "trade_price"))), 0.0)
                item["qty"] = _safe_float(rd.get(_pick_col(cols, ("quantity", "qty", "size", "q"))), 0.0)
                bm_name = _pick_col(cols, ("is_buyer_maker",))
                side_name = _pick_col(cols, ("side", "taker_side"))
                item["is_buyer_maker"] = rd.get(bm_name) if bm_name else None
                item["side"] = (str(rd.get(side_name)).lower() if side_name and rd.get(side_name) is not None else "")
            elif table == "mark_prices":
                item["mark_price"] = _safe_float(rd.get(_pick_col(cols, ("mark_price", "price", "p"))), 0.0)
            else:
                item["qty"] = _safe_float(rd.get(_pick_col(cols, ("quantity", "qty", "size", "q"))), 0.0)
                item["notional"] = _safe_float(rd.get(_pick_col(cols, ("notional", "quote_qty", "quote_quantity"))), 0.0)
                side_name = _pick_col(cols, ("side", "s"))
                item["side"] = (str(rd.get(side_name)).lower() if side_name and rd.get(side_name) is not None else "")
            out.append(item)
        out.sort(key=lambda x: x["ts_sec"])
        return out
