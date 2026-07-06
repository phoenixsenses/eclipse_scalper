"""BATCH-P4-001: Candle object foundation (Chart-Native Extension §4.1/§6.1).

Source: data/microstructure.db:agg_trades (read-only, real trade-level data,
indexed on (symbol, ts_ms)). Candles are built directly from trades, not
from a pre-existing kline table -- this lets us apply the known-at/
closed-candle discipline ourselves rather than trust an upstream source's
notion of "closed".

Closed-candle-only (Observatory §6.3): a bucket is only emitted if its
close_ts_ms <= as_of_ms (the reference "now" passed to build_candles). A
still-forming bucket is skipped entirely -- never stored as PARTIAL_CANDLE.
known_at_ts = close_ts_ms (the earliest moment the candle is valid to use).

No fabrication: a bucket with zero trades is skipped (not stored as a
degenerate flat candle) -- gaps in the candle sequence mean no trades
occurred in that window, not "price stayed exactly flat".

BATCH-P6-000 F-B3: a bucket with >=1 trade is not automatically reliable --
the collector itself can have recorded outages (data/microstructure.db:gaps,
stream='agg_trades'). A candle whose window overlaps a known collector gap
is marked data_quality=GAPPED instead of AVAILABLE: trades WERE seen, but
coverage during part of the window is not guaranteed complete, so this is
a distinct state from a clean AVAILABLE bucket.
"""
from __future__ import annotations
import hashlib
import sqlite3
import time
from pathlib import Path

from ami.enums import TIMEFRAME_MS
from ami.timing.contract import CandleState, DataQualityState

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADES_DB = REPO_ROOT / "data" / "microstructure.db"
CANDLE_DEFINITION_VERSION = "candle-agg_trades-v1"


def _candle_id(symbol: str, timeframe: str, open_ts_ms: int) -> str:
    key = f"{symbol}|{timeframe}|{open_ts_ms}|{CANDLE_DEFINITION_VERSION}"
    return "CDL-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _load_agg_trades_gaps(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    try:
        rows = conn.execute("SELECT start_ts_ms, end_ts_ms FROM gaps WHERE stream='agg_trades'").fetchall()
    except sqlite3.OperationalError:
        return []  # gaps table not present in this DB (e.g. a test fixture) -- no gap info, not fabricated
    return [(r[0], r[1]) for r in rows if r[0] is not None and r[1] is not None]


def _overlaps_any_gap(bucket_open: int, bucket_close: int, gaps: list[tuple[int, int]]) -> bool:
    return any(bucket_open < g_end and bucket_close > g_start for g_start, g_end in gaps)


def build_candles(symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int,
                   as_of_ms: int | None = None, trades_db: Path = DEFAULT_TRADES_DB,
                   venue: str | None = None) -> list[dict]:
    """Real, closed-candle-only OHLCV construction from agg_trades.

    [start_ts_ms, end_ts_ms) is bucketed at `timeframe` granularity. Only
    buckets whose close_ts_ms <= as_of_ms (default: now) are considered --
    still-forming buckets are silently skipped, not stored.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    bar_ms = TIMEFRAME_MS[timeframe]
    as_of = as_of_ms if as_of_ms is not None else int(time.time() * 1000)

    conn = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True)
    try:
        gaps = _load_agg_trades_gaps(conn)
        candles = []
        bucket_open = start_ts_ms - (start_ts_ms % bar_ms)
        while bucket_open < end_ts_ms:
            bucket_close = bucket_open + bar_ms
            if bucket_close > as_of:
                break  # still-forming or future bucket -- never stored
            rows = conn.execute(
                "SELECT price, quantity, is_buyer_maker FROM agg_trades "
                "WHERE symbol=? AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms ASC",
                (symbol, bucket_open, bucket_close),
            ).fetchall()
            if rows:
                prices = [r[0] for r in rows]
                open_px, close_px = prices[0], prices[-1]
                high_px, low_px = max(prices), min(prices)
                volume = sum(r[1] for r in rows)
                # is_buyer_maker=1 -> the resting order was a bid (buyer); the
                # taker was the seller. is_buyer_maker=0 -> taker was the buyer.
                taker_buy = sum(r[1] for r in rows if r[2] == 0)
                taker_sell = sum(r[1] for r in rows if r[2] == 1)
                data_quality = (DataQualityState.GAPPED.value
                                if _overlaps_any_gap(bucket_open, bucket_close, gaps)
                                else DataQualityState.AVAILABLE.value)
                candles.append({
                    "candle_id": _candle_id(symbol, timeframe, bucket_open),
                    "symbol": symbol,
                    "venue": venue,
                    "timeframe": timeframe,
                    "open_ts_ms": bucket_open,
                    "close_ts_ms": bucket_close,
                    "open": open_px, "high": high_px, "low": low_px, "close": close_px,
                    "volume": volume,
                    "trade_count": len(rows),
                    "taker_buy_volume": taker_buy,
                    "taker_sell_volume": taker_sell,
                    "is_closed": 1,
                    "partial_status": CandleState.CLOSED.value,
                    "known_at_ts": bucket_close,
                    "data_quality": data_quality,
                    "source_hash": None,
                    "candle_definition_version": CANDLE_DEFINITION_VERSION,
                })
            bucket_open = bucket_close
    finally:
        conn.close()
    return candles


def build_candles_streaming(symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int,
                             as_of_ms: int | None = None, trades_db: Path = DEFAULT_TRADES_DB,
                             venue: str | None = None, row_chunk: int = 200_000) -> list[dict]:
    """Same contract and output shape as build_candles(), but scans agg_trades
    in ONE ORDER BY ts_ms pass (via cursor.fetchmany chunks) instead of one
    query per bucket. Needed for full-history backfills (BATCH-P6-003b): the
    per-bucket approach means one small query per minute -- for ~140 days
    that is ~200K round trips, whereas a single ordered range scan reads the
    same ~173M-row table once. fetchmany() keeps peak memory bounded to
    `row_chunk` rows (not the whole table) since results still arrive in
    ts_ms order and are aggregated into the current bucket as they stream by.

    Cross-checked against build_candles() for exact output equality on the
    existing (small, already-verified) window before being used for anything
    new (test_candle_builder_streaming_matches_reference_on_same_window)."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    bar_ms = TIMEFRAME_MS[timeframe]
    as_of = as_of_ms if as_of_ms is not None else int(time.time() * 1000)

    conn = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True)
    try:
        gaps = _load_agg_trades_gaps(conn)
        candles = []

        def flush(bucket_open, bucket_close, rows):
            if not rows or bucket_close > as_of:
                return
            prices = [r[1] for r in rows]
            open_px, close_px = prices[0], prices[-1]
            high_px, low_px = max(prices), min(prices)
            volume = sum(r[2] for r in rows)
            taker_buy = sum(r[2] for r in rows if r[3] == 0)
            taker_sell = sum(r[2] for r in rows if r[3] == 1)
            data_quality = (DataQualityState.GAPPED.value
                            if _overlaps_any_gap(bucket_open, bucket_close, gaps)
                            else DataQualityState.AVAILABLE.value)
            candles.append({
                "candle_id": _candle_id(symbol, timeframe, bucket_open),
                "symbol": symbol,
                "venue": venue,
                "timeframe": timeframe,
                "open_ts_ms": bucket_open,
                "close_ts_ms": bucket_close,
                "open": open_px, "high": high_px, "low": low_px, "close": close_px,
                "volume": volume,
                "trade_count": len(rows),
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": taker_sell,
                "is_closed": 1,
                "partial_status": CandleState.CLOSED.value,
                "known_at_ts": bucket_close,
                "data_quality": data_quality,
                "source_hash": None,
                "candle_definition_version": CANDLE_DEFINITION_VERSION,
            })

        # bucket-align the SQL lower bound too -- otherwise the first partial
        # bucket [bucket_open, start_ts_ms) is silently truncated (found via
        # cross-check against build_candles() on real data: trade_count/OHLC
        # mismatched on the very first bucket of the window).
        bucket_open = start_ts_ms - (start_ts_ms % bar_ms)
        bucket_close = bucket_open + bar_ms
        bucket_rows: list[tuple] = []

        # Matches build_candles()'s own loop bound (`while bucket_open <
        # end_ts_ms`) exactly -- assumes as_of_ms <= end_ts_ms (true for
        # every call site here: seed_full_history always passes
        # as_of_ms=end_ts_ms), so the as_of gate in flush() never needs to
        # see rows past end_ts_ms.
        cur = conn.execute(
            "SELECT ts_ms, price, quantity, is_buyer_maker FROM agg_trades "
            "WHERE symbol=? AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms ASC",
            (symbol, bucket_open, end_ts_ms),
        )
        while True:
            chunk = cur.fetchmany(row_chunk)
            if not chunk:
                break
            for row in chunk:
                ts = row[0]
                while ts >= bucket_close:
                    flush(bucket_open, bucket_close, bucket_rows)
                    bucket_rows = []
                    bucket_open = bucket_close
                    bucket_close = bucket_open + bar_ms
                bucket_rows.append(row)
        flush(bucket_open, bucket_close, bucket_rows)
    finally:
        conn.close()
    return candles


def _upsert_candles(conn, candles: list[dict], schema_version: int, provenance: str, now: int) -> int:
    n = 0
    for c in candles:
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, "
            "open, high, low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, "
            "is_closed, partial_status, known_at_ts, data_quality, source_hash, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, open_ts_ms, candle_definition_version) DO UPDATE SET "
            "close=excluded.close, high=excluded.high, low=excluded.low, volume=excluded.volume, "
            "trade_count=excluded.trade_count, data_quality=excluded.data_quality, "
            "updated_ms=excluded.updated_ms",
            (c["candle_id"], c["symbol"], c["venue"], c["timeframe"], c["open_ts_ms"], c["close_ts_ms"],
             c["open"], c["high"], c["low"], c["close"], c["volume"], c["trade_count"],
             c["taker_buy_volume"], c["taker_sell_volume"], c["is_closed"], c["partial_status"],
             c["known_at_ts"], c["data_quality"], c["source_hash"], c["candle_definition_version"],
             schema_version, provenance, now, now),
        )
        n += 1
    return n


def derive_higher_timeframe(base_candles: list[dict], symbol: str, timeframe: str) -> list[dict]:
    """Aggregates already-built 1m candles into a coarser timeframe (e.g. 5m)
    WITHOUT a second agg_trades scan -- base_candles is presumed sorted by
    open_ts_ms and already closed-only/gap-aware. A higher-TF bucket with
    zero contributing base candles is skipped (no fabrication, same rule as
    build_candles). data_quality is GAPPED if ANY contributing base candle
    is GAPPED (conservative propagation, never silently upgraded)."""
    bar_ms = TIMEFRAME_MS[timeframe]
    groups: dict[int, list[dict]] = {}
    for c in base_candles:
        bucket_open = c["open_ts_ms"] - (c["open_ts_ms"] % bar_ms)
        groups.setdefault(bucket_open, []).append(c)

    out = []
    for bucket_open in sorted(groups):
        members = sorted(groups[bucket_open], key=lambda c: c["open_ts_ms"])
        bucket_close = bucket_open + bar_ms
        out.append({
            "candle_id": _candle_id(symbol, timeframe, bucket_open),
            "symbol": symbol,
            "venue": members[0]["venue"],
            "timeframe": timeframe,
            "open_ts_ms": bucket_open,
            "close_ts_ms": bucket_close,
            "open": members[0]["open"],
            "high": max(m["high"] for m in members),
            "low": min(m["low"] for m in members),
            "close": members[-1]["close"],
            "volume": sum(m["volume"] for m in members),
            "trade_count": sum(m["trade_count"] for m in members),
            "taker_buy_volume": sum(m["taker_buy_volume"] for m in members),
            "taker_sell_volume": sum(m["taker_sell_volume"] for m in members),
            "is_closed": 1,
            "partial_status": CandleState.CLOSED.value,
            "known_at_ts": members[-1]["known_at_ts"],
            "data_quality": (DataQualityState.GAPPED.value
                             if any(m["data_quality"] == DataQualityState.GAPPED.value for m in members)
                             else DataQualityState.AVAILABLE.value),
            "source_hash": None,
            "candle_definition_version": CANDLE_DEFINITION_VERSION,
        })
    return out


def seed(conn, symbol: str = "ETHUSDT", timeframes: tuple[str, ...] = ("1m", "5m"),
         lookback_hours: int = 48, provenance: str = "batch-p4-001-candle-builder",
         end_ts_ms: int | None = None) -> int:
    """Idempotent upsert into ami_candles. Bounded lookback window by design
    (small, reversible first batch) -- a fuller historical backfill is later
    research-wave work, not this foundational batch.

    `end_ts_ms` [BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1-EVIDENCE-
    RECONCILIATION]: optional, resolved at call time, defaults to None -> the
    original `now = time.time()` behavior, unchanged for every existing
    caller. Lets a test pin the window to a fixed historical timestamp
    instead of "now" -- same test-patchable pattern already used by
    `seed_full_history`'s `start_ts_ms`/`end_ts_ms`. No collector/runtime
    semantics changed; this function has no relationship to the live
    collector process, it only reads already-collected rows from
    `data/microstructure.db`."""
    now = end_ts_ms if end_ts_ms is not None else int(time.time() * 1000)
    start_ts_ms = now - lookback_hours * 3600_000
    n = 0
    for tf in timeframes:
        candles = build_candles(symbol, tf, start_ts_ms, now, as_of_ms=now)
        n += _upsert_candles(conn, candles, 5, provenance, now)
    conn.commit()
    return n


def seed_full_history(conn, symbol: str = "ETHUSDT", start_ts_ms: int | None = None,
                       end_ts_ms: int | None = None, trades_db: Path | None = None,
                       provenance: str = "batch-p6-003b-candle-full-history") -> dict:
    """BATCH-P6-003b: full-history backfill for W2 (needs anchor_n comparable
    to the all-history W1 population, not just the Phase 4 48h window).
    Builds 1m via the single-pass streaming scan (one agg_trades read, not
    ~200K per-minute queries), then derives 5m from the 1m results in Python
    (no second expensive DB scan). Same idempotent upsert as seed()."""
    if trades_db is None:
        trades_db = DEFAULT_TRADES_DB  # resolved at call time (test-patchable), not def-time-bound
    now = int(time.time() * 1000)
    if end_ts_ms is None:
        end_ts_ms = now
    if start_ts_ms is None:
        conn_trades = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True)
        try:
            start_ts_ms = conn_trades.execute(
                "SELECT MIN(ts_ms) FROM agg_trades WHERE symbol=?", (symbol,)
            ).fetchone()[0]
        finally:
            conn_trades.close()

    candles_1m = build_candles_streaming(symbol, "1m", start_ts_ms, end_ts_ms, as_of_ms=end_ts_ms,
                                          trades_db=trades_db)
    candles_5m = derive_higher_timeframe(candles_1m, symbol, "5m")

    n_1m = _upsert_candles(conn, candles_1m, 7, provenance, now)
    n_5m = _upsert_candles(conn, candles_5m, 7, provenance, now)
    conn.commit()
    return {"n_1m": n_1m, "n_5m": n_5m, "start_ts_ms": start_ts_ms, "end_ts_ms": end_ts_ms}


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"built {n} closed candles (ETHUSDT, 1m+5m, 48h lookback)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
