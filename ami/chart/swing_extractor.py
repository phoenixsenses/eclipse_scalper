"""BATCH-P4-002: confirmed Swing object extraction (Chart-Native Extension §4.2).

"pivot_ts and known_at_ts must be separate. A swing high is not knowable at
its exact peak until the confirmation rule is satisfied." (§4.2)

Confirmation rule (v1, versioned as SWING_DEFINITION_VERSION): symmetric
N-bar fractal. A candle at position i is a confirmed swing HIGH if its
`high` is the maximum within the window [i-N, i+N] (positional, not
time-based -- a gap from a skipped empty bucket does not break the count).
known_at_ts = the close_ts_ms of the candle N bars after the pivot -- the
earliest moment the pivot could be confirmed, never the pivot's own
timestamp.

Deliberately NOT implemented in v1 (documented limitation, not fabricated):
asymmetric/adaptive confirmation strength, deduplication of adjacent
flat-top runs into a single swing. A future SWING_DEFINITION_VERSION can
add these without mutating v1's frozen rows.
"""
from __future__ import annotations
import hashlib
import time

SWING_DEFINITION_VERSION = "fractal-n3-v1"
CONFIRMATION_BARS = 3


def _swing_id(symbol: str, timeframe: str, swing_type: str, pivot_ts_ms: int) -> str:
    key = f"{symbol}|{timeframe}|{swing_type}|{pivot_ts_ms}|{SWING_DEFINITION_VERSION}"
    return "SWG-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def extract_swings(candles: list[dict], symbol: str, timeframe: str,
                    confirmation_bars: int = CONFIRMATION_BARS) -> list[dict]:
    """candles: chronologically sorted dicts with open_ts_ms, close_ts_ms, high, low.
    Returns confirmed swings only -- an unconfirmed potential pivot near the
    end of the series (fewer than `confirmation_bars` bars after it) is
    never emitted."""
    n = confirmation_bars
    swings = []
    for i in range(n, len(candles) - n):
        window = candles[i - n: i + n + 1]
        pivot = candles[i]
        confirming_candle = candles[i + n]

        highs = [c["high"] for c in window]
        if pivot["high"] == max(highs) and highs.count(max(highs)) == 1:
            left_low = min(c["low"] for c in candles[i - n:i])
            right_low = min(c["low"] for c in candles[i + 1:i + n + 1])
            prominence_bps = (pivot["high"] - max(left_low, right_low)) / pivot["high"] * 1e4
            swings.append({
                "swing_id": _swing_id(symbol, timeframe, "HIGH", pivot["open_ts_ms"]),
                "symbol": symbol, "timeframe": timeframe, "swing_type": "HIGH",
                "pivot_ts": pivot["open_ts_ms"], "pivot_price": pivot["high"],
                "confirmation_ts": confirming_candle["close_ts_ms"],
                "confirmation_method": SWING_DEFINITION_VERSION,
                "prominence_bps": prominence_bps,
                "duration_bars": n,
                "left_strength": n, "right_strength": n,
                "known_at_ts": confirming_candle["close_ts_ms"],
            })

        lows = [c["low"] for c in window]
        if pivot["low"] == min(lows) and lows.count(min(lows)) == 1:
            left_high = max(c["high"] for c in candles[i - n:i])
            right_high = max(c["high"] for c in candles[i + 1:i + n + 1])
            prominence_bps = (min(left_high, right_high) - pivot["low"]) / pivot["low"] * 1e4
            swings.append({
                "swing_id": _swing_id(symbol, timeframe, "LOW", pivot["open_ts_ms"]),
                "symbol": symbol, "timeframe": timeframe, "swing_type": "LOW",
                "pivot_ts": pivot["open_ts_ms"], "pivot_price": pivot["low"],
                "confirmation_ts": confirming_candle["close_ts_ms"],
                "confirmation_method": SWING_DEFINITION_VERSION,
                "prominence_bps": prominence_bps,
                "duration_bars": n,
                "left_strength": n, "right_strength": n,
                "known_at_ts": confirming_candle["close_ts_ms"],
            })
    return swings


def seed(conn, symbol: str = "ETHUSDT", timeframe: str = "1m",
         provenance: str = "batch-p4-002-swing-extractor") -> int:
    """Reads closed candles from ami_candles (already ingested by candle_builder);
    writes confirmed swings into ami_swings. Read-only relative to ami_candles."""
    now = int(time.time() * 1000)
    rows = conn.execute(
        "SELECT open_ts_ms, close_ts_ms, high, low FROM ami_candles "
        "WHERE symbol=? AND timeframe=? ORDER BY open_ts_ms ASC",
        (symbol, timeframe),
    ).fetchall()
    candles = [{"open_ts_ms": r[0], "close_ts_ms": r[1], "high": r[2], "low": r[3]} for r in rows]
    swings = extract_swings(candles, symbol, timeframe)
    for s in swings:
        conn.execute(
            "INSERT INTO ami_swings (swing_id, symbol, timeframe, swing_type, pivot_ts, pivot_price, "
            "confirmation_ts, confirmation_method, prominence_bps, duration_bars, left_strength, "
            "right_strength, known_at_ts, swing_definition_version, schema_version, provenance, "
            "created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(swing_id) DO UPDATE SET prominence_bps=excluded.prominence_bps, "
            "updated_ms=excluded.updated_ms",
            (s["swing_id"], s["symbol"], s["timeframe"], s["swing_type"], s["pivot_ts"], s["pivot_price"],
             s["confirmation_ts"], s["confirmation_method"], s["prominence_bps"], s["duration_bars"],
             s["left_strength"], s["right_strength"], s["known_at_ts"], SWING_DEFINITION_VERSION,
             5, provenance, now, now),
        )
    conn.commit()
    return len(swings)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"extracted {n} confirmed swings (ETHUSDT 1m)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
