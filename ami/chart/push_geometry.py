"""BATCH-P4-004: Push object (Chart-Native Extension §7.1).

A push connects two consecutive, alternating-type confirmed swings (LOW->HIGH
= UP, HIGH->LOW = DOWN). Two consecutive same-type swings (possible with the
simple v1 fractal extractor) are NOT paired into a push -- documented
limitation, not a fabricated pairing.

known_at_ts: a push's core geometry is knowable once its END swing is
confirmed (end_swing.known_at_ts). pullback_after_bps additionally depends
on the FOLLOWING push (the reversal after this push's end swing) -- if that
next push exists, known_at_ts advances to the next swing's known_at_ts;
otherwise pullback_after_bps stays None (not yet knowable) and known_at_ts
is just the end swing's confirmation time.

liquidation_notional is NOT_IMPLEMENTED in v1 (would require joining
ami_events by time-window -- a legitimate future refinement, not fabricated
here as None).
"""
from __future__ import annotations
import hashlib
import time

PUSH_DEFINITION_VERSION = "push-v1"


def _push_id(symbol: str, timeframe: str, start_swing_id: str, end_swing_id: str) -> str:
    key = f"{symbol}|{timeframe}|{start_swing_id}|{end_swing_id}|{PUSH_DEFINITION_VERSION}"
    return "PSH-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _bps(a: float, b: float) -> float:
    """Signed change from a to b, in basis points of a."""
    return (b - a) / a * 1e4


def build_pushes(swings: list[dict], candles: list[dict], symbol: str, timeframe: str) -> list[dict]:
    """swings: chronologically sorted confirmed swings (swing_id, swing_type,
    pivot_ts, pivot_price, known_at_ts). candles: chronologically sorted with
    open_ts_ms, close_ts_ms, close, volume -- used for path length/speed/volume.
    """
    candles_by_ts = {c["open_ts_ms"]: c for c in candles}
    sorted_ts = sorted(candles_by_ts)

    raw_pushes = []
    for i in range(len(swings) - 1):
        a, b = swings[i], swings[i + 1]
        if a["swing_type"] == b["swing_type"]:
            continue  # non-alternating pair -- not a push, documented limitation
        direction = "UP" if a["swing_type"] == "LOW" else "DOWN"
        start_ts, end_ts = a["pivot_ts"], b["pivot_ts"]
        if end_ts <= start_ts:
            continue

        window_candles = [candles_by_ts[t] for t in sorted_ts if start_ts <= t < end_ts]
        displacement_bps = _bps(a["pivot_price"], b["pivot_price"])
        duration_seconds = (end_ts - start_ts) / 1000.0

        # path_length must connect the swing's actual extreme prices
        # (pivot_price, which is a high/low, not a close) at both ends --
        # otherwise the path is missing the boundary segments and can end up
        # shorter than the net displacement, breaking the triangle-inequality
        # invariant efficiency_ratio = |displacement_bps| / path_length_bps <= 1.
        # All path segments use the push's own start price as a single fixed
        # reference (same base as displacement_bps). Per-step "speed" below
        # intentionally uses each step's own local close-to-close base -- a
        # legitimate local rate-of-change concept, not a cumulative distance.
        price_path = [a["pivot_price"]] + [c["close"] for c in window_candles] + [b["pivot_price"]]
        path_length_abs = sum(abs(price_path[k + 1] - price_path[k]) for k in range(len(price_path) - 1))
        path_length_bps = (path_length_abs / a["pivot_price"] * 1e4) if path_length_abs > 0 else 0.0

        peak_speed = 0.0
        speeds = []
        volume = 0.0
        for j in range(len(window_candles) - 1):
            c0, c1 = window_candles[j], window_candles[j + 1]
            step_bps_local = abs(_bps(c0["close"], c1["close"]))
            step_seconds = max((c1["open_ts_ms"] - c0["open_ts_ms"]) / 1000.0, 1e-9)
            step_speed = step_bps_local / step_seconds
            speeds.append(step_speed)
            peak_speed = max(peak_speed, step_speed)
        for c in window_candles:
            volume += c.get("volume") or 0.0

        average_speed = abs(displacement_bps) / duration_seconds if duration_seconds > 0 else None
        efficiency_ratio = (abs(displacement_bps) / path_length_bps) if path_length_bps > 0 else None

        acceleration = None
        if len(speeds) >= 2:
            mid = len(speeds) // 2
            first_half = speeds[:mid] or speeds[:1]
            second_half = speeds[mid:] or speeds[-1:]
            acceleration = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))

        raw_pushes.append({
            "push_id": _push_id(symbol, timeframe, a["swing_id"], b["swing_id"]),
            "symbol": symbol, "timeframe": timeframe,
            "direction": direction,
            "start_swing_id": a["swing_id"], "end_swing_id": b["swing_id"],
            "start_ts": start_ts, "end_ts": end_ts,
            "displacement_bps": displacement_bps,
            "duration_seconds": duration_seconds,
            "bars": len(window_candles),
            "path_length_bps": path_length_bps if path_length_bps > 0 else None,
            "efficiency_ratio": efficiency_ratio,
            "volume": volume,
            "liquidation_notional": None,  # NOT_IMPLEMENTED v1
            "average_speed": average_speed,
            "peak_speed": peak_speed if window_candles else None,
            "acceleration": acceleration,
            "pullback_after_bps": None,
            "known_at_ts": b["known_at_ts"],
        })

    for i in range(len(raw_pushes) - 1):
        raw_pushes[i]["pullback_after_bps"] = abs(raw_pushes[i + 1]["displacement_bps"])
        raw_pushes[i]["known_at_ts"] = max(raw_pushes[i]["known_at_ts"], raw_pushes[i + 1]["known_at_ts"])

    return raw_pushes


def seed(conn, symbol: str = "ETHUSDT", timeframe: str = "1m",
         provenance: str = "batch-p4-004-push-geometry") -> int:
    now = int(time.time() * 1000)
    swing_rows = conn.execute(
        "SELECT swing_id, swing_type, pivot_ts, pivot_price, known_at_ts FROM ami_swings "
        "WHERE symbol=? AND timeframe=? ORDER BY pivot_ts ASC",
        (symbol, timeframe),
    ).fetchall()
    swings = [{"swing_id": r[0], "swing_type": r[1], "pivot_ts": r[2], "pivot_price": r[3], "known_at_ts": r[4]}
              for r in swing_rows]

    candle_rows = conn.execute(
        "SELECT open_ts_ms, close_ts_ms, close, volume FROM ami_candles WHERE symbol=? AND timeframe=? "
        "ORDER BY open_ts_ms ASC",
        (symbol, timeframe),
    ).fetchall()
    candles = [{"open_ts_ms": r[0], "close_ts_ms": r[1], "close": r[2], "volume": r[3]} for r in candle_rows]

    pushes = build_pushes(swings, candles, symbol, timeframe)
    for p in pushes:
        conn.execute(
            "INSERT INTO ami_pushes (push_id, symbol, timeframe, direction, start_swing_id, end_swing_id, "
            "start_ts, end_ts, displacement_bps, duration_seconds, bars, path_length_bps, efficiency_ratio, "
            "volume, liquidation_notional, average_speed, peak_speed, acceleration, pullback_after_bps, "
            "known_at_ts, push_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(push_id) DO UPDATE SET pullback_after_bps=excluded.pullback_after_bps, "
            "known_at_ts=excluded.known_at_ts, path_length_bps=excluded.path_length_bps, "
            "efficiency_ratio=excluded.efficiency_ratio, average_speed=excluded.average_speed, "
            "peak_speed=excluded.peak_speed, acceleration=excluded.acceleration, "
            "updated_ms=excluded.updated_ms",
            (p["push_id"], p["symbol"], p["timeframe"], p["direction"], p["start_swing_id"], p["end_swing_id"],
             p["start_ts"], p["end_ts"], p["displacement_bps"], p["duration_seconds"], p["bars"],
             p["path_length_bps"], p["efficiency_ratio"], p["volume"], p["liquidation_notional"],
             p["average_speed"], p["peak_speed"], p["acceleration"], p["pullback_after_bps"],
             p["known_at_ts"], PUSH_DEFINITION_VERSION, 5, provenance, now, now),
        )
    conn.commit()
    return len(pushes)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"built {n} pushes (ETHUSDT 1m)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
