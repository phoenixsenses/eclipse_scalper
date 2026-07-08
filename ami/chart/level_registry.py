"""BATCH-P4-003 + BATCH-P6-000 remediation: Level object registry
(Chart-Native Extension §4.3).

Implemented level_type sources in v1 (real, derivable from what P4-001/002
already built): SESSION_HIGH/LOW, PREVIOUS_DAY_HIGH/LOW, SWING_HIGH/LOW
(directly from ami_swings). NOT_IMPLEMENTED (documented limitation, not
fabricated): OPENING_RANGE_HIGH/LOW, EQUAL_HIGH/LOW_ZONE, BREAKOUT_LEVEL,
VWAP, ANCHORED_VWAP, VOLUME_PROFILE_HVN/LVN, CHANNEL_BOUNDARY, TRENDLINE,
CUSTOM_OPERATOR_LEVEL -- these need volume-profile/breakout/channel
infrastructure not yet built (later Phase 4/6 waves).

Session hour boundaries replicate tools/s34_state_machine_live_executor.py's
session_label() (read-only reference for consistency; that file is
protected and is not imported from here): ASIA [0,7) UTC, EUROPE [7,13),
US [13,21), OFF [21,24).

known_at_ts discipline: a session/day/swing level is only knowable once its
origin window has fully elapsed. Two rules enforced (FABLE REVIEW B F-B4/F-B1):

  F-B4 (truncation): if the lookback window itself starts mid-session or
  mid-day, the first observed period is INCOMPLETE (its true high/low
  cannot be known) and must be skipped entirely -- never emitted as if it
  were a real period extremum. Detected via: does the period's first
  candle actually start at the period's true boundary (within one bar)?

  F-B1 (boundary vs last-candle): known_at_ts must be the LATER of (a) the
  last member candle's close and (b) the period's own boundary-end
  timestamp. If a gap leaves the last candle closing before the period
  truly ends, the period is not yet "over" merely because data stopped --
  known_at_ts must not claim knowledge earlier than the real boundary.

Touch/rejection/acceptance stats (provisional v1 definition, not trained):
a candle "touches" a level if low<=price<=high. For a HIGH-type level,
ACCEPTANCE = close > price (broke and held above); REJECTION = close <= price
(touched then rejected back below). Mirrored for LOW-type levels.
strength_score = float(touch_count) (naive proxy, documented as provisional).

F-B2: these three stats are a SINGLE cumulative aggregate computed once, as
of build time, over every candle from known_at_ts through whatever is
currently the newest stored candle. They are NOT point-in-time-safe --
"touches as of build time" is a different, larger number than "touches as
of any earlier anchor timestamp a Phase 6 engine might be evaluating".
`touch_stats_point_in_time` is stored as 0 (False) for exactly this reason
and must stay 0 until a real point-in-time recomputation engine exists.
Any consumer must go through ami.research.feature_gateway (BATCH-P6-001),
which refuses to release these columns while the flag is 0.
"""
from __future__ import annotations
import hashlib
import time

from ami.enums import TIMEFRAME_MS

LEVEL_DEFINITION_VERSION = "level-v2-boundary-safe"

_HIGH_LEVEL_TYPES = {"SESSION_HIGH", "PREVIOUS_DAY_HIGH", "SWING_HIGH"}
_LOW_LEVEL_TYPES = {"SESSION_LOW", "PREVIOUS_DAY_LOW", "SWING_LOW"}

DAY_MS = 86_400_000
_SESSION_BOUNDS_HOURS = {"ASIA": (0, 7), "EUROPE": (7, 13), "US": (13, 21), "OFF": (21, 24)}


def _level_id(symbol: str, level_type: str, timeframe: str, origin_ts: int) -> str:
    key = f"{symbol}|{level_type}|{timeframe}|{origin_ts}|{LEVEL_DEFINITION_VERSION}"
    return "LVL-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _session_of_hour(hour: int) -> str:
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "EUROPE"
    if hour < 21:
        return "US"
    return "OFF"


def _session_bucket_start(ts_ms: int) -> tuple[int, str]:
    import datetime as dt

    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
    sess = _session_of_hour(d.hour)
    day_start = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    lo, _ = _SESSION_BOUNDS_HOURS[sess]
    return day_start + lo * 3600_000, sess


def compute_session_levels(candles: list[dict], symbol: str, timeframe: str) -> list[dict]:
    bar_ms = TIMEFRAME_MS[timeframe]
    groups: dict[tuple, list[dict]] = {}
    session_of_bucket: dict[int, str] = {}
    for c in candles:
        bucket_start, sess = _session_bucket_start(c["open_ts_ms"])
        groups.setdefault(bucket_start, []).append(c)
        session_of_bucket[bucket_start] = sess

    levels = []
    sorted_starts = sorted(groups)
    for i, start in enumerate(sorted_starts):
        if i + 1 >= len(sorted_starts):
            continue  # last (possibly still-open) session -- not yet fully elapsed
        members = groups[start]
        first_member_ts = min(c["open_ts_ms"] for c in members)
        if first_member_ts - start >= bar_ms:
            continue  # F-B4: period truncated at the lookback window's edge -- skip, do not fabricate
        lo, hi = _SESSION_BOUNDS_HOURS[session_of_bucket[start]]
        boundary_end_ts = start + (hi - lo) * 3600_000
        known_at_ts = max(members[-1]["close_ts_ms"], boundary_end_ts)  # F-B1
        levels.append({
            "level_type": "SESSION_HIGH", "price": max(c["high"] for c in members),
            "origin_ts": start, "known_at_ts": known_at_ts, "source_type": "SESSION",
        })
        levels.append({
            "level_type": "SESSION_LOW", "price": min(c["low"] for c in members),
            "origin_ts": start, "known_at_ts": known_at_ts, "source_type": "SESSION",
        })
    return levels


def compute_previous_day_levels(candles: list[dict], symbol: str, timeframe: str) -> list[dict]:
    bar_ms = TIMEFRAME_MS[timeframe]
    groups: dict[int, list[dict]] = {}
    for c in candles:
        day_start = (c["open_ts_ms"] // DAY_MS) * DAY_MS
        groups.setdefault(day_start, []).append(c)

    levels = []
    sorted_days = sorted(groups)
    for i, day_start in enumerate(sorted_days):
        if i + 1 >= len(sorted_days):
            continue  # last (possibly still-open) day -- not yet fully elapsed
        members = groups[day_start]
        first_member_ts = min(c["open_ts_ms"] for c in members)
        if first_member_ts - day_start >= bar_ms:
            continue  # F-B4: period truncated at the lookback window's edge -- skip, do not fabricate
        boundary_end_ts = day_start + DAY_MS
        known_at_ts = max(members[-1]["close_ts_ms"], boundary_end_ts)  # F-B1
        levels.append({
            "level_type": "PREVIOUS_DAY_HIGH", "price": max(c["high"] for c in members),
            "origin_ts": day_start, "known_at_ts": known_at_ts, "source_type": "DAY",
        })
        levels.append({
            "level_type": "PREVIOUS_DAY_LOW", "price": min(c["low"] for c in members),
            "origin_ts": day_start, "known_at_ts": known_at_ts, "source_type": "DAY",
        })
    return levels


def compute_swing_levels(swings: list[dict]) -> list[dict]:
    levels = []
    for s in swings:
        level_type = "SWING_HIGH" if s["swing_type"] == "HIGH" else "SWING_LOW"
        levels.append({
            "level_type": level_type, "price": s["pivot_price"],
            "origin_ts": s["pivot_ts"], "known_at_ts": s["known_at_ts"], "source_type": "SWING",
        })
    return levels


def _touch_stats(level_type: str, price: float, known_at_ts: int, candles: list[dict]) -> dict:
    is_high_type = level_type in _HIGH_LEVEL_TYPES
    touch_count = rejection_count = acceptance_count = 0
    last_touch_ts = None
    for c in candles:
        if c["open_ts_ms"] < known_at_ts:
            continue
        if c["low"] <= price <= c["high"]:
            touch_count += 1
            last_touch_ts = c["close_ts_ms"]
            if is_high_type:
                if c["close"] > price:
                    acceptance_count += 1
                else:
                    rejection_count += 1
            else:
                if c["close"] < price:
                    acceptance_count += 1
                else:
                    rejection_count += 1
    return {
        "touch_count": touch_count, "rejection_count": rejection_count,
        "acceptance_count": acceptance_count, "last_touch_ts": last_touch_ts,
        "strength_score": float(touch_count),
    }


_SUPERSEDED_LEVEL_DEFINITION_VERSIONS = ["level-v1"]  # F-B4/F-B1: known-buggy, controlled cleanup


def _delete_superseded_levels(conn, symbol: str, timeframe: str) -> int:
    """Controlled, reproducible cleanup of rows produced by the pre-BATCH-P6-000
    level_registry (window-edge truncation bug F-B4 + boundary/known_at bug
    F-B1). These are software defects, not research verdicts -- they are not
    "rejected findings" subject to the never-delete rule; re-running this
    function is idempotent (0 rows deleted once the superseded version is
    already gone)."""
    n = 0
    for old_version in _SUPERSEDED_LEVEL_DEFINITION_VERSIONS:
        cur = conn.execute(
            "DELETE FROM ami_levels WHERE symbol=? AND timeframe=? AND level_definition_version=?",
            (symbol, timeframe, old_version),
        )
        n += cur.rowcount
    return n


def seed(conn, symbol: str = "ETHUSDT", timeframe: str = "1m",
         provenance: str = "batch-p6-000-level-registry-boundary-safe") -> int:
    now = int(time.time() * 1000)
    deleted = _delete_superseded_levels(conn, symbol, timeframe)
    if deleted:
        conn.commit()

    rows = conn.execute(
        "SELECT open_ts_ms, close_ts_ms, high, low, close FROM ami_candles "
        "WHERE symbol=? AND timeframe=? ORDER BY open_ts_ms ASC",
        (symbol, timeframe),
    ).fetchall()
    candles = [{"open_ts_ms": r[0], "close_ts_ms": r[1], "high": r[2], "low": r[3], "close": r[4]} for r in rows]

    swing_rows = conn.execute(
        "SELECT swing_type, pivot_ts, pivot_price, known_at_ts FROM ami_swings WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchall()
    swings = [{"swing_type": r[0], "pivot_ts": r[1], "pivot_price": r[2], "known_at_ts": r[3]} for r in swing_rows]

    levels = (
        compute_session_levels(candles, symbol, timeframe)
        + compute_previous_day_levels(candles, symbol, timeframe)
        + compute_swing_levels(swings)
    )

    n = 0
    for lv in levels:
        stats = _touch_stats(lv["level_type"], lv["price"], lv["known_at_ts"], candles)
        level_id = _level_id(symbol, lv["level_type"], timeframe, lv["origin_ts"])
        conn.execute(
            "INSERT INTO ami_levels (level_id, symbol, level_type, price, origin_ts, known_at_ts, "
            "timeframe, touch_count, rejection_count, acceptance_count, last_touch_ts, strength_score, "
            "touch_stats_point_in_time, source_type, level_definition_version, schema_version, "
            "provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(level_id) DO UPDATE SET touch_count=excluded.touch_count, "
            "rejection_count=excluded.rejection_count, acceptance_count=excluded.acceptance_count, "
            "last_touch_ts=excluded.last_touch_ts, strength_score=excluded.strength_score, "
            "updated_ms=excluded.updated_ms",
            (level_id, symbol, lv["level_type"], lv["price"], lv["origin_ts"], lv["known_at_ts"],
             timeframe, stats["touch_count"], stats["rejection_count"], stats["acceptance_count"],
             stats["last_touch_ts"], stats["strength_score"], 0,  # F-B2: never point-in-time-safe yet
             lv["source_type"], LEVEL_DEFINITION_VERSION,
             5, provenance, now, now),
        )
        n += 1
    conn.commit()
    return n


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"registered {n} levels (SESSION/PREVIOUS_DAY/SWING, ETHUSDT 1m)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
