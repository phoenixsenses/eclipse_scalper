"""BATCH-P4-001: atomic candle morphology (Chart-Native Extension §6.1/§6.3).

Only the atomic (single-candle) feature set and a 3-way close-quality label
are implemented in v1. REJECTION_CLOSE/ACCEPTANCE_CLOSE/INDECISION_CLOSE/
FOLLOW_THROUGH_CONFIRMED/FAILED are NOT_IMPLEMENTED here -- they need
multi-candle and level context (Level object, BATCH-P4-003) and are
documented as deferred, not fabricated.

MORPHOLOGY_DEFINITION_VERSION thresholds (0.7 / 0.3 split for close-location)
are provisional engineering defaults, not trained on data (CN §6.3: "must be
versioned and selected on training data only" -- an actual train-selected
threshold is Phase 6 research work). Versioned here so a future trained
threshold set does not silently overwrite this one.
"""
from __future__ import annotations
import time

MORPHOLOGY_DEFINITION_VERSION = "morphology-provisional-v1"

_NEAR_HIGH_THRESHOLD = 0.7
_NEAR_LOW_THRESHOLD = 0.3


def compute_morphology(candle: dict) -> dict:
    """candle: dict with open/high/low/close (as in ami_candles). Returns a
    dict of morphology fields; ratio fields are None (not 0) when range==0
    (all trades printed at one price -- ratios are mathematically undefined,
    not fabricated as zero)."""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = h - l
    if rng <= 0:
        return {
            "range_abs": rng, "body_ratio": None, "upper_wick_ratio": None,
            "lower_wick_ratio": None, "close_location_value": None,
            "open_location_value": None, "directional_body": None,
            "close_quality_label": None,
        }
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    close_location_value = (c - l) / rng
    open_location_value = (o - l) / rng
    directional_body = (c - o) / rng

    if close_location_value >= _NEAR_HIGH_THRESHOLD:
        label = "CLOSE_NEAR_HIGH"
    elif close_location_value <= _NEAR_LOW_THRESHOLD:
        label = "CLOSE_NEAR_LOW"
    else:
        label = "MID_RANGE_CLOSE"

    return {
        "range_abs": rng,
        "body_ratio": body / rng,
        "upper_wick_ratio": upper_wick / rng,
        "lower_wick_ratio": lower_wick / rng,
        "close_location_value": close_location_value,
        "open_location_value": open_location_value,
        "directional_body": directional_body,
        "close_quality_label": label,
    }


def seed(conn, provenance: str = "batch-p4-001-candle-morphology") -> int:
    """Reads ami_candles from the warehouse; writes ami_candle_morphology (1:1)."""
    now = int(time.time() * 1000)
    rows = conn.execute("SELECT candle_id, open, high, low, close FROM ami_candles").fetchall()
    n = 0
    for candle_id, o, h, l, c in rows:
        m = compute_morphology({"open": o, "high": h, "low": l, "close": c})
        conn.execute(
            "INSERT INTO ami_candle_morphology (candle_id, range_abs, body_ratio, upper_wick_ratio, "
            "lower_wick_ratio, close_location_value, open_location_value, directional_body, "
            "close_quality_label, morphology_definition_version, schema_version, provenance, created_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(candle_id) DO UPDATE SET range_abs=excluded.range_abs, "
            "body_ratio=excluded.body_ratio, close_quality_label=excluded.close_quality_label",
            (candle_id, m["range_abs"], m["body_ratio"], m["upper_wick_ratio"], m["lower_wick_ratio"],
             m["close_location_value"], m["open_location_value"], m["directional_body"],
             m["close_quality_label"], MORPHOLOGY_DEFINITION_VERSION, 5, provenance, now),
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
        print(f"computed morphology for {n} candles")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
