from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.data_layer_probe import (
    compute_row_count_deltas,
    detect_ts_col,
    is_heartbeat_fresh,
    normalize_ts_to_seconds,
    parse_heartbeat_json,
    serialize_heartbeat_json,
)


def test_detect_ts_col_prefers_candidate_order():
    cols = ["price", "event_time", "timestamp", "qty"]
    assert detect_ts_col(cols) == "timestamp"


def test_detect_ts_col_case_insensitive():
    cols = ["SYMBOL", "Created_At", "price"]
    assert detect_ts_col(cols) == "Created_At"


def test_detect_ts_col_ts_ms_supported():
    cols = ["symbol", "ts_ms", "price"]
    assert detect_ts_col(cols) == "ts_ms"


def test_detect_ts_col_none():
    assert detect_ts_col(["symbol", "price"]) is None


def test_normalize_ts_to_seconds():
    assert normalize_ts_to_seconds(1_700_000_000) == 1_700_000_000.0
    assert normalize_ts_to_seconds(1_700_000_000_000) == 1_700_000_000.0


def test_parse_heartbeat_json_and_freshness():
    raw = serialize_heartbeat_json({"ts_utc": "2023-11-14T22:13:20+00:00", "connected": True})
    hb = parse_heartbeat_json(raw)
    assert hb is not None
    assert is_heartbeat_fresh(hb, fresh_sec=120, now_ts=1_700_000_050.0)
    assert not is_heartbeat_fresh(hb, fresh_sec=120, now_ts=1_700_000_300.0)


def test_compute_row_count_deltas():
    start = {"agg_trades": 10, "mark_prices": 20}
    end = {"agg_trades": 25, "mark_prices": 20, "liquidations": 3}
    out = compute_row_count_deltas(start, end)
    assert out["agg_trades"] == 15
    assert out["mark_prices"] == 0
    assert out["liquidations"] == 3
