from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.features.micro_features import compute_features, load_symbol_window
from tests.fixtures.microstructure import build_collector_schema_fixture, cleanup_temp_path, make_temp_micro_db


def test_compute_features_synthetic():
    records = [
        {"ts_ms": 1000, "symbol": "BTCUSDT", "best_bid": 99.0, "best_ask": 101.0, "bid_vol": 60.0, "ask_vol": 40.0, "trade_count": 2},
        {"ts_ms": 2000, "symbol": "BTCUSDT", "best_bid": 100.0, "best_ask": 102.0, "bid_vol": 70.0, "ask_vol": 30.0, "trade_count": 3},
        {"ts_ms": 3000, "symbol": "BTCUSDT", "best_bid": 101.0, "best_ask": 103.0, "bid_vol": 80.0, "ask_vol": 20.0, "trade_count": 1},
    ]
    out = compute_features(records, volatility_window=3)
    assert len(out) == 3
    assert out[0]["mid"] == 100.0
    assert out[0]["spread"] == 2.0
    assert out[0]["imbalance"] == 0.6
    assert out[1]["trade_intensity"] == 5
    assert out[1]["ret_1"] is not None
    assert out[2]["micro_volatility"] is not None


def test_compute_features_from_deterministic_db_fixture() -> None:
    db = make_temp_micro_db(prefix="test_micro_features_compute")
    try:
        build_collector_schema_fixture(
            db,
            symbols=["BTCUSDT"],
            start_ms=1_700_000_000_000,
            rows_per_symbol=12,
            include_true_book=True,
        )
        conn = sqlite3.connect(str(db))
        try:
            records = load_symbol_window(conn, "BTCUSDT")
        finally:
            conn.close()
        out = compute_features(records, volatility_window=5)
        assert out
        assert any(row["mid"] is not None for row in out)
        assert any(row["spread"] is not None for row in out)
        assert any(row["trade_intensity"] > 0 for row in out)
    finally:
        cleanup_temp_path(db)
