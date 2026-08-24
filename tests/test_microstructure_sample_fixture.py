from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.features.micro_features import compute_features, load_symbol_window
from src.microphys.io.sqlite_reader import SQLiteMicroReader
from tools import validate_microstructure_contract as vmc


FIXTURE_DB = Path("tests/fixtures/microstructure_sample.db")
FIXTURE_MANIFEST = Path("tests/fixtures/microstructure_sample_manifest.json")


def test_sample_fixture_manifest_matches_db() -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["db"] == "tests/fixtures/microstructure_sample.db"
    assert manifest["schema"] == "collector_baseline_v1"
    assert manifest["true_top_of_book"] is False

    conn = sqlite3.connect(str(FIXTURE_DB))
    try:
        for table, info in manifest["tables"].items():
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert int(row_count) == int(info["rows"])
    finally:
        conn.close()


def test_sample_fixture_contract_warns_without_true_book() -> None:
    payload = vmc.analyze_contract(FIXTURE_DB, ["BTCUSDT", "ETHUSDT"])
    assert payload["status"] == "warn"
    assert payload["feature_capability"]["tier"] == "trade_plus_liq_mark_proxy"
    assert payload["feature_capability"]["requires_book"] is False
    assert "true_top_of_book_missing" in payload["warnings"]


def test_sample_fixture_reader_and_feature_compute() -> None:
    reader = SQLiteMicroReader(FIXTURE_DB)
    btc_range = reader.get_ts_range("trades", "btcusdt")
    assert btc_range[0] is not None
    assert btc_range[1] is not None

    trades = reader.read_trades("BTCUSDT", 1_700_000_000.0, 1_700_000_050.0)
    marks = reader.read_top_of_book("BTCUSDT", 1_700_000_000.0, 1_700_000_050.0)
    liqs = reader.read_liquidations("BTCUSDT", 1_700_000_000.0, 1_700_000_050.0)

    assert trades
    assert marks
    assert liqs
    assert all(item.symbol == "BTCUSDT" for item in trades)

    conn = sqlite3.connect(str(FIXTURE_DB))
    try:
        records = load_symbol_window(conn, "ETHUSDT")
    finally:
        conn.close()

    features = compute_features(records, volatility_window=5)
    assert features
    assert any(row["mid"] is not None for row in features)
    assert all(row["spread"] is None for row in features)
    assert any((row["trade_intensity"] or 0) > 0 for row in features)
