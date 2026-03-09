from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_event_lanes import check_gate


def test_check_gate_uses_latest_db_window_for_historical_fixture() -> None:
    result = check_gate(
        db="tests/fixtures/microstructure_sample.db",
        symbol="ETHUSDT",
        lookback_min=60,
        bucket_sec=5,
        stale_after_sec=60,
    )
    assert result["symbol"] == "ETHUSDT"
    assert result["gate"] in {"ALLOWED", "BLOCKED"}
    assert result["reason"] != "no_data"
    assert result["buckets_loaded"] > 0
    assert "book_proxy_pressure" in result["lanes"]
    assert "volatility_burst" in result["lanes"]
