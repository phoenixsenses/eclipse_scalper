from __future__ import annotations

import time
from pathlib import Path

try:
    from core.micro_features import MicroFeatureEngine
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.micro_features import MicroFeatureEngine

from tests.fixtures.microstructure import build_collector_schema_fixture, cleanup_temp_path, make_temp_micro_db


def _mk_db() -> Path:
    return make_temp_micro_db(prefix="test_micro_min")


def test_builder_with_minimal_three_tables() -> None:
    db = _mk_db()
    now_ms = int(time.time() * 1000)
    build_collector_schema_fixture(db, symbols=["BTCUSDT"], start_ms=now_ms - 39_000, rows_per_symbol=40)

    try:
        eng = MicroFeatureEngine(str(db), "BTCUSDT", lookback_sec=120, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        ready, reason, _detail = eng.get_readiness("BTCUSDT")
        assert ready is True
        assert reason == "ok"
    finally:
        cleanup_temp_path(db)

