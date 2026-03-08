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


def _mk_db_path(tag: str) -> Path:
    return make_temp_micro_db(prefix=f"test_micro_{tag}")


def test_micro_features_ready_and_nonempty() -> None:
    db = _mk_db_path("ready")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 70_000
    try:
        build_collector_schema_fixture(db, symbols=["BTCUSDT"], start_ms=start_ms, rows_per_symbol=70)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=90, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        ready, reason, _detail = eng.get_readiness("BTCUSDT")
        assert ready is True
        assert reason == "ok"
        assert feat.mark_price > 0
        assert feat.trade_intensity > 0
    finally:
        cleanup_temp_path(db)


def test_micro_features_min_samples_reason() -> None:
    db = _mk_db_path("minsamples")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 10_000
    try:
        build_collector_schema_fixture(db, symbols=["BTCUSDT"], start_ms=start_ms, rows_per_symbol=2)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=30, update_interval_sec=1.0)
        feat = eng._compute_once()
        # features may still be present, readiness should flag min_samples
        assert feat is not None
        ready, reason, detail = eng.get_readiness("BTCUSDT")
        assert ready is False
        assert reason == "min_samples"
        assert "trades_30s" in detail
    finally:
        cleanup_temp_path(db)


def test_trade_intensity_uses_per_minute_units() -> None:
    db = _mk_db_path("intensity_units")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 70_000
    try:
        build_collector_schema_fixture(db, symbols=["BTCUSDT"], start_ms=start_ms, rows_per_symbol=70)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=90, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        diag = eng.get_diag("BTCUSDT")
        counts = dict(diag.get("sample_counts") or {})
        trades_30s = int(counts.get("trades_30s", 0) or 0)
        # 30s sample converted to per-minute equivalent.
        assert feat.trade_intensity == float(trades_30s) * 2.0
    finally:
        cleanup_temp_path(db)
