from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

try:
    from core.micro_features import MicroFeatureEngine
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.micro_features import MicroFeatureEngine


def _mk_db_path(tag: str) -> Path:
    base = Path("data")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"test_micro_{tag}_{uuid.uuid4().hex}.db"


def _build_db(path: Path, *, symbol: str, start_ms: int, n: int = 90) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (symbol TEXT, ts_ms INTEGER, price REAL, quantity REAL, is_buyer_maker INTEGER)"
        )
        conn.execute("CREATE TABLE mark_prices (symbol TEXT, ts_ms INTEGER, mark_price REAL)")
        conn.execute("CREATE TABLE liquidations (symbol TEXT, ts_ms INTEGER, side TEXT, quantity REAL)")
        for i in range(n):
            ts = start_ms + (i * 1000)
            mark = 100.0 + (0.01 * i)
            price = mark + (0.01 if i % 2 == 0 else -0.01)
            qty = 1.0 + (i % 3) * 0.1
            is_bm = 0 if i % 3 else 1
            conn.execute(
                "INSERT INTO agg_trades(symbol, ts_ms, price, quantity, is_buyer_maker) VALUES(?,?,?,?,?)",
                (symbol, ts, price, qty, is_bm),
            )
            conn.execute(
                "INSERT INTO mark_prices(symbol, ts_ms, mark_price) VALUES(?,?,?)",
                (symbol, ts, mark),
            )
            if i % 10 == 0:
                conn.execute(
                    "INSERT INTO liquidations(symbol, ts_ms, side, quantity) VALUES(?,?,?,?)",
                    (symbol, ts, "sell", 3.0),
                )
        conn.commit()
    finally:
        conn.close()


def test_micro_features_ready_and_nonempty() -> None:
    db = _mk_db_path("ready")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 70_000
    try:
        _build_db(db, symbol="BTCUSDT", start_ms=start_ms, n=70)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=90, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        ready, reason, _detail = eng.get_readiness("BTCUSDT")
        assert ready is True
        assert reason == "ok"
        assert feat.mark_price > 0
        assert feat.trade_intensity > 0
    finally:
        db.unlink(missing_ok=True)


def test_micro_features_min_samples_reason() -> None:
    db = _mk_db_path("minsamples")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 10_000
    try:
        _build_db(db, symbol="BTCUSDT", start_ms=start_ms, n=2)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=30, update_interval_sec=1.0)
        feat = eng._compute_once()
        # features may still be present, readiness should flag min_samples
        assert feat is not None
        ready, reason, detail = eng.get_readiness("BTCUSDT")
        assert ready is False
        assert reason == "min_samples"
        assert "trades_30s" in detail
    finally:
        db.unlink(missing_ok=True)


def test_trade_intensity_uses_per_minute_units() -> None:
    db = _mk_db_path("intensity_units")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 70_000
    try:
        _build_db(db, symbol="BTCUSDT", start_ms=start_ms, n=70)
        eng = MicroFeatureEngine(str(db), ["BTCUSDT"], lookback_sec=90, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        diag = eng.get_diag("BTCUSDT")
        counts = dict(diag.get("sample_counts") or {})
        trades_30s = int(counts.get("trades_30s", 0) or 0)
        # 30s sample converted to per-minute equivalent.
        assert feat.trade_intensity == float(trades_30s) * 2.0
    finally:
        db.unlink(missing_ok=True)
