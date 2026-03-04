from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path

from core.order_placement import OrderPlacementEngine


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"order_place_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _mk_orderbook_db(path: Path, *, symbol: str, bid_px: float, ask_px: float, bid_qty: float, ask_qty: float) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE orderbook (ts_ms INTEGER, symbol TEXT, bid_price REAL, bid_size REAL, ask_price REAL, ask_size REAL)"
        )
        conn.execute(
            "INSERT INTO orderbook(ts_ms,symbol,bid_price,bid_size,ask_price,ask_size) VALUES (?,?,?,?,?,?)",
            (1_700_000_000_000, symbol, bid_px, bid_qty, ask_px, ask_qty),
        )
        conn.commit()
    finally:
        conn.close()


def test_best_mode_uses_best_bid_ask() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        _mk_orderbook_db(db, symbol="ETHUSDT", bid_px=100.0, ask_px=100.1, bid_qty=1000.0, ask_qty=800.0)
        eng = OrderPlacementEngine(db_path=str(db), mode="best", queue_depth_threshold=500, default_tick_size=0.01)
        d_buy = eng.decide(symbol="ETHUSDT", side="long", ref_price=100.05, tick_size=0.01, fallback_limit=99.9)
        d_sell = eng.decide(symbol="ETHUSDT", side="short", ref_price=100.05, tick_size=0.01, fallback_limit=100.2)
        assert abs(d_buy.limit_price - 100.0) < 1e-9
        assert abs(d_sell.limit_price - 100.1) < 1e-9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_adaptive_mode_inside_spread_when_queue_deep() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        _mk_orderbook_db(db, symbol="ETHUSDT", bid_px=100.0, ask_px=100.2, bid_qty=2000.0, ask_qty=1200.0)
        eng = OrderPlacementEngine(db_path=str(db), mode="adaptive", queue_depth_threshold=500, default_tick_size=0.01)
        d = eng.decide(symbol="ETHUSDT", side="long", ref_price=100.1, tick_size=0.01, fallback_limit=99.9)
        assert d.mode == "inside_spread"
        assert d.limit_price > 100.0
        assert d.limit_price < 100.2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fallback_when_no_orderbook() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "missing.db"
        eng = OrderPlacementEngine(db_path=str(db), mode="best", queue_depth_threshold=500, default_tick_size=0.01)
        d = eng.decide(symbol="ETHUSDT", side="long", ref_price=100.0, tick_size=0.01, fallback_limit=99.5)
        assert abs(d.limit_price - 99.5) < 1e-9
        assert d.reason.startswith("fallback")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

