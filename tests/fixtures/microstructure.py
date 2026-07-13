from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


def make_temp_micro_db(prefix: str = "micro_fixture") -> Path:
    root = Path("localtests") / "microstructure"
    root.mkdir(parents=True, exist_ok=True)
    return (root / f"{prefix}_{uuid.uuid4().hex[:8]}.db").resolve()


def cleanup_temp_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
    try:
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        pass


def build_collector_schema_fixture(
    path: Path,
    *,
    symbols: Iterable[str] = ("BTCUSDT",),
    start_ms: int | None = None,
    rows_per_symbol: int = 60,
    include_liquidations: bool = True,
    include_true_book: bool = False,
) -> None:
    base_ms = int(start_ms if start_ms is not None else (time.time() * 1000) - 120_000)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER)"
        )
        conn.execute(
            "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)"
        )
        conn.execute(
            "CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, side TEXT, price REAL, quantity REAL, notional REAL, trade_time_ms INTEGER)"
        )
        if include_true_book:
            conn.execute(
                "CREATE TABLE top_of_book (ts_ms INTEGER, symbol TEXT, bid_px REAL, ask_px REAL, bid_qty REAL, ask_qty REAL)"
            )

        for sym_index, symbol in enumerate([str(s).upper() for s in symbols]):
            sym_offset = sym_index * 10_000
            base_price = 100.0 * float(sym_index + 1)
            for i in range(int(rows_per_symbol)):
                ts_ms = base_ms + sym_offset + (i * 1000)
                mark = base_price + (0.01 * i)
                price = mark + (0.01 if i % 2 == 0 else -0.01)
                qty = 1.0 + ((i + sym_index) % 3) * 0.1
                notional = price * qty
                is_buyer_maker = 0 if i % 3 else 1
                conn.execute(
                    "INSERT INTO agg_trades (ts_ms, symbol, price, quantity, notional, is_buyer_maker) VALUES (?, ?, ?, ?, ?, ?)",
                    (ts_ms, symbol, price, qty, notional, is_buyer_maker),
                )
                conn.execute(
                    "INSERT INTO mark_prices (ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms) VALUES (?, ?, ?, ?, ?)",
                    (ts_ms, symbol, mark, 0.0, ts_ms + 28_800_000),
                )
                if include_liquidations and (i % 10 == 0):
                    liq_side = "SELL" if i % 20 == 0 else "BUY"
                    liq_qty = 2.0 + (0.5 * (i % 3))
                    liq_notional = mark * liq_qty
                    conn.execute(
                        "INSERT INTO liquidations VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ts_ms + 500, symbol, liq_side, mark, liq_qty, liq_notional, ts_ms + 500),
                    )
                if include_true_book:
                    conn.execute(
                        "INSERT INTO top_of_book VALUES (?, ?, ?, ?, ?, ?)",
                        (ts_ms, symbol, mark - 0.05, mark + 0.05, 5.0 + i % 4, 4.0 + i % 5),
                    )
        conn.commit()
    finally:
        conn.close()


def build_fixture_manifest(path: Path, symbols: Iterable[str]) -> dict[str, Any]:
    conn = sqlite3.connect(str(path))
    try:
        ranges: dict[str, dict[str, float | int | None]] = {}
        for symbol in [str(s).upper() for s in symbols]:
            row = conn.execute(
                "SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM agg_trades WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if not row or row[0] is None or row[1] is None:
                ranges[symbol] = {"min_ts_ms": None, "max_ts_ms": None, "rows": 0}
                continue
            ranges[symbol] = {
                "min_ts_ms": int(row[0]),
                "max_ts_ms": int(row[1]),
                "rows": int(row[2] or 0),
            }
        return {
            "db": str(path),
            "symbols": ranges,
        }
    finally:
        conn.close()


def load_micro_edge_rows(
    path: Path,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    bucket_sec: int = 5,
    vol_window: int = 6,
) -> list[dict[str, Any]]:
    from tools.micro_edge_smoke import _load_symbol_trades_marks_and_liqs
    from tools.micro_edge_lib import build_bucket_features

    conn = sqlite3.connect(str(path))
    try:
        # root points at the fixture db's own (non-archive) directory so
        # plan_read() falls through to a direct sqlite read; source_db_path
        # is pinned explicitly so execute_read() never falls back to the
        # real data/microstructure.db.
        trades, marks, liqs = _load_symbol_trades_marks_and_liqs(
            conn,
            str(symbol).upper(),
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            root=str(Path(path).resolve().parent),
            source_db_path=str(path),
        )
    finally:
        conn.close()
    return build_bucket_features(
        trades=trades,
        marks=marks,
        liqs=liqs,
        bucket_sec=int(bucket_sec),
        vol_window=int(vol_window),
    )
