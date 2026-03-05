from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from core.chart_generator import build_equity_curve_png


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/track3_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE trades (exit_time REAL, pnl_bps REAL)")
        now = time.time()
        rows = [(now - 3600 + i * 60, (1 if i % 2 == 0 else -0.5)) for i in range(100)]
        conn.executemany("INSERT INTO trades(exit_time,pnl_bps) VALUES(?,?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_build_equity_curve_png() -> None:
    wd = _workdir()
    db = wd / "paper_trades.db"
    png = wd / "chart.png"
    try:
        _mk_db(db)
        ok = build_equity_curve_png(db, png, days=7)
        assert ok is True
        assert png.exists()
        assert png.stat().st_size > 0
    finally:
        shutil.rmtree(wd, ignore_errors=True)

