from __future__ import annotations

import sqlite3
import sys
import time
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.microstructure_collector import MicrostructureCollector


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/phase2_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_collector_maybe_checkpoint_records_timestamp() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, x TEXT)")
        conn.commit()
    finally:
        conn.close()

    collector = MicrostructureCollector(
        symbols=["ETHUSDT"],
        db_path=str(db),
        checkpoint_interval_sec=60.0,
    )
    collector.conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        collector._last_checkpoint_ts = time.time() - 61.0
        collector._maybe_checkpoint()
        assert collector._last_checkpoint_ts_utc is not None
        first_ts = float(collector._last_checkpoint_ts)
        collector._maybe_checkpoint()
        assert float(collector._last_checkpoint_ts) == first_ts
    finally:
        collector.conn.close()
        shutil.rmtree(wd, ignore_errors=True)
