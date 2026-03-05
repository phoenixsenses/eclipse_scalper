from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

try:
    from tools.replay_slice import run_replay
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.replay_slice import run_replay


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL)"
        )
        conn.execute("INSERT INTO agg_trades (ts_ms, symbol, price) VALUES (1709251200000, 'ETHUSDT', 1.0)")
        conn.execute("INSERT INTO agg_trades (ts_ms, symbol, price) VALUES (1709251202000, 'ETHUSDT', 1.1)")
        conn.execute("INSERT INTO agg_trades (ts_ms, symbol, price) VALUES (1709251201000, 'ETHUSDT', 1.2)")
        conn.commit()
    finally:
        conn.close()


def test_replay_slice_runs_deterministic() -> None:
    db = Path("eclipse_scalper/localtests/replay_slice") / f"{uuid.uuid4().hex}.db"
    _mk_db(db)
    rc = run_replay(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        speed=1000.0,
        progress_every=1,
    )
    assert rc == 0

