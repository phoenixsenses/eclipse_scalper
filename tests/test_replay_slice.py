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
    """--health-root is pinned to an isolated temp directory here (Part C
    isolation principle) -- discovered 2026-07-10: an earlier version of
    this test had no such override and left a stray replay.json in the
    real repo's logs/health/ every time it ran, picked up by the live
    canonical overall.json as a phantom "replay" component."""
    real_replay_health = Path("logs/health/replay.json")
    existed_before = real_replay_health.exists()
    mtime_before = real_replay_health.stat().st_mtime if existed_before else None

    base = Path("eclipse_scalper/localtests/replay_slice") / uuid.uuid4().hex
    db = base / "db" / "sample.db"
    health_root = base / "health"
    _mk_db(db)
    rc = run_replay(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        speed=1000.0,
        progress_every=1,
        health_root=str(health_root),
    )
    assert rc == 0
    assert (health_root / "replay.json").exists()

    assert real_replay_health.exists() == existed_before
    if existed_before:
        assert real_replay_health.stat().st_mtime == mtime_before

