from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


try:
    from tools.replay_strategy import replay_to_decisions, write_decisions_jsonl
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.replay_strategy import replay_to_decisions, write_decisions_jsonl


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL, quantity REAL)"
        )
        rows = [
            (1709251200000, "ETHUSDT", 100.0, 1.0),
            (1709251201000, "ETHUSDT", 100.1, 1.2),
            (1709251202000, "ETHUSDT", 100.2, 0.8),
            (1709251203000, "ETHUSDT", 100.3, 0.7),
            (1709251204000, "ETHUSDT", 100.4, 0.9),
            (1709251205000, "ETHUSDT", 100.5, 0.5),
        ]
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_replay_strategy_deterministic_output_bytes() -> None:
    base = Path("eclipse_scalper/localtests/replay_strategy") / uuid.uuid4().hex
    db = base / "db.sqlite"
    _mk_db(db)
    out1 = base / "decisions_1.jsonl"
    out2 = base / "decisions_2.jsonl"

    d1, events1 = replay_to_decisions(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        strategy_name="baseline",
        strategy_config={"period": 2},
    )
    d2, events2 = replay_to_decisions(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        strategy_name="baseline",
        strategy_config={"period": 2},
    )
    assert events1 == events2
    assert d1 == d2
    assert len(d1) > 0
    assert all("decision_id" in x for x in d1)

    write_decisions_jsonl(out1, d1)
    write_decisions_jsonl(out2, d2)
    assert out1.read_bytes() == out2.read_bytes()

