from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path


try:
    from tools.eval_run import run_eval
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.eval_run import run_eval


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


def test_eval_run_writes_required_artifacts() -> None:
    base = Path("eclipse_scalper/localtests/eval_run") / uuid.uuid4().hex
    db = base / "db.sqlite"
    run_dir = base / "run"
    _mk_db(db)

    out = run_eval(
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="baseline",
        strategy_config={"period": 2},
        run_dir=run_dir,
    )
    assert out["events_replayed"] > 0
    assert out["decisions_count"] > 0

    required = [
        "config.json",
        "health.json",
        "decisions.jsonl",
        "state_vector.jsonl",
        "fills.jsonl",
        "metrics.json",
        "summary.md",
    ]
    for name in required:
        assert (run_dir / name).exists(), name

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["strategy"] == "baseline"
    assert cfg["start"] == "2024-03-01T00:00:00Z"
    assert cfg["end"] == "2024-03-01T00:01:00Z"

    decisions_lines = (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(decisions_lines) > 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions_count"] == len(decisions_lines)
    assert int(metrics["state_vectors_count"]) > 0
