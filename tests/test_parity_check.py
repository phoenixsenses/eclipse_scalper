from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

try:
    from tools.parity_check import run_parity_check
    from tools.replay_strategy import replay_to_decisions, write_decisions_jsonl
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.parity_check import run_parity_check
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
            (1709251201000, "ETHUSDT", 100.2, 1.0),
            (1709251202000, "ETHUSDT", 100.4, 1.0),
            (1709251203000, "ETHUSDT", 100.6, 1.0),
            (1709251204000, "ETHUSDT", 100.8, 1.0),
        ]
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_parity_check_match_and_mismatch() -> None:
    base = Path("eclipse_scalper/localtests/parity_check") / uuid.uuid4().hex
    db = base / "db.sqlite"
    paper_path = base / "paper.jsonl"
    _mk_db(db)
    decisions, _ = replay_to_decisions(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        strategy_name="baseline",
        strategy_config={"period": 2},
    )
    write_decisions_jsonl(paper_path, decisions)
    report_ok = run_parity_check(
        paper_decisions=paper_path,
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="baseline",
        strategy_config={"period": 2},
    )
    assert report_ok["match"] is True
    assert report_ok["first_divergence_index"] is None

    rows = [json.loads(x) for x in paper_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows
    rows[0]["decision_id"] = "BROKEN_ID"
    with paper_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")))
            f.write("\n")
    report_bad = run_parity_check(
        paper_decisions=paper_path,
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="baseline",
        strategy_config={"period": 2},
    )
    assert report_bad["match"] is False
    assert report_bad["first_divergence_index"] == 0
    assert report_bad["paper_hash"] != report_bad["replay_hash"]

