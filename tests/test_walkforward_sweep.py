from __future__ import annotations

import csv
import sqlite3
import uuid
from pathlib import Path

try:
    from tools.walkforward_sweep import run_walkforward_sweep
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.walkforward_sweep import run_walkforward_sweep


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL, quantity REAL)"
        )
        rows = []
        ts0 = 1709251200000
        for i in range(60):
            rows.append((ts0 + i * 1000, "ETHUSDT", 100.0 + (0.05 * i), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_walkforward_sweep_outputs_and_determinism() -> None:
    base = Path("eclipse_scalper/localtests/walkforward_sweep") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out1 = base / "out1"
    out2 = base / "out2"
    _mk_db(db)
    slices = [
        ("2024-03-01T00:00:00Z", "2024-03-01T00:00:15Z"),
        ("2024-03-01T00:00:15Z", "2024-03-01T00:00:30Z"),
    ]
    grid = [
        ("fee_bps", ["0", "0.6"]),
        ("spread_bps", ["0"]),
        ("horizon_sec", ["5", "10"]),
    ]

    run_walkforward_sweep(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out1,
        slices=slices,
        grid=grid,
        grid_strategy=[],
        top_n=10,
        sort_by="stability_score",
        sort_desc=True,
    )
    index_rows = _read_csv(out1 / "index.csv")
    assert len(index_rows) == 4
    assert (out1 / "summary.md").exists()
    for row in index_rows:
        combo_id = row["combo_id"]
        assert (out1 / "combos" / combo_id / "walkforward" / "stability.csv").exists()

    run_walkforward_sweep(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out2,
        slices=slices,
        grid=grid,
        grid_strategy=[],
        top_n=10,
        sort_by="stability_score",
        sort_desc=True,
    )
    assert (out1 / "index.csv").read_text(encoding="utf-8") == (out2 / "index.csv").read_text(encoding="utf-8")
