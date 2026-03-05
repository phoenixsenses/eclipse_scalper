from __future__ import annotations

import csv
import sqlite3
import uuid
from pathlib import Path

try:
    from tools.sweep_eval import _parse_grid, run_sweep
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.sweep_eval import _parse_grid, run_sweep


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL, quantity REAL)"
        )
        rows = []
        ts = 1709251200000
        px = 100.0
        for i in range(40):
            rows.append((ts + i * 1000, "ETHUSDT", px + (i * 0.1), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_sweep_eval_index_and_summary_deterministic() -> None:
    base = Path("eclipse_scalper/localtests/sweep_eval") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out1 = base / "out1"
    out2 = base / "out2"
    _mk_db(db)
    grid = _parse_grid("fee_bps=0,0.6;spread_bps=0,10;horizon_sec=5,10")
    result1 = run_sweep(
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out1,
        grid=grid,
        base_qty=0.1,
        top_n=3,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    assert result1["count"] == 8
    assert (out1 / "index.csv").exists()
    assert (out1 / "summary.md").exists()
    rows1 = _read_csv(out1 / "index.csv")
    assert len(rows1) == 8
    assert "pnl_net_sum" in rows1[0]
    assert "run_dir" in rows1[0]
    summary = (out1 / "summary.md").read_text(encoding="utf-8")
    assert "## Top N" in summary
    assert "| rank | run_dir |" in summary

    result2 = run_sweep(
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out2,
        grid=grid,
        base_qty=0.1,
        top_n=3,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    assert result2["count"] == 8
    csv1 = (out1 / "index.csv").read_text(encoding="utf-8")
    csv2 = (out2 / "index.csv").read_text(encoding="utf-8")
    assert csv1 == csv2

