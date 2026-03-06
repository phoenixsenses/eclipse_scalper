from __future__ import annotations

import csv
import sqlite3
import uuid
from pathlib import Path

try:
    from tools.walkforward_eval import run_walkforward
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.walkforward_eval import run_walkforward


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL, quantity REAL)"
        )
        rows = []
        ts0 = 1709251200000
        for i in range(30):
            rows.append((ts0 + i * 1000, "ETHUSDT", 100.0 + (0.1 * i), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_walkforward_eval_outputs_and_determinism() -> None:
    base = Path("eclipse_scalper/localtests/walkforward") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out1 = base / "out1"
    out2 = base / "out2"
    _mk_db(db)
    slices = [
        ("2024-03-01T00:00:00Z", "2024-03-01T00:00:10Z"),
        ("2024-03-01T00:00:10Z", "2024-03-01T00:00:20Z"),
    ]
    run_walkforward(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out1,
        slices=slices,
        fee_bps=0.0,
        spread_bps=0.0,
        qty=1.0,
        horizon_sec=3,
        top_k=2,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    assert (out1 / "index.csv").exists()
    assert (out1 / "stability.csv").exists()
    assert (out1 / "summary.md").exists()
    idx1 = _read_csv(out1 / "index.csv")
    assert len(idx1) == 2
    stab1 = _read_csv(out1 / "stability.csv")
    assert len(stab1) == 1
    assert "slices_count" in stab1[0]
    assert int(float(stab1[0]["slices_count"])) == 2

    run_walkforward(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out2,
        slices=slices,
        fee_bps=0.0,
        spread_bps=0.0,
        qty=1.0,
        horizon_sec=3,
        top_k=2,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    assert (out1 / "index.csv").read_text(encoding="utf-8") == (out2 / "index.csv").read_text(encoding="utf-8")

