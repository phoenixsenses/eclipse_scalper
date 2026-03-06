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
        for i in range(80):
            rows.append((ts0 + i * 1000, "ETHUSDT", 100.0 + (0.02 * i), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_grid_strategy_overrides_expand_combo_space() -> None:
    base = Path("eclipse_scalper/localtests/wf_grid_strategy") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out = base / "out"
    _mk_db(db)
    rows = run_walkforward_sweep(
        db=db,
        symbols=["ETHUSDT"],
        strategy="micro_edge_pocket",
        strategy_config={"side": "buy", "filters": {"imbalance_gte": 0.3}},
        out_dir=out,
        slices=[("2024-03-01T00:00:00Z", "2024-03-01T00:00:20Z"), ("2024-03-01T00:00:20Z", "2024-03-01T00:00:40Z")],
        grid=[("fee_bps", ["0.0"]), ("spread_bps", ["0.0"]), ("horizon_sec", ["5"])],
        grid_strategy=[("filters.imbalance_gte", ["0.30", "0.40"]), ("cooldown_ms", ["0", "250"])],
        top_n=10,
        sort_by="combined_score",
        sort_desc=True,
    )
    assert rows["count"] == 4
    idx = _read_csv(out / "index.csv")
    assert len(idx) == 4
    assert "strategy_overrides_json" in idx[0]
    assert "combined_score" in idx[0]
    assert "imbalance_gte" in idx[0]
    assert "cooldown_ms" in idx[0]

