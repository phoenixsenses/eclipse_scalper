from __future__ import annotations

import json
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
            """
            CREATE TABLE agg_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                symbol TEXT,
                price REAL,
                quantity REAL,
                trade_intensity REAL,
                imbalance REAL,
                spread REAL
            )
            """
        )
        rows = []
        for i in range(20):
            rows.append((1709251200000 + i * 1000, "ETHUSDT", 100.0 + i * 0.1, 1.0, 2600.0, 0.6, 0.0002))
        conn.executemany(
            "INSERT INTO agg_trades (ts_ms, symbol, price, quantity, trade_intensity, imbalance, spread) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_sweep_eval_runs_micro_edge_strategy() -> None:
    base = Path("eclipse_scalper/localtests/sweep_eval_micro_edge") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out = base / "out"
    _mk_db(db)
    cfg = {
        "rule": "micro_edge_v3_passive_alpha",
        "side": "buy",
        "symbol_whitelist": ["ETHUSDT"],
        "event_source_table": "agg_trades",
        "cooldown_ms": 0,
        "filters": {"imbalance_gte": 0.4, "intensity_gte": 2500, "spread_lte": 0.0003},
    }
    result = run_sweep(
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="micro_edge_pocket",
        strategy_config=cfg,
        out_dir=out,
        grid=_parse_grid("fee_bps=0;spread_bps=0;horizon_sec=5"),
        base_qty=1.0,
        top_n=3,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    assert result["count"] == 1
    idx = (out / "index.json").read_text(encoding="utf-8")
    data = json.loads(idx)
    assert data["rows"][0]["strategy"] == "micro_edge_pocket"

