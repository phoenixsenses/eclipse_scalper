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
        rows = [
            (1709251200000, "ETHUSDT", 100.0, 1.0, 2000.0, 0.3, 0.0002),
            (1709251201000, "ETHUSDT", 100.2, 1.0, 2600.0, 0.5, 0.0002),
            (1709251202000, "ETHUSDT", 100.4, 1.0, 2700.0, 0.6, 0.0002),
            (1709251203000, "ETHUSDT", 100.6, 1.0, 2800.0, 0.7, 0.0002),
            (1709251204000, "ETHUSDT", 100.8, 1.0, 2900.0, 0.8, 0.0002),
        ]
        conn.executemany(
            "INSERT INTO agg_trades (ts_ms, symbol, price, quantity, trade_intensity, imbalance, spread) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_eval_run_micro_edge_strategy_outputs() -> None:
    base = Path("eclipse_scalper/localtests/eval_run_micro_edge") / uuid.uuid4().hex
    db = base / "db.sqlite"
    run_dir = base / "run"
    _mk_db(db)
    strategy_cfg = {
        "rule": "micro_edge_v3_passive_alpha",
        "side": "buy",
        "symbol_whitelist": ["ETHUSDT"],
        "event_source_table": "agg_trades",
        "cooldown_ms": 0,
        "filters": {"imbalance_gte": 0.4, "intensity_gte": 2500, "spread_lte": 0.0003},
    }
    out = run_eval(
        db=db,
        symbols=["ETHUSDT"],
        start="2024-03-01T00:00:00Z",
        end="2024-03-01T00:01:00Z",
        strategy="micro_edge_pocket",
        strategy_config=strategy_cfg,
        run_dir=run_dir,
        fee_bps=0.0,
        spread_bps=0.0,
        qty=1.0,
        horizon_sec=1,
    )
    assert out["decisions_count"] > 0
    decisions = [json.loads(x) for x in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert decisions
    params = decisions[0].get("params", {})
    assert "pocket_id" in params
    assert params.get("rule") == "micro_edge_v3_passive_alpha"
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["strategy"] == "micro_edge_pocket"
    assert cfg["strategy_config"]["rule"] == "micro_edge_v3_passive_alpha"

