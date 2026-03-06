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
            (1709251201000, "ETHUSDT", 100.4, 1.0),
            (1709251202000, "ETHUSDT", 100.8, 1.0),
            (1709251203000, "ETHUSDT", 101.2, 1.0),
            (1709251204000, "ETHUSDT", 101.6, 1.0),
            (1709251205000, "ETHUSDT", 102.0, 1.0),
        ]
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_eval_run_with_fills_metrics_consistent() -> None:
    base = Path("eclipse_scalper/localtests/eval_run_fills") / uuid.uuid4().hex
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
        fee_bps=1.0,
        spread_bps=10.0,
        qty=1.0,
        horizon_sec=2,
    )
    assert out["fills_count"] > 0
    fills = []
    for line in (run_dir / "fills.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            fills.append(json.loads(line))
    filled = [f for f in fills if str(f.get("status")) == "filled"]
    assert len(filled) > 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert int(metrics["fills_count"]) == len(filled)
    pnl_sum = sum(float(f.get("pnl") or 0.0) for f in filled)
    assert abs(float(metrics["pnl_sum"]) - pnl_sum) < 1e-9
    assert "fee_sum" in metrics
    assert "adverse_sum" in metrics
    assert "win_rate" in metrics
    assert (run_dir / "skipped.jsonl").exists()
    assert "skipped_count" in metrics
    assert "skipped_reasons" in metrics
    assert "decision_to_fill_rate" in metrics
    assert "horizon_price_source_counts" in metrics
    assert "avg_adverse_samples" in metrics
    assert "pnl_gross_sum" in metrics
    assert "pnl_net_sum" in metrics
    if filled:
        assert "fill_px_raw" in filled[0]
        assert "horizon_px_raw" in filled[0]
