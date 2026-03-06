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
        for i in range(20):
            rows.append((ts0 + i * 1000, "ETHUSDT", 100.0 + (0.1 * i), 1.0))
        for i in range(20):
            rows.append((ts0 + 20000 + i * 1000, "ETHUSDT", 102.0 - (0.1 * i), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_walkforward_writes_regime_segmented_stability() -> None:
    base = Path("eclipse_scalper/localtests/wf_regime_stability") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out = base / "out"
    _mk_db(db)
    run_walkforward(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out,
        slices=[("2024-03-01T00:00:00Z", "2024-03-01T00:00:15Z"), ("2024-03-01T00:00:20Z", "2024-03-01T00:00:35Z")],
        fee_bps=0.0,
        spread_bps=0.0,
        qty=1.0,
        horizon_sec=3,
        top_k=2,
        sort_by="pnl_net_sum",
        sort_desc=True,
    )
    idx = _read_csv(out / "index.csv")
    regimes = {r["regime"] for r in idx}
    assert regimes == {"up", "down"}
    assert (out / "stability.csv").exists()
    assert (out / "stability_all.csv").exists()
    assert (out / "stability_up.csv").exists()
    assert (out / "stability_down.csv").exists()
    all_row = _read_csv(out / "stability_all.csv")[0]
    up_row = _read_csv(out / "stability_up.csv")[0]
    down_row = _read_csv(out / "stability_down.csv")[0]
    assert float(all_row["combined_score"]) != 0.0 or float(all_row["stability_score"]) == 0.0
    assert up_row["regime"] == "up"
    assert down_row["regime"] == "down"

