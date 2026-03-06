from __future__ import annotations

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
            rows.append((ts0 + i * 1000, "ETHUSDT", 100.0 + (0.03 * i), 1.0))
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_walkforward_sweep_promotes_top1_to_latest() -> None:
    base = Path("eclipse_scalper/localtests/wf_sweep_promote") / uuid.uuid4().hex
    db = base / "db.sqlite"
    out_dir = base / "out"
    latest_dir = base / "latest"
    _mk_db(db)

    out = run_walkforward_sweep(
        db=db,
        symbols=["ETHUSDT"],
        strategy="baseline",
        strategy_config={"period": 2},
        out_dir=out_dir,
        slices=[("2024-03-01T00:00:00Z", "2024-03-01T00:00:20Z"), ("2024-03-01T00:00:20Z", "2024-03-01T00:00:40Z")],
        grid=[("fee_bps", ["0", "0.6"]), ("spread_bps", ["0"]), ("horizon_sec", ["5"])],
        grid_strategy=[],
        top_n=10,
        sort_by="combined_score",
        sort_desc=True,
        promote_top=1,
        latest_dir=latest_dir,
        latest_candidates_dir=base / "latest_candidates",
        promote_include_glob=["stability*.csv"],
        promote_extra="stability_all.csv",
        promote_strict_extra=False,
        promote_print_env=False,
        promote_enable_alpha_gate=True,
    )

    assert int(out["count"]) == 2
    assert len(out.get("promoted") or []) == 1
    assert (latest_dir / "metrics.json").exists()
    assert (latest_dir / "config.json").exists()
    assert (latest_dir / "stability.csv").exists()
    assert (latest_dir / "stability_up.csv").exists()
    assert (latest_dir / "stability_down.csv").exists()
    assert (latest_dir / "run_dir.txt").exists()
    run_dir_txt = (latest_dir / "run_dir.txt").read_text(encoding="utf-8").strip()
    assert "walkforward" in run_dir_txt

