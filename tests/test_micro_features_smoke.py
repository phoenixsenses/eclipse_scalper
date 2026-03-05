from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_micro_features import build_micro_features


def _latest_symbol_ts(db: Path, symbol: str) -> tuple[float, float] | None:
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT MIN(ts_ms), MAX(ts_ms) FROM agg_trades WHERE symbol=?",
            (symbol,),
        ).fetchone()
        if not row or row[0] is None or row[1] is None:
            return None
        return float(row[0]) / 1000.0, float(row[1]) / 1000.0
    finally:
        conn.close()


@pytest.mark.skipif(not Path("data/microstructure.db").exists(), reason="local microstructure db missing")
def test_build_micro_features_smoke_real_db() -> None:
    db = Path("data/microstructure.db")
    rng = _latest_symbol_ts(db, "ETHUSDT")
    if rng is None:
        pytest.skip("ETHUSDT not found")
    _mn, mx = rng
    start = max(0.0, mx - 300.0)

    out_dir = Path("localtests") / "micro_features_smoke" / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_micro_features(
        db_path=db,
        out_root=out_dir,
        symbol="ETHUSDT",
        interval_ms=100,
        window_sec=300,
        start_ts=start,
        end_ts=mx,
        rv_window_sec=5.0,
    )
    assert manifest["dates"]
    total_rows = sum(int(d["rows"]) for d in manifest["dates"])
    assert total_rows > 0
