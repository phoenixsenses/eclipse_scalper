from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from tools import feature_distribution_analysis as fda


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"feature_dist_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_db(path: Path, symbol: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT, quantity REAL, is_buyer_maker INTEGER)"
        )
        now_ms = int(time.time() * 1000)
        rows = []
        for i in range(600):
            ts_ms = now_ms - 60_000 + i * 100
            rows.append((ts_ms, symbol, 1.0 + (i % 5) * 0.1, 0 if i % 2 == 0 else 1))
        conn.executemany(
            "INSERT INTO agg_trades(ts_ms,symbol,quantity,is_buyer_maker) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_feature_distribution_writes_markdown_and_plots(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        out_md = tmp / "report.md"
        plots = tmp / "plots"
        _build_db(db, "ETHUSDT")

        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--symbol",
                "ETHUSDT",
                "--lookback-hours",
                "2",
                "--out",
                str(out_md),
                "--plots-dir",
                str(plots),
            ],
        )
        rc = fda.main()
        assert rc == 0
        assert out_md.exists()
        txt = out_md.read_text(encoding="utf-8")
        assert "Feature Distribution Analysis" in txt
        assert "imbalance_hist" in txt
        assert "intensity_hist" in txt
        assert (plots / "feature_dist_ETHUSDT_imbalance.png").exists()
        assert (plots / "feature_dist_ETHUSDT_trade_intensity.png").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

