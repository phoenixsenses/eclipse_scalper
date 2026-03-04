from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from tools import analyze_fill_timing as aft


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"fill_timing_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_analyze_fill_timing_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        live = tmp / "papertrades_live.parquet"
        trade_db = tmp / "paper_trades.db"
        out_md = tmp / "fill_timing.md"
        out_json = tmp / "fill_timing.json"

        pd.DataFrame(
            [
                {"entry_ts_utc": "2026-03-04T10:00:00Z", "fill_delay_bars": 2, "filled": True, "ttl_expired": False, "pnl_net": 0.01},
                {"entry_ts_utc": "2026-03-04T10:00:01Z", "fill_delay_bars": 12, "filled": False, "ttl_expired": True, "pnl_net": -0.02},
            ]
        ).to_parquet(live, index=False)

        conn = sqlite3.connect(str(trade_db))
        try:
            conn.execute(
                "CREATE TABLE trades (entry_time REAL, exit_time REAL, side TEXT, pnl_bps REAL, max_adverse_bps REAL, elapsed_sec REAL, exit_reason TEXT)"
            )
            conn.execute(
                "INSERT INTO trades(entry_time,exit_time,side,pnl_bps,max_adverse_bps,elapsed_sec,exit_reason) VALUES (?,?,?,?,?,?,?)",
                (1.0, 2.0, "sell", 1.0, 2.5, 8.0, "horizon"),
            )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--live-parquet",
                str(live),
                "--trade-db",
                str(trade_db),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ],
        )
        assert aft.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "ok"
        assert int(payload["live_summary"]["rows"]) == 2
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

