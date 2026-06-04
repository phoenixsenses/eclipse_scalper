from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tools import shadow_lane_signal_emitter as emitter


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL,
            funding_rate REAL,
            next_funding_time_ms INTEGER
        );
        CREATE TABLE detector_signals (
            id INTEGER PRIMARY KEY,
            signal_id TEXT,
            signal_ts_ms INTEGER,
            signal_type TEXT,
            symbol TEXT,
            entry_price REAL,
            basis_at_entry REAL,
            liq_composition TEXT,
            confidence_band TEXT,
            session_tag TEXT
        );
        """
    )
    ts = int(datetime(2023, 11, 14, 14, 0, tzinfo=timezone.utc).timestamp() * 1000)
    con.execute("INSERT INTO mark_prices VALUES (1, ?, 'ETHUSDT', 2000, 0.0001, NULL)", (ts,))
    con.execute("INSERT INTO mark_prices VALUES (2, ?, 'ETHUSDT', 1980, 0.0001, NULL)", (ts + 900_000,))
    con.execute("INSERT INTO mark_prices VALUES (3, ?, 'SOLUSDT', 100, -0.0001, NULL)", (ts,))
    con.execute("INSERT INTO mark_prices VALUES (4, ?, 'SOLUSDT', 99, -0.0001, NULL)", (ts + 900_000,))
    con.execute("INSERT INTO liquidations VALUES (1, ?, 'ETHUSDT', 'BUY', 2000, 200, 400000, ?)", (ts, ts))
    con.execute("INSERT INTO liquidations VALUES (2, ?, 'SOLUSDT', 'BUY', 100, 600, 60000, ?)", (ts, ts))
    con.execute(
        "INSERT INTO detector_signals VALUES (1, 's1', ?, 'SHORT', 'ETHUSDT', 2000, 0.3, 'single_large', 'medium', 'us_peak')",
        (ts,),
    )
    con.commit()
    con.close()


def test_backfill_emits_matching_shadow_events(tmp_path: Path) -> None:
    db = tmp_path / "micro.db"
    _make_db(db)
    out = tmp_path / "telemetry.jsonl"
    state = tmp_path / "state.json"
    run = tmp_path / "run.json"
    args = Namespace(
        db=str(db),
        output_jsonl=str(out),
        state=str(state),
        limit_per_family=100,
        preview_limit=10,
        backfill_existing=True,
        dry_run=False,
        out_json=str(run),
    )
    payload = emitter.emit_shadow_signals(args)
    assert payload["emitted_count"] >= 3
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    families = {row["data"]["signal_family"] for row in rows}
    assert "ETH_BUY250K_SHORT_900_UTC14" in families
    assert "ETH_BUY250K_SHORT_900_UTC19" not in families
    assert "SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE" in families
    assert "SOL_BUY25K_SHORT_900_FUNDING_NEGATIVE" in families
    assert "S34_SHORT_900_BASIS_POSITIVE" in families
    assert "S34_SHORT_900_CONFIDENCE_MEDIUM" in families
    assert all(row["data"]["status"] == "SHADOW_ONLY" for row in rows)


def test_default_initializes_without_backfill(tmp_path: Path) -> None:
    db = tmp_path / "micro.db"
    _make_db(db)
    out = tmp_path / "telemetry.jsonl"
    state = tmp_path / "state.json"
    args = Namespace(
        db=str(db),
        output_jsonl=str(out),
        state=str(state),
        limit_per_family=100,
        preview_limit=10,
        backfill_existing=False,
        dry_run=False,
        out_json=str(tmp_path / "run.json"),
    )
    payload = emitter.emit_shadow_signals(args)
    assert payload["emitted_count"] == 0
    assert state.exists()
    assert not out.exists()
