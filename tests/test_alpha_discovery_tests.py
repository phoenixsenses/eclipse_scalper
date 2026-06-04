from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tools import alpha_discovery_tests as adt


def _ts(hour: int, day: int = 1) -> int:
    return int(datetime(2026, 1, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


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
            symbol TEXT,
            entry_price REAL,
            basis_at_entry REAL,
            liq_composition TEXT,
            confidence_band TEXT,
            session_tag TEXT,
            fingerprint_class TEXT,
            entry_book_state TEXT
        );
        """
    )
    row_id = 1
    for i in range(30):
        ts = _ts(14, 1 + i)
        con.execute("INSERT INTO liquidations VALUES (?, ?, 'ETHUSDT', 'BUY', 2000, 200, 400000, ?)", (row_id, ts, ts))
        con.execute("INSERT INTO mark_prices VALUES (?, ?, 'ETHUSDT', 2000, 0.0001, NULL)", (row_id, ts))
        con.execute("INSERT INTO mark_prices VALUES (?, ?, 'ETHUSDT', 1980, 0.0001, NULL)", (row_id + 1000, ts + 900000))
        row_id += 1
    for i in range(10):
        ts = _ts(3, 1 + i)
        con.execute("INSERT INTO liquidations VALUES (?, ?, 'SOLUSDT', 'BUY', 100, 600, 60000, ?)", (row_id, ts, ts))
        con.execute("INSERT INTO mark_prices VALUES (?, ?, 'SOLUSDT', 100, -0.0001, NULL)", (row_id, ts))
        con.execute("INSERT INTO mark_prices VALUES (?, ?, 'SOLUSDT', 99, -0.0001, NULL)", (row_id + 1000, ts + 900000))
        row_id += 1
    con.commit()
    con.close()


def _args(db: Path, telemetry: Path) -> Namespace:
    return Namespace(
        db=str(db),
        telemetry_path=str(telemetry),
        max_events=100,
        min_n=20,
        min_wr=60.0,
        min_mean_bps=8.0,
        folds=5,
        fee_rt_bps="2,4,8,10",
        out_md="",
        out_json="",
    )


def test_alpha_discovery_promotes_targeted_lane(tmp_path: Path) -> None:
    db = tmp_path / "micro.db"
    telemetry = tmp_path / "telemetry.jsonl"
    _make_db(db)
    payload = adt.build_payload(_args(db, telemetry))
    names = {row["candidate"] for row in payload["promoted"]}
    assert "ETHUSDT_BUY250000_SHORT_900_UTC14" in names
    assert payload["candidate_count"] > 0


def test_shadow_telemetry_summary_reads_research_events(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "event": "research.shadow_signal",
                "data": {
                    "signal_family": "X",
                    "forward_labels": {"return_bps_900s": 12.5},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = adt._load_shadow_telemetry(telemetry)
    assert summary["rows"] == 1
    assert summary["families"][0]["family"] == "X"
    assert summary["families"][0]["mean_bps"] == 12.5
