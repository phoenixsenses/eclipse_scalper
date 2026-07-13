from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
import json
from pathlib import Path

from tools import funding_rate_analysis as fra


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/track4_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_dbs(trades_db: Path, micro_db: Path) -> None:
    now = time.time()
    conn = sqlite3.connect(str(trades_db))
    try:
        conn.execute("CREATE TABLE trades (entry_time REAL, exit_time REAL, side TEXT)")
        conn.executemany(
            "INSERT INTO trades(entry_time,exit_time,side) VALUES(?,?,?)",
            [(now - 300, now - 120, "SELL"), (now - 200, now - 60, "BUY")],
        )
        conn.commit()
    finally:
        conn.close()
    conn = sqlite3.connect(str(micro_db))
    try:
        conn.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, funding_rate REAL)")
        conn.executemany(
            "INSERT INTO mark_prices(ts_ms,symbol,funding_rate) VALUES(?,?,?)",
            [(int((now - 300) * 1000), "ETHUSDT", 0.0001), (int((now - 100) * 1000), "ETHUSDT", 0.0002)],
        )
        conn.commit()
    finally:
        conn.close()


def test_funding_rate_analysis_smoke(monkeypatch) -> None:
    wd = _workdir()
    trades_db = wd / "paper.db"
    micro_db = wd / "micro.db"
    out_md = wd / "funding.md"
    out_json = wd / "funding.json"
    _mk_dbs(trades_db, micro_db)
    try:
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--trades-db",
                str(trades_db),
                "--micro-db",
                str(micro_db),
                "--symbol",
                "ETHUSDT",
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ],
        )
        rc = fra.main()
        assert rc == 0
        assert out_md.exists()
        assert out_json.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["run_summary"]["run_type"] == "funding_rate_analysis"
    finally:
        shutil.rmtree(wd, ignore_errors=True)
