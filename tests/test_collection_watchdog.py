from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
import shutil
import uuid

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import collection_watchdog as cw


def _mk_db(path: Path, ts_ms: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, mark_price REAL NOT NULL)"
        )
        conn.execute("INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES(?,?,?)", (ts_ms, "ETHUSDT", 100.0))
        conn.commit()
    finally:
        conn.close()


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/phase1_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_get_latest_timestamp_and_stale() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db, ts_ms=1_700_000_000_000)
    latest = cw.get_latest_timestamp(db_path=db, symbols=["ETHUSDT"])
    assert latest["_table"] == "mark_prices"
    assert latest["ETHUSDT"] == 1_700_000_000.0
    assert cw._is_stale(1000.0, stale_threshold_sec=100, now_sec=1200.0) is True
    assert cw._is_stale(1150.0, stale_threshold_sec=100, now_sec=1200.0) is False
    shutil.rmtree(wd, ignore_errors=True)


def test_run_once_dry_run_alert(monkeypatch) -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db, ts_ms=1_700_000_000_000)
    sent: list[str] = []
    overall_payloads: list[dict] = []

    async def _fake_send(msg: str, dry_run: bool) -> None:
        if dry_run:
            sent.append(msg)

    monkeypatch.setattr(cw, "_send_telegram", _fake_send)
    monkeypatch.setattr(
        cw,
        "analyze_research_fitness",
        lambda **kwargs: {
            "status": "warn",
            "db_ready": True,
            "warnings": ["no_spread:ETHUSDT"],
            "failures": [],
            "contract": {"tier": "trade_plus_liq_mark_proxy"},
        },
    )
    monkeypatch.setattr(cw, "write_overall_health", lambda payload, root="logs/health": overall_payloads.append(payload))
    monkeypatch.setattr(cw.time, "time", lambda: 1_700_000_500.0)
    logger = logging.getLogger("test_collection_watchdog")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    res = asyncio.run(
        cw.run_once(
            db_path=db,
            symbols=["ETHUSDT"],
            table="",
            stale_threshold_sec=300,
            logger=logger,
            dry_run=True,
        )
    )
    assert res["ok"] is False
    assert sent and "COLLECTOR STALE" in sent[0]
    assert overall_payloads
    comps = overall_payloads[-1]["components"]
    assert comps["data_research_fitness"]["status"] == "warning"
    assert comps["data_research_fitness"]["symbols"] == ["ETHUSDT"]
    shutil.rmtree(wd, ignore_errors=True)


def test_setup_logger_rotating() -> None:
    wd = _workdir()
    log_path = wd / "watchdog.log"
    lg = cw._setup_logger(log_path)
    lg.info("x")
    assert log_path.exists()
    shutil.rmtree(wd, ignore_errors=True)
