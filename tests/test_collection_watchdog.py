from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import shutil
import sys
import uuid

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


def test_setup_logger_rotating() -> None:
    wd = _workdir()
    log_path = wd / "watchdog.log"
    lg = cw._setup_logger(log_path)
    lg.info("x")
    assert log_path.exists()
    shutil.rmtree(wd, ignore_errors=True)


def test_module_has_no_persistent_loop_or_operational_health_writer() -> None:
    """Retirement guard: this module must never again be able to loop or to
    write logs/health/overall.json / logs/health/watchdog.json -- both are
    exclusively owned by tools/heartbeat_watchdog.py."""
    assert not hasattr(cw, "run_loop")
    assert not hasattr(cw, "_merged_overall_health")
    assert not hasattr(cw, "write_overall_health")
    assert not hasattr(cw, "write_component_health")


def test_main_delegates_once_to_research_fitness_report_no_operational_writes(monkeypatch, tmp_path) -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db, ts_ms=1_700_000_000_000)
    csv = wd / "event_diary.csv"
    csv.write_text("ts,event\n", encoding="utf-8")
    out_path = tmp_path / "research_fitness.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collection_watchdog.py",
            "--db",
            str(db),
            "--csv",
            str(csv),
            "--symbols",
            "ETHUSDT",
            "--out",
            str(out_path),
        ],
    )
    rc = cw.main()

    assert rc in (0, 1, 2)
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in ("ready", "limited", "blocked")
    assert payload["symbols"] == ["ETHUSDT"]
    shutil.rmtree(wd, ignore_errors=True)
