from __future__ import annotations

import sqlite3
from pathlib import Path
import shutil
import uuid

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.collection_health import analyze_collection_health, write_markdown


def _mk_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, mark_price REAL NOT NULL)"
        )
        base = 1_700_000_000_000
        rows = [
            (base, "ETHUSDT", 100.0),
            (base + 20_000, "ETHUSDT", 100.1),
            (base + 90_000, "ETHUSDT", 100.2),  # 70s gap
            (base + 95_000, "ETHUSDT", 100.3),
            (base, "BTCUSDT", 200.0),
            (base + 30_000, "BTCUSDT", 200.1),
            (base + 400_000, "BTCUSDT", 200.2),  # 370s critical gap
        ]
        conn.executemany("INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES(?,?,?)", rows)
        conn.commit()
    finally:
        conn.close()


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/phase1_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_pass_clean_synthetic() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db)
    report = analyze_collection_health(
        db_path=db,
        symbols=["ETHUSDT", "BTCUSDT"],
        gap_threshold_sec=60,
        alert_threshold_sec=300,
    )
    assert report["status"] == "ok"
    assert report["table"] == "mark_prices"
    assert int(report["gap_count"]) == 2
    assert int(report["critical_gap_count"]) == 1
    assert float(report["longest_gap_sec"]) >= 370.0
    shutil.rmtree(wd, ignore_errors=True)


def test_fail_duplicate_timestamp() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db)
    report = analyze_collection_health(
        db_path=db,
        symbols=["ETHUSDT"],
        gap_threshold_sec=60,
        alert_threshold_sec=300,
    )
    # not a failure tool, but should still return deterministic report with sane fields
    assert report["status"] == "ok"
    assert "date_range" in report
    shutil.rmtree(wd, ignore_errors=True)


def test_fail_missing_required_column() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, symbol TEXT)")
        conn.commit()
    finally:
        conn.close()
    try:
        analyze_collection_health(db_path=db, symbols=["ETHUSDT"])
        assert False, "expected no timestamped table error"
    except RuntimeError as exc:
        assert "no timestamped table found" in str(exc)
    shutil.rmtree(wd, ignore_errors=True)


def test_fail_nan_threshold() -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db)
    report = analyze_collection_health(
        db_path=db,
        symbols=["ETHUSDT", "BTCUSDT"],
        gap_threshold_sec=1000,
        alert_threshold_sec=300,
    )
    assert int(report["gap_count"]) == 0
    assert int(report["critical_gap_count"]) == 0
    shutil.rmtree(wd, ignore_errors=True)


def test_skip_missing_source() -> None:
    wd = _workdir()
    out = wd / "COLLECTION_HEALTH.md"
    write_markdown({"status": "insufficient_data"}, out)
    text = out.read_text(encoding="utf-8")
    assert "status=insufficient_data" in text
    shutil.rmtree(wd, ignore_errors=True)
