from __future__ import annotations

import sqlite3
import sys
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import db_maintenance


def _mk_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, x TEXT)")
        conn.execute("INSERT INTO t(x) VALUES('a')")
        conn.commit()
    finally:
        conn.close()


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/phase2_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_db_maintenance_wal_alert(monkeypatch) -> None:
    wd = _workdir()
    db = wd / "micro.db"
    _mk_db(db)
    wal_path = wd / "synthetic_big.wal"
    wal_path.write_bytes(b"x" * (2 * 1024 * 1024))
    alerts: list[str] = []
    monkeypatch.setattr(db_maintenance, "_send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(db_maintenance, "_wal_path", lambda _db: wal_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "db_maintenance",
            "--db",
            str(db),
            "--backup-dir",
            str(wd / "bak"),
            "--max-wal-mb",
            "1.0",
            "--min-free-gb",
            "0.0",
        ],
    )
    try:
        rc = db_maintenance.main()
        assert rc == 1
        assert alerts and "WAL too large" in alerts[0]
    finally:
        shutil.rmtree(wd, ignore_errors=True)
