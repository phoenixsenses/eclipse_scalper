from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path

from tools import db_maintenance as dm


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"db_maintenance_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t(v) VALUES ('x')")
        conn.commit()
    finally:
        conn.close()


def test_db_maintenance_main_creates_backups_and_prunes(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        trade_db = tmp / "paper.db"
        bdir = tmp / "backups"
        _mk_db(db)
        _mk_db(trade_db)

        ticks = iter(
            [
                "20260304_010101",
                "20260304_010102",
                "20260304_010103",
                "20260304_010104",
            ]
        )
        monkeypatch.setattr(dm.time, "strftime", lambda _fmt: next(ticks))

        # First run creates two backups.
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--trade-db",
                str(trade_db),
                "--backup-dir",
                str(bdir),
                "--keep",
                "1",
                "--min-free-gb",
                "0",
            ],
        )
        rc1 = dm.main()
        assert rc1 == 0
        files1 = sorted(bdir.glob("*"))
        assert len(files1) == 2

        # Second run should keep only latest 1 per stem => still 2 files total.
        rc2 = dm.main()
        assert rc2 == 0
        files2 = sorted(bdir.glob("*"))
        assert len(files2) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

