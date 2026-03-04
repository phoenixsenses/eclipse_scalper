from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from tools import preflight_check as pf


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"preflight_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _mk_db(path: Path, ts_ms: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE mark_prices (ts INTEGER, price REAL)")
        conn.execute("INSERT INTO mark_prices(ts,price) VALUES (?,?)", (int(ts_ms), 100.0))
        conn.commit()
    finally:
        conn.close()


def test_preflight_passes_with_fresh_db(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        db = tmp / "micro.db"
        now_ms = int(time.time() * 1000.0)
        _mk_db(db, now_ms)
        out_json = tmp / "reports" / "preflight.json"
        out_md = tmp / "reports" / "preflight.md"

        monkeypatch.setenv("SCALPER_DRY_RUN", "1")
        monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT")
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--trade-db",
                str(tmp / "paper.db"),
                "--max-db-stale-sec",
                "3600",
                "--min-free-gb",
                "0",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert pf.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert out_md.exists()
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
