from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from tools import db_introspect as di


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"db_introspect_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_db_introspect_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        out_md = tmp / "schema.md"
        out_json = tmp / "tables.json"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT, price REAL)")
            conn.execute("CREATE INDEX idx_agg_symbol ON agg_trades(symbol)")
            conn.commit()
        finally:
            conn.close()
        monkeypatch.setattr("sys.argv", ["x", "--db", str(db), "--out-md", str(out_md), "--out-json", str(out_json)])
        assert di.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["run_summary"]["run_type"] == "db_introspect"
        assert len(payload["tables"]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
