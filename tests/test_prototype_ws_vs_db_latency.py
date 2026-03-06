from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from tools import prototype_ws_vs_db_latency as p


def _mk_local_tmp() -> Path:
    t = Path("localtests") / f"ws_db_{uuid.uuid4().hex[:8]}"
    t.mkdir(parents=True, exist_ok=True)
    return t.resolve()


def test_prototype_ws_vs_db_latency_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        hb = tmp / "collector_heartbeat.json"
        out_md = tmp / "wsdb.md"
        out_json = tmp / "wsdb.json"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT)")
            conn.execute("INSERT INTO agg_trades(ts_ms, symbol) VALUES (?, ?)", (int(time.time() * 1000.0), "ETHUSDT"))
            conn.commit()
        finally:
            conn.close()
        hb.write_text(json.dumps({"connected": True, "progress_lag_sec": 1}), encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--symbol",
                "ETHUSDT",
                "--collector-heartbeat",
                str(hb),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ],
        )
        assert p.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "ok"
        assert payload["run_summary"]["run_type"] == "prototype_ws_vs_db_latency"
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
