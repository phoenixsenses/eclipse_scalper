from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from core.performance_monitor import compute_daily_metrics, detect_anomalies, weekly_digest_markdown


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/track3_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE trades (exit_time REAL, pnl_bps REAL)")
        now = time.time()
        rows = [(now - 3600 + i * 120, -2.0 if i % 5 else 3.0) for i in range(120)]
        conn.executemany("INSERT INTO trades(exit_time,pnl_bps) VALUES(?,?)", rows)
        conn.commit()
    finally:
        conn.close()


def _mk_health(path: Path) -> None:
    now = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(100):
        reg = "UP" if i % 3 else "DOWN"
        lines.append(json.dumps({"ts": now - 3600 + i * 30, "regime": reg}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_metrics_and_anomalies() -> None:
    wd = _workdir()
    db = wd / "paper_trades.db"
    hh = wd / "health.jsonl"
    try:
        _mk_db(db)
        _mk_health(hh)
        m = compute_daily_metrics(trades_db=db, health_history_path=hh, journal_path=wd / "missing.jsonl")
        assert m.trade_count > 0
        anomalies = detect_anomalies(m, history_db=db, expected_sigma_bps=1.0)
        assert isinstance(anomalies, list)
        weekly = weekly_digest_markdown([m])
        assert "Weekly Digest" in weekly
    finally:
        shutil.rmtree(wd, ignore_errors=True)

