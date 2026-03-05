from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from tools import reconcile_paper_vs_backtest as rpb


def _workdir() -> Path:
    p = Path("eclipse_scalper/localtests/track4_tests") / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_paper_db(path: Path) -> None:
    now = time.time()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE trades (entry_time REAL, exit_time REAL, side TEXT, regime TEXT, entry_price REAL, exit_price REAL, pnl_bps REAL, max_adverse_bps REAL, exit_type TEXT, exit_reason TEXT)"
        )
        conn.execute(
            "INSERT INTO trades(entry_time,exit_time,side,regime,entry_price,exit_price,pnl_bps,max_adverse_bps,exit_type,exit_reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (now - 120, now - 20, "SELL", "UP", 100.0, 99.8, 20.0, 4.0, "horizon", "ok"),
        )
        conn.commit()
    finally:
        conn.close()


def test_reconcile_report_generation(monkeypatch) -> None:
    wd = _workdir()
    paper = wd / "paper.db"
    rank = wd / "rank.json"
    bt = wd / "bt.json"
    out = wd / "RECON.md"
    _mk_paper_db(paper)
    rank.write_text(json.dumps({"ranking": [{"npa_core": 0.0001, "fill_rate_after_gate": 0.6, "avg_adverse_bps_on_fills": 3.0}]}), encoding="utf-8")
    bt.write_text(json.dumps({"trades": [{"entry_time": time.time() - 125, "entry_price": 100.1, "exit_price": 99.9, "pnl_bps": 18.0, "max_adverse_bps": 3.5}]}), encoding="utf-8")
    try:
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--paper-db",
                str(paper),
                "--rank-json",
                str(rank),
                "--backtest-json",
                str(bt),
                "--out",
                str(out),
            ],
        )
        rc = rpb.main()
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        assert "RECONCILIATION" in text
        assert "Trade-level Matching" in text
    finally:
        shutil.rmtree(wd, ignore_errors=True)

