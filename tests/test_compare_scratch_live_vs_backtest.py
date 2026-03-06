from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from tools import compare_scratch_live_vs_backtest as cmp


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"scratch_cmp_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_compare_scratch_live_vs_backtest_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "paper_trades.db"
        sell_json = tmp / "sell.json"
        buy_json = tmp / "buy.json"
        out_md = tmp / "cmp.md"
        out_json = tmp / "cmp.json"

        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE trades (side TEXT, exit_reason TEXT)")
            conn.executemany(
                "INSERT INTO trades(side,exit_reason) VALUES (?,?)",
                [("sell", "scratch"), ("sell", "horizon"), ("buy", "scratch"), ("buy", "scratch")],
            )
            conn.commit()
        finally:
            conn.close()

        sell_json.write_text(json.dumps({"baseline": {"n": 10, "scratch_frac": 0.4}}), encoding="utf-8")
        buy_json.write_text(json.dumps({"baseline": {"n": 10, "scratch_frac": 0.5}}), encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--trade-db",
                str(db),
                "--backtest-sell-json",
                str(sell_json),
                "--backtest-buy-json",
                str(buy_json),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ],
        )
        assert cmp.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "ok"
        assert payload["run_summary"]["run_type"] == "compare_scratch_live_vs_backtest"
        assert "delta_sell_abs" in payload
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
