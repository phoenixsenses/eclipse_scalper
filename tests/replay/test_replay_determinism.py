from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from src.microphys.replay import compute_replay_parity, load_live_fill_rows, load_simulated_fill_rows


def _tmp_dir() -> Path:
    p = Path("localtests") / f"replay_determinism_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_replay_parity_is_deterministic_for_shuffled_inputs() -> None:
    sim_rows = [
        {
            "event_id": "s2",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "entry_time": 1700000005.0,
            "filled": True,
            "fill_delay_sec": 6.0,
            "pnl_bps": -0.7,
            "max_adverse_bps": 2.0,
        },
        {
            "event_id": "s1",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "entry_time": 1700000000.0,
            "filled": True,
            "fill_delay_sec": 4.0,
            "pnl_bps": 1.2,
            "max_adverse_bps": 1.0,
        },
    ]
    live_rows = [
        {
            "event_id": "l1",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "entry_time": 1700000000.2,
            "elapsed_sec": 5.0,
            "pnl_bps": 1.0,
            "max_adverse_bps": 1.1,
        },
        {
            "event_id": "l2",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "entry_time": 1700000004.9,
            "elapsed_sec": 7.0,
            "pnl_bps": -0.4,
            "max_adverse_bps": 1.8,
        },
    ]
    a = compute_replay_parity(sim_rows, live_rows, match_window_sec=1.0).to_dict()
    b = compute_replay_parity(list(reversed(sim_rows)), list(reversed(live_rows)), match_window_sec=1.0).to_dict()
    assert a == b
    assert int(a["matched_count"]) == 2
    assert abs(float(a["mean_abs_dt_sec"])) > 0.0


def test_loaders_and_parity_from_files() -> None:
    root = _tmp_dir()
    sim_path = root / "sim.jsonl"
    sim_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "evt1",
                        "symbol": "ETHUSDT",
                        "side": "BUY",
                        "entry_time": 1700001000.0,
                        "filled": True,
                        "fill_delay_sec": 3.0,
                        "pnl_bps": 0.5,
                        "max_adverse_bps": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "event_id": "evt2",
                        "symbol": "ETHUSDT",
                        "side": "SELL",
                        "entry_time": 1700001010.0,
                        "filled": True,
                        "fill_delay_sec": 8.0,
                        "pnl_bps": -0.2,
                        "max_adverse_bps": 1.6,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    db_path = root / "paper_trades.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trades (symbol TEXT, side TEXT, entry_time REAL, exit_time REAL, elapsed_sec REAL, pnl_bps REAL, max_adverse_bps REAL)"
        )
        conn.executemany(
            "INSERT INTO trades(symbol,side,entry_time,exit_time,elapsed_sec,pnl_bps,max_adverse_bps) VALUES (?,?,?,?,?,?,?)",
            [
                ("ETHUSDT", "BUY", 1700001000.3, 1700001005.3, 5.0, 0.4, 0.9),
                ("ETHUSDT", "SELL", 1700001010.1, 1700001019.1, 9.0, -0.1, 1.5),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    sim = load_simulated_fill_rows(sim_path)
    live = load_live_fill_rows(db_path)
    res = compute_replay_parity(sim, live, match_window_sec=1.0)
    assert res.sim_count == 2
    assert res.live_count == 2
    assert res.matched_count == 2
    assert abs(res.mean_pnl_bps_delta) <= 0.2
