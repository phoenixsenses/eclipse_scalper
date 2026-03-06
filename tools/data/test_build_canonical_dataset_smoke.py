from __future__ import annotations

import sqlite3
from pathlib import Path
import sys
import json
import shutil

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.build_canonical_dataset import main


def _seed_micro_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER);
        CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT, price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER);
        CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, side TEXT, price REAL, quantity REAL, notional REAL, trade_time_ms INTEGER);
        """
    )
    rows_mark = [
        (1770000000000, "BTCUSDT", 100.0, 0.0, None),
        (1770000001000, "BTCUSDT", 101.0, 0.0, None),
        (1770000000000, "ETHUSDT", 50.0, 0.0, None),
    ]
    rows_trade = [
        (1770000000500, "BTCUSDT", 100.5, 2.0, 201.0, 0),
        (1770000001500, "BTCUSDT", 101.2, 1.0, 101.2, 1),
        (1770000000500, "ETHUSDT", 50.2, 3.0, 150.6, 0),
    ]
    conn.executemany("INSERT INTO mark_prices VALUES (?, ?, ?, ?, ?)", rows_mark)
    conn.executemany("INSERT INTO agg_trades VALUES (?, ?, ?, ?, ?, ?)", rows_trade)
    conn.commit()
    conn.close()


def test_build_canonical_dataset_smoke() -> None:
    repo = REPO_ROOT / "localtests" / "canonical_build_smoke"
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    (repo / "data").mkdir(parents=True, exist_ok=True)
    (repo / "logs").mkdir(parents=True, exist_ok=True)
    (repo / "tmp").mkdir(parents=True, exist_ok=True)

    _seed_micro_db(repo / "data" / "microstructure.db")

    event_diary = pd.DataFrame(
        [
            {"event_id": "E1", "ts_ms": 1770000000000, "timestamp": "2026-02-02T00:00:00Z", "symbol": "BTCUSDT", "event_type": "PRICE_MOVE", "direction": "UP"},
            {"event_id": "E2", "ts_ms": 1770000000000, "timestamp": "2026-02-02T00:00:00Z", "symbol": "ETHUSDT", "event_type": "PRICE_MOVE", "direction": "DOWN"},
        ]
    )
    event_diary.to_csv(repo / "data" / "event_diary.csv", index=False)
    (repo / "tmp" / "event_diary_shadow.csv").write_text("timestamp,symbol,event_type\n", encoding="utf-8")

    exec_journal = [
        {
            "ts": "2026-02-02T00:00:01Z",
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "ENTRY-BTCUSD-abc123",
                "state_to": "SUBMITTED",
                "reason": "router_send",
                "meta": {"k": "BTCUSDT"},
            },
        },
        {
            "ts": "2026-02-02T00:00:02Z",
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "ENTRY-BTCUSD-abc123",
                "state_to": "ACKED",
                "reason": "exchange_ack",
                "meta": {"k": "BTCUSDT"},
            },
        },
        {
            "ts": "2026-02-02T00:00:03Z",
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "EXIT-BTCUSD-abc123",
                "state_to": "SUBMITTED",
                "reason": "router_send",
                "meta": {"k": "BTCUSDT"},
            },
        },
    ]
    with (repo / "logs" / "execution_journal.jsonl").open("w", encoding="utf-8") as fh:
        for row in exec_journal:
            fh.write(json.dumps(row) + "\n")

    ohlcv = pd.DataFrame(
        [
            {"timestamp": "2026-02-02T00:00:00Z", "symbol": "BTCUSDT", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
            {"timestamp": "2026-02-02T00:01:00Z", "symbol": "BTCUSDT", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 11},
            {"timestamp": "2026-02-02T00:00:00Z", "symbol": "ETHUSDT", "open": 50, "high": 51, "low": 49, "close": 50.5, "volume": 20},
        ]
    )
    ohlcv.to_csv(repo / "data" / "ohlcv_test.csv", index=False)

    out_dir = repo / "data" / "canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ohlcv_leak.csv").write_text("timestamp,symbol,open,high,low,close,volume\n", encoding="utf-8")
    rc = main(
        [
            "--symbols",
            "BTCUSDT",
            "--start",
            "2026-02-02T00:00:00Z",
            "--end",
            "2026-02-02T00:00:03Z",
            "--out",
            str(out_dir),
            "--repo-root",
            str(repo),
        ]
    )
    assert rc == 0

    for name in [
        "canonical_microstructure.parquet",
        "canonical_events.parquet",
        "canonical_ohlcv.parquet",
        "canonical_merged.parquet",
        "manifest.json",
        "build_log.txt",
    ]:
        assert (out_dir / name).exists(), f"missing output {name}"

    merged = pd.read_parquet(out_dir / "canonical_merged.parquet")
    required = {
        "timestamp",
        "symbol",
        "price",
        "bid",
        "ask",
        "spread",
        "volume",
        "volatility",
        "signal_state",
        "entry_candidate",
        "entry_executed",
        "exit_candidate",
        "position_state",
    }
    assert required.issubset(set(merged.columns))
    assert str(merged["timestamp"].dtype).endswith("UTC]")
    assert set(merged["symbol"].unique()) == {"BTCUSDT"}
    assert merged["entry_candidate"].any()
    assert merged["entry_executed"].any()
    assert merged["exit_candidate"].any()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    discovered = " ".join(manifest.get("discovered_input_files", [])).lower()
    assert "data/canonical" not in discovered
    assert "/tmp/" not in discovered
    assert "\\tmp\\" not in discovered
    assert "pytest-of-" not in discovered
