from __future__ import annotations

import json
from pathlib import Path
import sys
import shutil

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.build_canonical_dataset import BuildContext, _flatten_execution_journal, load_jsonl


def test_execution_journal_parse_smoke() -> None:
    test_root = REPO_ROOT / "localtests" / "execution_journal_parse_smoke"
    if test_root.exists():
        shutil.rmtree(test_root, ignore_errors=True)
    test_root.mkdir(parents=True, exist_ok=True)
    journal_path = test_root / "execution_journal.jsonl"
    rows = [
        {
            "ts": 1770000000.0,
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "ENTRY-BTCUSD-a1",
                "state_to": "SUBMITTED",
                "reason": "router_send",
                "meta": {"k": "BTCUSDT"},
            },
        },
        {
            "timestamp": "2026-02-02T00:00:01Z",
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "ENTRY-BTCUSD-a1",
                "state_to": "ACKED",
                "reason": "exchange_ack",
                "meta": {"k": "BTCUSDT"},
            },
        },
        {
            "time": "2026-02-02T00:00:02Z",
            "event": "state.transition",
            "data": {
                "machine": "order_intent",
                "entity": "EXIT-BTCUSD-a1",
                "state_to": "SUBMITTED",
                "reason": "router_send",
                "meta": {"k": "BTCUSDT"},
            },
        },
    ]
    with journal_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    raw = load_jsonl(journal_path)
    ctx = BuildContext(
        repo_root=REPO_ROOT,
        out_dir=test_root / "out",
        symbols=["BTCUSDT"],
        start_ts=pd.Timestamp("2026-02-02T00:00:00Z"),
        end_ts=pd.Timestamp("2026-02-02T00:01:00Z"),
    )
    parsed = _flatten_execution_journal(raw, ctx)

    assert not parsed.empty
    assert parsed["entry_candidate"].dtype == bool
    assert parsed["entry_executed"].dtype == bool
    assert parsed["exit_candidate"].dtype == bool
    assert parsed["symbol"].eq("BTCUSDT").any()
    assert parsed["entry_candidate"].any()
    assert parsed["entry_executed"].any()
    assert parsed["exit_candidate"].any()
