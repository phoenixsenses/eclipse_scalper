from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.micro_edge_backtest as backtest_mod


def _rows_for_split_test(n: int = 40):
    rows = []
    mid = 100.0
    for i in range(n):
        imb = 0.9 if i % 2 == 0 else -0.9
        mid = mid * (1.0005 if imb > 0 else 0.9995)
        rows.append(
            {
                "ts_ms": float(i * 1000),
                "mid": float(mid),
                "imbalance": float(imb),
                "trade_intensity": float(10 + i),
                "spread": 0.0002,
                "ret_1": (0.0001 if i % 3 == 0 else -0.0001),
            }
        )
    return rows


def test_debug_split_has_no_cross_file_key_overlap(monkeypatch):
    rows = _rows_for_split_test()
    labels = [1 if i % 2 == 0 else -1 for i in range(len(rows))]
    thresholds = {"imb_q10": -1.0, "imb_q90": 1.0, "int_q90": 0.0, "spr_q90": 1.0}
    long_path = Path("dbg_long.jsonl")
    short_path = Path("dbg_short.jsonl")
    all_path = Path("dbg_all.jsonl")
    written: dict[str, list[dict]] = {}

    def _fake_append(path: Path, record: dict):
        written.setdefault(str(path), []).append(json.loads(json.dumps(record)))

    monkeypatch.setattr(backtest_mod, "append_jsonl", _fake_append)

    sim = backtest_mod.simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds=thresholds,
        labels=labels,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
        debug_samples=30,
        debug_symbol="BTCUSDT",
        debug_out_path=all_path,
        debug_out_long_path=long_path,
        debug_out_short_path=short_path,
    )

    long_rows = written.get(str(long_path), [])
    short_rows = written.get(str(short_path), [])
    long_keys = {f"{r['symbol']}|{r['ts_bucket']}|{r['signal_idx']}" for r in long_rows}
    short_keys = {f"{r['symbol']}|{r['ts_bucket']}|{r['signal_idx']}" for r in short_rows}
    assert long_keys.isdisjoint(short_keys)

    union_count = len(long_keys | short_keys)
    assert union_count == len(long_rows) + len(short_rows)
    assert union_count == int(sim["debug_stats"]["debug_written"])
