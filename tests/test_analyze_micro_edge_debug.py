from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_micro_edge_debug as mod


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "symbol": "BTCUSDT",
            "ts_bucket": 1,
            "signal_idx": 0,
            "entry_idx": 1,
            "exit_idx": 2,
            "rule_name": "r1",
            "feature": {"imbalance": 0.8, "ret_1": 0.1},
            "resolved_side": "LONG",
            "entry_price": 100,
            "exit_price": 101,
            "gross_ret": 0.01,
            "cost": 0.001,
            "net_ret": 0.009,
            "label_used": 1,
            "label_used_text": "up",
            "direction_match": True,
            "timing": "t1",
        },
        {
            "ts_utc": "2026-01-01T00:00:01Z",
            "symbol": "BTCUSDT",
            "ts_bucket": 2,
            "signal_idx": 1,
            "entry_idx": 2,
            "exit_idx": 3,
            "rule_name": "r1",
            "feature": {"imbalance": -0.7, "spread": 0.2},
            "resolved_side": "SHORT",
            "entry_price": 101,
            "exit_price": 102,
            "gross_ret": -0.01,
            "cost": 0.001,
            "net_ret": -0.011,
            "label_used": -1,
            "label_used_text": "down",
            "direction_match": False,
            "timing": "t1",
        },
        {
            "ts_utc": "2026-01-01T00:00:02Z",
            "symbol": "ETHUSDT",
            "ts_bucket": 3,
            "signal_idx": 2,
            "entry_idx": 3,
            "exit_idx": 4,
            "rule_name": "r2",
            "feature": {"spread": 0.3},
            "resolved_side": "LONG",
            "entry_price": 200,
            "exit_price": 201,
            "gross_ret": 0.005,
            "cost": 0.001,
            "net_ret": 0.004,
            "label_used": 1,
            "label_used_text": "up",
            "direction_match": "true",
            "timing": "t2",
        },
        {
            "ts_utc": "2026-01-01T00:00:03Z",
            "symbol": "ETHUSDT",
            "ts_bucket": 4,
            "signal_idx": 3,
            "entry_idx": 4,
            "exit_idx": 5,
            "rule_name": "r2",
            "feature": {"spread": 0.4},
            "resolved_side": "SHORT",
            "entry_price": 201,
            "exit_price": 200,
            "gross_ret": 0.004,
            "cost": 0.001,
            "net_ret": 0.003,
            "label_used": -1,
            "label_used_text": "down",
            "direction_match": 1,
            "timing": "t2",
        },
        {
            "ts_utc": "2026-01-01T00:00:04Z",
            "symbol": "BTCUSDT",
            "ts_bucket": 5,
            "signal_idx": 4,
            "entry_idx": 5,
            "exit_idx": 6,
            "rule_name": "r2",
            "feature": {"trade_intensity": 12},
            "resolved_side": "LONG",
            "entry_price": 102,
            "exit_price": 102,
            "gross_ret": 0.0,
            "cost": 0.001,
            "net_ret": -0.001,
            "label_used": 0,
            "label_used_text": "flat",
            "direction_match": None,
            "timing": "t1",
        },
        {
            "ts_utc": "2026-01-01T00:00:05Z",
            "symbol": "ETHUSDT",
            "ts_bucket": 6,
            "signal_idx": 5,
            "entry_idx": 6,
            "exit_idx": 7,
            "rule_name": "r1",
            "feature": {"imbalance": 0.1},
            "resolved_side": "SHORT",
            "entry_price": 199,
            "exit_price": 198,
            "gross_ret": 0.006,
            "cost": 0.001,
            "net_ret": 0.005,
            "label_used": -1,
            "label_used_text": "down",
            "direction_match": "yes",
            "timing": "t2",
        },
    ]
    lines = [json.dumps(r) for r in rows]
    lines.insert(2, "{not valid json}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyzer_main_and_csv():
    root = Path(__file__).resolve().parents[1] / "logs"
    root.mkdir(parents=True, exist_ok=True)
    debug_path = root / "test_analyze_micro_edge_debug.jsonl"
    csv_path = root / "test_analyze_micro_edge_summary.csv"
    _write_fixture(debug_path)

    loaded = mod.load_debug_rows(debug_path)
    assert loaded["invalid_json"] == 1
    report = mod.build_report(loaded["rows"], top_features=20)
    assert report["overall"]["n"] == 6
    assert abs(report["overall"]["win_rate"] - (4.0 / 6.0)) < 1e-12

    code = mod.main(["--debug", str(debug_path), "--out-csv", str(csv_path), "--top-features", "10"])
    assert code == 0
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        headers = list(rd.fieldnames or [])
        rows = list(rd)
    expected = [
        "group_type",
        "group_value",
        "n",
        "win_rate",
        "avg_net_ret",
        "median_net_ret",
        "p10",
        "p90",
        "p10_gross",
        "p90_gross",
        "avg_cost",
        "avg_gross_ret",
        "dir_match_rate",
        "p90_net_negative",
    ]
    assert headers == expected
    assert len(rows) > 0
