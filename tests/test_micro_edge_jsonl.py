from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import extract_best_rule_delta, parse_jsonl_lines, serialize_record
from tools.micro_edge_smoke import build_json_record


def test_json_record_serialization_sample():
    rep = {
        "symbol": "BTCUSDT",
        "raw_count": 100,
        "bucket_count": 50,
        "up": 10,
        "down": 20,
        "flat": 20,
        "baseline_hit_rate": 0.55,
        "feature_corr": {"ret_1": 0.01},
        "rules": {
            "imbalance_gt_0.6": {"hit_rate": 0.60, "n": 40, "delta_vs_baseline": 0.05},
        },
    }
    rec = build_json_record(rep, lookback_min=240, bucket_sec=5, horizon_sec=60)
    s = serialize_record(rec)
    assert '"symbol": "BTCUSDT"' in s
    assert rec["naive_rules"]["imbalance_gt_0.6"]["n"] == 40


def test_parse_jsonl_lines_filters_invalid():
    lines = [
        '{"symbol":"BTCUSDT","baseline_hit_rate":0.5}',
        'not-json',
        '',
        '{"symbol":"ETHUSDT","naive_rules":{}}',
    ]
    out = parse_jsonl_lines(lines)
    assert len(out) == 2
    assert out[0]["symbol"] == "BTCUSDT"
    assert out[1]["symbol"] == "ETHUSDT"


def test_extract_best_rule_delta():
    rec = {
        "naive_rules": {
            "a": {"delta_vs_baseline": 0.01},
            "b": {"delta_vs_baseline": 0.03},
            "c": {"delta_vs_baseline": -0.02},
        }
    }
    assert extract_best_rule_delta(rec) == 0.03
