from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_smoke import build_json_record


def test_build_json_record_schema_is_stable() -> None:
    rep = {
        "symbol": "ETHUSDT",
        "raw_count": 120,
        "bucket_count": 40,
        "up": 12,
        "down": 9,
        "flat": 5,
        "baseline_hit_rate": 0.57,
        "feature_corr": {"imbalance": 0.12, "spread": None},
        "rules": {
            "micro_edge_v3_passive_alpha": {"hit_rate": 0.64, "n": 22, "delta_vs_baseline": 0.07, "baseline": 0.57},
            "broken_rule": "skip-me",
        },
        "label_definition": {
            "timing": "signal at t, entry at t+1 mark, exit at t+1+h mark",
            "label": "sign((mark[t+1+h]/mark[t+1])-1) with threshold",
        },
    }
    record = build_json_record(rep=rep, lookback_min=240, bucket_sec=5, horizon_sec=60, min_rule_n=25)

    assert set(record) == {
        "ts_utc",
        "symbol",
        "lookback_min",
        "bucket_sec",
        "horizon_sec",
        "raw_rows",
        "bucket_rows",
        "label_counts",
        "baseline_hit_rate",
        "min_rule_n",
        "correlations",
        "naive_rules",
        "label_definition",
        "run_summary",
    }
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", str(record["ts_utc"]))
    assert record["symbol"] == "ETHUSDT"
    assert record["raw_rows"] == 120
    assert record["bucket_rows"] == 40
    assert record["label_counts"] == {"up": 12, "down": 9, "flat": 5}
    assert record["min_rule_n"] == 25
    assert record["correlations"]["spread"] is None
    assert set(record["naive_rules"]) == {"micro_edge_v3_passive_alpha"}
    assert record["naive_rules"]["micro_edge_v3_passive_alpha"] == {
        "hit_rate": 0.64,
        "n": 22,
        "delta_vs_baseline": 0.07,
    }
    assert record["run_summary"]["version"] == "v1"
    assert record["run_summary"]["run_type"] == "micro_edge_smoke"
    assert record["run_summary"]["inputs"]["symbol"] == "ETHUSDT"
    assert record["run_summary"]["metrics"]["rule_count"] == 1
    assert record["label_definition"]["horizon_steps"] == 12
    assert record["label_definition"]["threshold"] == 0.0002
    assert record["label_definition"]["label_values"] == {"up": 1, "flat": 0, "down": -1}


def test_build_json_record_fills_missing_label_definition_defaults() -> None:
    record = build_json_record(
        rep={
            "symbol": "BTCUSDT",
            "raw_count": 0,
            "bucket_count": 0,
            "up": 0,
            "down": 0,
            "flat": 0,
            "baseline_hit_rate": None,
            "feature_corr": {},
            "rules": {},
        },
        lookback_min=60,
        bucket_sec=3,
        horizon_sec=30,
        min_rule_n=10,
    )

    assert record["naive_rules"] == {}
    assert record["run_summary"]["metrics"]["rule_count"] == 0
    assert record["label_definition"]["horizon_steps"] == 10
    assert "hit_definition" in record["label_definition"]
    assert "baseline_definition" in record["label_definition"]
