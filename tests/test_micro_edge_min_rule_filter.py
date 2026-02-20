from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import extract_best_rule_delta_min_n, filter_rules_min_n


def test_filter_rules_min_n():
    rules = {
        "a": {"n": 50, "delta_vs_baseline": 0.10},
        "b": {"n": 120, "delta_vs_baseline": 0.05},
    }
    out = filter_rules_min_n(rules, min_rule_n=100)
    assert "a" not in out
    assert "b" in out


def test_extract_best_rule_delta_min_n():
    rec = {
        "naive_rules": {
            "a": {"n": 50, "delta_vs_baseline": 0.20},
            "b": {"n": 120, "delta_vs_baseline": 0.05},
            "c": {"n": 200, "delta_vs_baseline": 0.15},
        }
    }
    assert extract_best_rule_delta_min_n(rec, min_rule_n=100) == 0.15
    assert extract_best_rule_delta_min_n(rec, min_rule_n=250) is None
