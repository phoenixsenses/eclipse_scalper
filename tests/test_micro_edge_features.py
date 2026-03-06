from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import build_bucket_features, evaluate_naive_rules


def test_build_bucket_features_has_spread_and_imbalance():
    trades = [
        {"ts_ms": 1_000, "price": 100.2, "quantity": 1.0, "is_buyer_maker": 0},
        {"ts_ms": 1_500, "price": 100.1, "quantity": 2.0, "is_buyer_maker": 1},
        {"ts_ms": 6_100, "price": 100.4, "quantity": 1.5, "is_buyer_maker": 0},
        {"ts_ms": 6_700, "price": 100.5, "quantity": 1.0, "is_buyer_maker": 0},
    ]
    marks = [
        {"ts_ms": 1_000, "mark_price": 100.0},
        {"ts_ms": 6_000, "mark_price": 100.3},
    ]
    liqs = [
        {"ts_ms": 1_100, "side": "sell", "quantity": 3.0, "price": 99.9},
        {"ts_ms": 6_300, "side": "buy", "quantity": 1.0, "price": 100.4},
    ]
    rows = build_bucket_features(trades=trades, marks=marks, liqs=liqs, bucket_sec=5, vol_window=3)
    assert len(rows) >= 2
    first = rows[0]
    second = rows[1]
    assert first["spread"] is not None
    assert first["imbalance"] is not None
    assert first["liq_imbalance"] is not None
    assert second["spread"] is not None
    assert second["imbalance"] is not None
    assert second["liq_rate_per_sec"] is not None


def test_evaluate_naive_rules_non_empty():
    rows = [
        {"imbalance": -0.9, "trade_intensity": 10.0, "spread": 0.01, "ret_1": -0.001, "liq_imbalance": -0.8, "liq_rate_per_sec": 20.0},
        {"imbalance": 0.95, "trade_intensity": 11.0, "spread": 0.01, "ret_1": 0.001, "liq_imbalance": 0.9, "liq_rate_per_sec": 21.0},
        {"imbalance": -0.8, "trade_intensity": 9.5, "spread": 0.03, "ret_1": 0.002, "liq_imbalance": -0.7, "liq_rate_per_sec": 19.0},
        {"imbalance": 0.9, "trade_intensity": 12.0, "spread": 0.04, "ret_1": -0.003, "liq_imbalance": 0.85, "liq_rate_per_sec": 22.0},
        {"imbalance": 0.0, "trade_intensity": 1.0, "spread": 0.005, "ret_1": 0.0, "liq_imbalance": 0.0, "liq_rate_per_sec": 0.0},
    ]
    labels = [-1, 1, -1, 1, 0]
    rules = evaluate_naive_rules(rows, labels, baseline_hit_rate=0.5)
    assert rules
    assert any((v.get("n") or 0) > 0 for v in rules.values())
    assert "liquidation_spike_reversal" in rules
