from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import compute_rule_thresholds, rule_fires, rule_predicted_side
from tools.micro_edge_signal_v2 import enrich_rows_with_v2


def test_v2_feature_math_smoke() -> None:
    rows = [
        {"ts_ms": 1.0, "mid": 100.0, "spread": 0.0005, "trade_intensity": 2000.0, "imbalance": 0.4, "ret_1": 0.0001, "micro_volatility": 0.001},
        {"ts_ms": 2.0, "mid": 100.1, "spread": 0.0004, "trade_intensity": 2200.0, "imbalance": 0.5, "ret_1": 0.0001, "micro_volatility": 0.0011},
        {"ts_ms": 3.0, "mid": 100.2, "spread": 0.0003, "trade_intensity": 2100.0, "imbalance": 0.6, "ret_1": -0.0001, "micro_volatility": 0.0009},
    ]
    out = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)
    assert len(out) == 3
    # Spread compresses between row0->row1 so recompress should be positive at row1.
    assert float(out[1]["v2_spread_recompress"]) > 0.0
    # Intensity decreases row1->row2, so d_intensity at row2 should be negative.
    assert float(out[2]["v2_d_intensity"]) < 0.0
    assert 0.0 <= float(out[2]["v2_confidence"]) <= 1.0


def test_v2_rule_emits_long_and_short() -> None:
    rows = []
    # first segment positive imbalance + stabilizing
    for i in range(60):
        rows.append(
            {
                "ts_ms": float(i),
                "mid": 100.0 + i * 0.01,
                "spread": 0.0006 - i * 0.000002,
                "trade_intensity": 2200.0 + (i % 4) * 50.0,
                "imbalance": 0.7,
                "ret_1": 0.0001,
                "micro_volatility": 0.001 - i * 0.000002,
            }
        )
    # second segment negative imbalance + stabilizing
    for i in range(60, 120):
        j = i - 60
        rows.append(
            {
                "ts_ms": float(i),
                "mid": 101.0 - j * 0.01,
                "spread": 0.0006 - j * 0.000002,
                "trade_intensity": 2200.0 + (j % 4) * 50.0,
                "imbalance": -0.7,
                "ret_1": -0.0001,
                "micro_volatility": 0.001 - j * 0.000002,
            }
        )
    out = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)
    thr = compute_rule_thresholds(out)
    thr["v2_min_score"] = 0.0
    thr["v2_min_persistence"] = 0.0
    thr["v2_min_confidence"] = 0.0
    sides = []
    for r in out:
        if rule_fires("micro_edge_v2_passive_alpha", r, thr):
            s = rule_predicted_side("micro_edge_v2_passive_alpha", r, default_side="LONG")
            if s:
                sides.append(s)
    assert "LONG" in sides
    assert "SHORT" in sides


def test_v2_enrich_deterministic() -> None:
    rows = [
        {"ts_ms": float(i), "mid": 100.0 + i * 0.01, "spread": 0.0004, "trade_intensity": 2000.0, "imbalance": 0.2 if i % 2 == 0 else -0.2, "ret_1": 0.0001 if i % 2 == 0 else -0.0001, "micro_volatility": 0.001}
        for i in range(100)
    ]
    a = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=("db", "SYM", 10, 1, "micro_edge_v2_passive_alpha"))
    b = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=("db", "SYM", 10, 1, "micro_edge_v2_passive_alpha"))
    assert len(a) == len(b)
    for i in (10, 25, 50, 75):
        assert float(a[i]["v2_score"]) == float(b[i]["v2_score"])
        assert float(a[i]["v2_confidence"]) == float(b[i]["v2_confidence"])
        assert float(a[i]["v3_score"]) == float(b[i]["v3_score"])
        assert float(a[i]["v3_confidence"]) == float(b[i]["v3_confidence"])


def test_v2_cache_returns_isolated_copies() -> None:
    rows = [
        {"ts_ms": float(i), "mid": 100.0 + i * 0.01, "spread": 0.0004, "trade_intensity": 2000.0, "imbalance": 0.2, "ret_1": 0.0001, "micro_volatility": 0.001}
        for i in range(20)
    ]
    cache_key = ("db", "SYM", 10, 1, "micro_edge_v2_passive_alpha")
    first = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=cache_key)
    original_score = float(first[5]["v2_score"])
    first[5]["v2_score"] = 999.0
    second = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=cache_key)
    assert float(second[5]["v2_score"]) == original_score
    assert float(first[5]["v2_score"]) == 999.0


def test_v2_enrich_sanitizes_non_finite_inputs() -> None:
    rows = [
        {"ts_ms": 1.0, "mid": 100.0, "spread": 0.0005, "trade_intensity": 2000.0, "imbalance": 0.4, "ret_1": 0.0001, "micro_volatility": 0.001},
        {"ts_ms": 2.0, "mid": 100.1, "spread": float("nan"), "trade_intensity": float("inf"), "imbalance": -0.4, "ret_1": float("-inf"), "micro_volatility": float("nan")},
        {"ts_ms": 3.0, "mid": 100.2, "spread": 0.0004, "trade_intensity": 2100.0, "imbalance": 0.3, "ret_1": 0.0002, "micro_volatility": 0.0012},
    ]
    out = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)
    for key in ("v2_score", "v2_confidence", "v3_score", "v3_confidence", "v3_toxicity"):
        assert math.isfinite(float(out[1][key]))


def test_v2_liquidation_signal_changes_scores() -> None:
    base_rows = [
        {
            "ts_ms": float(i),
            "mid": 100.0 + i * 0.01,
            "spread": 0.0005 - min(i, 10) * 0.00001,
            "trade_intensity": 8.0 + (i % 3),
            "imbalance": 0.35,
            "ret_1": 0.0001,
            "micro_volatility": 0.001,
            "liq_imbalance": 0.0,
            "liq_rate_per_sec": 0.0,
        }
        for i in range(20)
    ]
    liq_rows = copy.deepcopy(base_rows)
    for i in range(12, 20):
        liq_rows[i]["liq_imbalance"] = 0.9
        liq_rows[i]["liq_rate_per_sec"] = 12.0

    base = enrich_rows_with_v2(base_rows, bucket_sec=1, cache_key=None)
    liq = enrich_rows_with_v2(liq_rows, bucket_sec=1, cache_key=None)

    assert float(liq[-1]["v2_liq_spike"]) > 0.0
    assert float(liq[-1]["v2_liq_reversal_signal"]) > 0.0
    assert float(liq[-1]["v2_score"]) != float(base[-1]["v2_score"])
    assert float(liq[-1]["v3_score"]) != float(base[-1]["v3_score"])


def test_v3_no_lookahead_stability() -> None:
    rows = []
    for i in range(120):
        rows.append(
            {
                "ts_ms": float(i),
                "mid": 100.0 + i * 0.01,
                "spread": 0.0005 - min(i, 50) * 0.000002,
                "trade_intensity": 1800.0 + (i % 7) * 80.0,
                "imbalance": 0.4 if i < 60 else -0.4,
                "ret_1": 0.00008 if i < 60 else -0.00008,
                "micro_volatility": 0.001 + (0.0002 if i > 90 else 0.0),
            }
        )
    base = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)
    changed = copy.deepcopy(rows)
    for j in range(110, 120):
        changed[j]["trade_intensity"] = 99999.0
        changed[j]["ret_1"] = -0.01 if j % 2 else 0.01
        changed[j]["spread"] = 0.02
    mod = enrich_rows_with_v2(changed, bucket_sec=1, cache_key=None)
    for i in range(0, 100):
        assert float(base[i]["v3_score"]) == float(mod[i]["v3_score"])
        assert float(base[i]["v3_confidence"]) == float(mod[i]["v3_confidence"])
        assert float(base[i]["v3_intensity_slope"]) == float(mod[i]["v3_intensity_slope"])


def test_v3_selection_differs_from_v2() -> None:
    rows = []
    for i in range(180):
        fast = i > 40 and i < 120
        rows.append(
            {
                "ts_ms": float(i),
                "mid": 100.0 + (i * 0.005 if i < 90 else (180 - i) * 0.005),
                "spread": 0.0008 - (0.000004 * i if i < 100 else 0.0004),
                "trade_intensity": (1200.0 + i * 10.0) if fast else (900.0 + (i % 5) * 20.0),
                "imbalance": 0.65 if i % 3 else -0.35,
                "ret_1": 0.00012 if i < 90 else -0.00012,
                "micro_volatility": 0.0012 if fast else 0.0007,
            }
        )
    out = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)
    thr = compute_rule_thresholds(out)
    v2_idx = [i for i, r in enumerate(out) if rule_fires("micro_edge_v2_passive_alpha", r, thr)]
    v3_idx = [i for i, r in enumerate(out) if rule_fires("micro_edge_v3_passive_alpha", r, thr)]
    assert len(v2_idx) > 0
    assert len(v3_idx) > 0
    assert set(v2_idx) != set(v3_idx)
