from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import event_passes_feature_bounds


def test_feature_filter_bounds_and_counters():
    counters = {"filter_drop_missing_key": 0, "filter_drop_below_min": 0, "filter_drop_above_max": 0}
    feat = {"spread": 0.0003, "trade_intensity": 1200.0}
    assert event_passes_feature_bounds(feat, {"trade_intensity": 1000.0}, {"spread": 0.0005}, counters) is True

    assert event_passes_feature_bounds(feat, {"imbalance": 0.1}, {}, counters) is False
    assert counters["filter_drop_missing_key"] == 1

    assert event_passes_feature_bounds(feat, {"trade_intensity": 1500.0}, {}, counters) is False
    assert counters["filter_drop_below_min"] == 1

    assert event_passes_feature_bounds(feat, {}, {"spread": 0.0002}, counters) is False
    assert counters["filter_drop_above_max"] == 1

