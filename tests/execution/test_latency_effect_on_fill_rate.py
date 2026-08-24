from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from execution.passive_execution_simulator import simulate_passive_fill


def _base_params() -> dict:
    return {
        "seed": 7,
        "base_touch": 0.95,
        "base_full_cond_touch": 0.9,
        "base_adverse_bps": 0.1,
        "maker_fee_bps": 0.5,
        "edges": {
            "spread": [0.00005, 0.00010],
            "trade_intensity": [2000.0, 4000.0],
            "vol_proxy": [0.00005, 0.00010],
            "imbalance_for_fill": [0.4, 0.8],
        },
        "touch_rates": {
            "spread": [0.95, 0.95, 0.95],
            "trade_intensity": [0.95, 0.95, 0.95],
            "vol_proxy": [0.95, 0.95, 0.95],
            "imbalance_for_fill": [0.95, 0.95, 0.95],
        },
        "full_rates": {
            "spread": [0.9, 0.9, 0.9],
            "trade_intensity": [0.9, 0.9, 0.9],
            "vol_proxy": [0.9, 0.9, 0.9],
            "imbalance_for_fill": [0.9, 0.9, 0.9],
        },
        "adverse_means": {
            "spread": [0.1, 0.1, 0.1],
            "trade_intensity": [0.1, 0.1, 0.1],
            "vol_proxy": [0.1, 0.1, 0.1],
            "imbalance_for_fill": [0.1, 0.1, 0.1],
        },
        "queue_competition_strength": 0.0,
        "adverse_toxicity_strength": 0.0,
        "latency_touch_penalty_per_bar": 0.08,
    }


def test_higher_latency_reduces_fill_rate_and_increases_offset() -> None:
    features = {"spread": 0.00005, "trade_intensity": 2500.0, "vol_proxy": 0.00003, "imbalance_for_fill": 0.55}
    base = _base_params()
    low = {
        **base,
        "latency_enabled": True,
        "latency_profile": "fixed",
        "latency_send_ms": 5.0,
        "latency_exchange_recv_ms": 5.0,
        "latency_book_effective_ms": 5.0,
        "latency_ack_ms": 5.0,
        "latency_fill_ms": 5.0,
        "latency_bucket_sec": 1.0,
    }
    high = {
        **base,
        "latency_enabled": True,
        "latency_profile": "fixed",
        "latency_send_ms": 2000.0,
        "latency_exchange_recv_ms": 1500.0,
        "latency_book_effective_ms": 1000.0,
        "latency_ack_ms": 1000.0,
        "latency_fill_ms": 1000.0,
        "latency_bucket_sec": 1.0,
    }

    n = 120
    low_filled = 0
    high_filled = 0
    low_offsets: list[int] = []
    high_offsets: list[int] = []
    for i in range(n):
        event = {
            "event_id": f"evt_{i}",
            "symbol": "ETHUSDT",
            "side": "LONG",
            "entry_price": 100.0,
            "future_mids": [99.9, 99.8, 99.75, 99.7, 99.65, 99.6, 99.55, 99.5, 99.45, 99.4, 99.35, 99.3],
            "bucket_sec": 1.0,
            "decision_ts_ms": 1_772_000_000_000 + i,
        }
        out_low = simulate_passive_fill(event, horizon_sec=60, features=features, params=low)
        out_high = simulate_passive_fill(event, horizon_sec=60, features=features, params=high)
        if bool(out_low.get("filled")):
            low_filled += 1
            low_offsets.append(int(out_low.get("fill_index_offset", 0) or 0))
        if bool(out_high.get("filled")):
            high_filled += 1
            high_offsets.append(int(out_high.get("fill_index_offset", 0) or 0))

    assert low_filled >= high_filled
    if low_offsets and high_offsets:
        assert (sum(high_offsets) / len(high_offsets)) >= (sum(low_offsets) / len(low_offsets))

