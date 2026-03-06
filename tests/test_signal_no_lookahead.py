from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import signal_aligned_forward_returns, signal_aligned_labels
from tools.micro_edge_signal_v2 import enrich_rows_with_v2


def _build_rows(n: int = 80):
    rows = []
    for i in range(n):
        rows.append(
            {
                "ts_ms": float(i),
                "mid": 100.0 + i * 0.01,
                "spread": 0.0005 - min(i, 20) * 0.000005,
                "trade_intensity": 1500.0 + (i % 5) * 100.0,
                "imbalance": 0.4 if i % 2 == 0 else -0.3,
                "ret_1": 0.0001 if i % 3 else -0.00008,
                "micro_volatility": 0.001 + (i % 4) * 0.00005,
            }
        )
    return rows


def test_signal_labels_only_depend_on_horizon_window() -> None:
    mids = [100.0 + i * 0.01 for i in range(30)]
    horizon_steps = 3
    base_rets = signal_aligned_forward_returns(mids, horizon_steps=horizon_steps)
    _, base_labels = signal_aligned_labels(mids, horizon_steps=horizon_steps, threshold=0.00001)

    changed = list(mids)
    for i in range(15, len(changed)):
        changed[i] = changed[i] * 10.0

    mod_rets = signal_aligned_forward_returns(changed, horizon_steps=horizon_steps)
    _, mod_labels = signal_aligned_labels(changed, horizon_steps=horizon_steps, threshold=0.00001)

    # For t <= 10, the latest accessed future point is t + 1 + horizon == 14.
    for i in range(0, 11):
        assert base_rets[i] == mod_rets[i]
        assert base_labels[i] == mod_labels[i]


def test_enrich_rows_ignores_far_future_mutations() -> None:
    rows = _build_rows()
    base = enrich_rows_with_v2(rows, bucket_sec=1, cache_key=None)

    changed = copy.deepcopy(rows)
    for i in range(60, len(changed)):
        changed[i]["spread"] = 0.05
        changed[i]["trade_intensity"] = 999999.0
        changed[i]["ret_1"] = -0.05 if i % 2 else 0.05
        changed[i]["imbalance"] = -0.99 if i % 2 else 0.99
    mod = enrich_rows_with_v2(changed, bucket_sec=1, cache_key=None)

    for i in range(0, 50):
        assert float(base[i]["v2_score"]) == float(mod[i]["v2_score"])
        assert float(base[i]["v2_confidence"]) == float(mod[i]["v2_confidence"])
        assert float(base[i]["v3_score"]) == float(mod[i]["v3_score"])
        assert float(base[i]["v3_confidence"]) == float(mod[i]["v3_confidence"])
