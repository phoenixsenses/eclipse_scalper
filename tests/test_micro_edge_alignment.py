from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_backtest import simulate_rule_trades
from tools.micro_edge_lib import compute_rule_thresholds, evaluate_naive_rules, signal_aligned_labels


def _build_synth_rows(n: int = 120):
    imbs = [0.8 if i % 2 == 0 else -0.8 for i in range(n)]
    mids = [100.0] * n
    mids[1] = 100.0
    for j in range(1, n - 1):
        s = 1.0 if imbs[j - 1] > 0 else -1.0
        mids[j + 1] = mids[j] * (1.0 + 0.001 * s)
    rows = []
    for i in range(n):
        rows.append(
            {
                "ts_ms": float(i * 1000),
                "mid": float(mids[i]),
                "imbalance": float(imbs[i]),
                "trade_intensity": (100.0 if i % 10 == 0 else 1.0),
                "spread": 0.01,
                "ret_1": (None if i == 0 else (mids[i] / mids[i - 1] - 1.0)),
            }
        )
    return rows


def test_smoke_backtest_alignment_intensity_rule():
    rows = _build_synth_rows()
    fwd, labels = signal_aligned_labels([r["mid"] for r in rows], horizon_steps=1, threshold=0.0002)
    lbl_valid = [int(x) for x in labels if x is not None and int(x) != 0]
    up = sum(1 for x in lbl_valid if x > 0)
    baseline = max(up, len(lbl_valid) - up) / len(lbl_valid)
    rules = evaluate_naive_rules(rows, labels, baseline_hit_rate=baseline)
    assert "intensity_spike_imbalance_cont" in rules
    smoke_hit = float(rules["intensity_spike_imbalance_cont"]["hit_rate"])
    assert smoke_hit > 0.95

    thresholds = compute_rule_thresholds(rows)
    sim = simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds=thresholds,
        labels=labels,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=0.0,
        slip_bps=0.0,
    )
    win_rate = float(sim["metrics"]["win_rate"])
    assert win_rate > 0.95
