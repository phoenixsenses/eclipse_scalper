from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.microstructure import (
    build_collector_schema_fixture,
    cleanup_temp_path,
    load_micro_edge_rows,
    make_temp_micro_db,
)
from tools.micro_edge_lib import compute_rule_thresholds, evaluate_naive_rules, signal_aligned_labels


def test_micro_edge_pipeline_from_shared_fixture() -> None:
    db = make_temp_micro_db(prefix="micro_edge_pipeline")
    try:
        build_collector_schema_fixture(
            db,
            symbols=["ETHUSDT"],
            start_ms=1_700_000_000_000,
            rows_per_symbol=80,
            include_true_book=False,
        )
        rows = load_micro_edge_rows(
            db,
            symbol="ETHUSDT",
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_000_000 + 120_000,
            bucket_sec=5,
            vol_window=6,
        )
        mids = [r.get("mid") for r in rows]
        _, labels = signal_aligned_labels(mids, horizon_steps=2, threshold=0.0001)
        baseline = 0.5
        rules = evaluate_naive_rules(rows, labels, baseline_hit_rate=baseline)
        thresholds = compute_rule_thresholds(rows)

        assert rows
        assert len(rows) > 5
        assert any(r.get("trade_intensity") is not None for r in rows)
        assert any(r.get("spread") is not None for r in rows)
        assert any(r.get("liq_rate_per_sec") is not None for r in rows)
        assert rules
        assert "liquidation_spike_reversal" in rules
        assert (rules["liquidation_spike_reversal"].get("n") or 0) > 0
        assert "imb_q90" in thresholds
        assert "int_q90" in thresholds
        assert "liq_rate_q90" in thresholds
    finally:
        cleanup_temp_path(db)
