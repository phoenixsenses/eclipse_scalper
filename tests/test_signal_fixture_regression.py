from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.microstructure import (
    build_collector_schema_fixture,
    cleanup_temp_path,
    load_micro_edge_rows,
    make_temp_micro_db,
)
from tools.micro_edge_lib import signal_aligned_forward_returns, signal_aligned_labels
from tools.micro_edge_signal_v2 import enrich_rows_with_v2


def test_signal_fixture_v2_determinism() -> None:
    db = make_temp_micro_db(prefix="signal_regression")
    try:
        start_ms = 1_700_000_000_000
        build_collector_schema_fixture(db, symbols=["ETHUSDT"], start_ms=start_ms, rows_per_symbol=120)
        rows = load_micro_edge_rows(
            db,
            symbol="ETHUSDT",
            start_ms=start_ms,
            end_ms=start_ms + 180_000,
            bucket_sec=5,
            vol_window=6,
        )
        a = enrich_rows_with_v2(rows, bucket_sec=5, cache_key=("fixture", "ETHUSDT", 120, 5, "micro_edge_v2_passive_alpha"))
        b = enrich_rows_with_v2(rows, bucket_sec=5, cache_key=("fixture", "ETHUSDT", 120, 5, "micro_edge_v2_passive_alpha"))
        assert len(a) == len(b)
        for i in (2, 5, 10, min(len(a) - 1, 15)):
            assert float(a[i]["v2_score"]) == float(b[i]["v2_score"])
            assert float(a[i]["v2_confidence"]) == float(b[i]["v2_confidence"])
            assert float(a[i]["v3_score"]) == float(b[i]["v3_score"])
            assert float(a[i]["v3_confidence"]) == float(b[i]["v3_confidence"])
    finally:
        cleanup_temp_path(db)


def test_signal_fixture_no_lookahead() -> None:
    db = make_temp_micro_db(prefix="signal_no_lookahead")
    try:
        start_ms = 1_700_000_000_000
        build_collector_schema_fixture(db, symbols=["ETHUSDT"], start_ms=start_ms, rows_per_symbol=140)
        rows = load_micro_edge_rows(
            db,
            symbol="ETHUSDT",
            start_ms=start_ms,
            end_ms=start_ms + 200_000,
            bucket_sec=5,
            vol_window=6,
        )
        base = enrich_rows_with_v2(rows, bucket_sec=5, cache_key=None)

        changed = copy.deepcopy(rows)
        mutate_from = max(1, len(changed) - 6)
        for i in range(mutate_from, len(changed)):
            changed[i]["spread"] = 0.05
            changed[i]["trade_intensity"] = 999999.0
            changed[i]["ret_1"] = -0.05 if i % 2 else 0.05
            changed[i]["imbalance"] = -0.99 if i % 2 else 0.99
        mod = enrich_rows_with_v2(changed, bucket_sec=5, cache_key=None)

        stable_until = max(0, mutate_from - 2)
        for i in range(0, stable_until):
            assert float(base[i]["v2_score"]) == float(mod[i]["v2_score"])
            assert float(base[i]["v2_confidence"]) == float(mod[i]["v2_confidence"])
            assert float(base[i]["v3_score"]) == float(mod[i]["v3_score"])
            assert float(base[i]["v3_confidence"]) == float(mod[i]["v3_confidence"])

        mids = [r.get("mid") for r in rows]
        base_rets = signal_aligned_forward_returns(mids, horizon_steps=2)
        _, base_labels = signal_aligned_labels(mids, horizon_steps=2, threshold=0.0001)
        changed_mids = list(mids)
        for i in range(max(0, len(changed_mids) - 5), len(changed_mids)):
            if changed_mids[i] is not None:
                changed_mids[i] = float(changed_mids[i]) * 10.0
        mod_rets = signal_aligned_forward_returns(changed_mids, horizon_steps=2)
        _, mod_labels = signal_aligned_labels(changed_mids, horizon_steps=2, threshold=0.0001)
        compare_until = max(0, len(changed_mids) - 8)
        for i in range(compare_until):
            assert base_rets[i] == mod_rets[i]
            assert base_labels[i] == mod_labels[i]
    finally:
        cleanup_temp_path(db)
