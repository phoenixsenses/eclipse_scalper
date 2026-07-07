"""BATCH-SPOT-PERP-BASIS-READINESS-AND-CONTRACT-V1 -- focused, outcome-blind
validation of ami/research/spot_perp_basis_readiness_audit.py.

No outcome table is ever opened. No experiment, result, nullifier, or gate
receipt is created by this file or the module it tests.

Run: pytest tests/test_ami_research_spot_perp_basis_readiness_audit.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import inspect
import sqlite3

import pytest

from ami.research import spot_perp_basis_readiness_audit as audit

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_MICROSTRUCTURE_PATH = "D:/eclipse_scalper/data/microstructure.db"


def _ro_canonical():
    return sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)


def _ro_micro():
    return sqlite3.connect(f"file:{REAL_MICROSTRUCTURE_PATH}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# 1. Timestamp alignment / no post-birth joins (pure function, synthetic data)
# ---------------------------------------------------------------------------

def test_nearest_at_or_before_never_returns_future_timestamp():
    sorted_ts = [100, 200, 300, 400]
    assert audit.nearest_at_or_before(sorted_ts, 250) == 200
    assert audit.nearest_at_or_before(sorted_ts, 400) == 400
    assert audit.nearest_at_or_before(sorted_ts, 100) == 100


def test_nearest_at_or_before_returns_none_when_target_precedes_all_samples():
    sorted_ts = [100, 200, 300]
    assert audit.nearest_at_or_before(sorted_ts, 50) is None


def test_nearest_at_or_before_exact_boundary_inclusive():
    """target_ts_ms == a sample's own timestamp must select that sample
    (inclusive lower bound), never the one after it."""
    sorted_ts = [1000, 2000, 3000]
    assert audit.nearest_at_or_before(sorted_ts, 2000) == 2000


def test_classify_signal_spot_coverage_absent_before_collection():
    result = audit.classify_signal_spot_coverage(50, [100, 200, 300])
    assert result["quality_status"] == audit.QUALITY_ABSENT
    assert result["nearest_spot_ts_ms"] is None
    assert result["staleness_ms"] is None


def test_classify_signal_spot_coverage_fresh_within_healthy_age():
    result = audit.classify_signal_spot_coverage(
        1_000_000, [999_000], healthy_age_ms=10 * 60 * 1000)
    assert result["quality_status"] == audit.QUALITY_FRESH
    assert result["staleness_ms"] == 1000


def test_classify_signal_spot_coverage_stale_beyond_healthy_age():
    healthy_age_ms = 10 * 60 * 1000
    birth = 100_000_000
    stale_sample = birth - healthy_age_ms - 1
    result = audit.classify_signal_spot_coverage(birth, [stale_sample], healthy_age_ms=healthy_age_ms)
    assert result["quality_status"] == audit.QUALITY_STALE
    assert result["staleness_ms"] == healthy_age_ms + 1


def test_classify_signal_spot_coverage_boundary_exactly_at_healthy_age_is_fresh():
    healthy_age_ms = 10 * 60 * 1000
    birth = 100_000_000
    boundary_sample = birth - healthy_age_ms
    result = audit.classify_signal_spot_coverage(birth, [boundary_sample], healthy_age_ms=healthy_age_ms)
    assert result["quality_status"] == audit.QUALITY_FRESH
    assert result["staleness_ms"] == healthy_age_ms


# ---------------------------------------------------------------------------
# 2. Staleness enforcement / exclusion determinism (real data, read-only)
# ---------------------------------------------------------------------------

def test_anchor_accounting_reconciles_exactly():
    canon = _ro_canonical()
    micro = _ro_micro()
    try:
        result = audit.anchor_accounting(canon, micro)
    finally:
        canon.close()
        micro.close()
    assert result["reconciliation_ok"] is True
    assert result["total_anchors"] == 324
    assert sum(result["spot_quality_breakdown"].values()) == 324
    assert set(result["spot_quality_breakdown"]) <= {
        audit.QUALITY_ABSENT, audit.QUALITY_STALE, audit.QUALITY_FRESH}


def test_anchor_accounting_idempotent_across_two_independent_runs():
    """Same real, read-only source -> byte-identical accounting on a
    second, independent call -- no hidden mutable state, no randomness."""
    canon1, micro1 = _ro_canonical(), _ro_micro()
    canon2, micro2 = _ro_canonical(), _ro_micro()
    try:
        r1 = audit.anchor_accounting(canon1, micro1)
        r2 = audit.anchor_accounting(canon2, micro2)
    finally:
        canon1.close(); micro1.close()
        canon2.close(); micro2.close()
    assert r1["spot_quality_breakdown"] == r2["spot_quality_breakdown"]
    assert r1["fresh_spot_rows"] == r2["fresh_spot_rows"]
    assert r1["fresh_spot_independent_cycles"] == r2["fresh_spot_independent_cycles"]
    assert [r["signal_id"] for r in r1["rows"]] == [r["signal_id"] for r in r2["rows"]]
    assert [r["spot_quality_status"] for r in r1["rows"]] == [r["spot_quality_status"] for r in r2["rows"]]


def test_mark_prices_coverage_near_total_and_low_staleness():
    """Sanity re-check of the already-established mark_prices known-at
    cleanliness (CVD/absorption precedent) in this new context."""
    canon = _ro_canonical()
    micro = _ro_micro()
    try:
        result = audit.anchor_accounting(canon, micro)
    finally:
        canon.close()
        micro.close()
    assert result["mark_absent_rows"] == 0
    assert result["mark_stale_over_10s_rows"] == 0


# ---------------------------------------------------------------------------
# 3. Symbol/venue identity
# ---------------------------------------------------------------------------

def test_source_coverage_summary_scoped_to_single_symbol():
    micro = _ro_micro()
    try:
        spot = audit.source_coverage_summary(micro, "spot_prices", symbol="ETHUSDT")
        mark = audit.source_coverage_summary(micro, "mark_prices", symbol="ETHUSDT")
    finally:
        micro.close()
    assert spot["symbol"] == "ETHUSDT"
    assert mark["symbol"] == "ETHUSDT"
    assert spot["row_count"] > 0
    assert mark["row_count"] > 0


def test_fetch_sorted_timestamps_rejects_unexpected_table():
    micro = _ro_micro()
    try:
        with pytest.raises(ValueError):
            audit.fetch_sorted_timestamps(micro, "agg_trades")
    finally:
        micro.close()


def test_anchor_universe_scoped_to_ethusdt_only():
    canon = _ro_canonical()
    try:
        signals = audit.fetch_anchor_universe(canon)
    finally:
        canon.close()
    assert len(signals) == 324
    # ami_signal_lifecycle carries no symbol column of its own -- confirmed
    # single-symbol population via the canonical row count matching the
    # already-established 324/220/104 identity, not assumed here silently
    directions = {s["direction"] for s in signals}
    assert directions == {"LONG", "SHORT"}


# ---------------------------------------------------------------------------
# 4. Exact/proxy separation
# ---------------------------------------------------------------------------

def test_no_proxy_status_in_quality_taxonomy():
    """This readiness audit's taxonomy has no PROXY tier -- spot_prices is
    used as-is (Binance spot REST ticker) or not at all; no book-depth or
    other proxy source is substituted anywhere in this module."""
    assert "PROXY" not in audit.QUALITY_ABSENT
    assert "PROXY" not in audit.QUALITY_STALE
    assert "PROXY" not in audit.QUALITY_FRESH
    src = inspect.getsource(audit)
    assert "book_ticker" not in src


def test_module_never_executes_sql_naming_the_outcome_table():
    """Static guard: no SQL string ever passed to `.execute()` (or
    `.executescript()`/`.executemany()`) anywhere in this module may name
    the outcome table or its columns. Deliberately narrower than a blunt
    substring scan of the whole module source -- this module's own
    docstrings legitimately *name* `ami_lifecycle_path_observations` in
    prose (explaining that it is never opened), which a naive substring
    check would misclassify as a violation (same false-positive class
    already found and fixed in the absorption-impact rehearsal batch,
    fc43e972). AST-parses the module, walks every `Call` node whose method
    name is an execute-family method, and checks only its string-literal
    arguments."""
    import ast

    tree = ast.parse(inspect.getsource(audit))
    forbidden = ("ami_lifecycle_path_observations", "endpoint_return_bps", "mfe_bps", "mae_bps")
    execute_methods = {"execute", "executescript", "executemany"}
    sql_literals_found = 0
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in execute_methods:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    sql_literals_found += 1
                    if any(term in arg.value for term in forbidden):
                        violations.append(arg.value)
    assert sql_literals_found > 0, "sanity check: the scan found no SQL literals at all -- scan is broken"
    assert violations == []


# ---------------------------------------------------------------------------
# 5. Known-at proof / duplicate-cycle prevention
# ---------------------------------------------------------------------------

def test_verify_no_lookahead_clean_on_real_accounting():
    canon = _ro_canonical()
    micro = _ro_micro()
    try:
        result = audit.anchor_accounting(canon, micro)
    finally:
        canon.close()
        micro.close()
    kav = audit.verify_no_lookahead(result["rows"])
    assert kav["known_at_violations"] == 0
    assert kav["violations"] == []


def test_verify_no_lookahead_detects_synthetic_violation():
    rows = [{"signal_id": "SIG-X", "spot_staleness_ms": -5, "mark_staleness_ms": None}]
    kav = audit.verify_no_lookahead(rows)
    assert kav["known_at_violations"] == 1


def test_verify_duplicate_cycle_free_collapses_multi_signal_cycles():
    rows = [
        {"signal_id": "A", "independent_cycle_id": "CYC-1", "source_event_id": "E-1",
         "signal_birth_ts": 200, "spot_quality_status": audit.QUALITY_FRESH},
        {"signal_id": "B", "independent_cycle_id": "CYC-1", "source_event_id": "E-1",
         "signal_birth_ts": 100, "spot_quality_status": audit.QUALITY_FRESH},
        {"signal_id": "C", "independent_cycle_id": "CYC-2", "source_event_id": "E-2",
         "signal_birth_ts": 150, "spot_quality_status": audit.QUALITY_FRESH},
        {"signal_id": "D", "independent_cycle_id": "CYC-3", "source_event_id": "E-3",
         "signal_birth_ts": 300, "spot_quality_status": audit.QUALITY_STALE},
    ]
    result = audit.verify_duplicate_cycle_free(rows)
    assert result["fresh_signal_count"] == 3
    assert result["representative_cycle_count"] == 2  # CYC-1 collapses to signal B (earliest)
    assert result["duplicates_collapsed"] == 1


def test_verify_duplicate_cycle_free_on_real_accounting_matches_module_output():
    canon = _ro_canonical()
    micro = _ro_micro()
    try:
        result = audit.anchor_accounting(canon, micro)
    finally:
        canon.close()
        micro.close()
    dup = audit.verify_duplicate_cycle_free(result["rows"])
    assert dup["representative_cycle_count"] == result["fresh_spot_independent_cycles"]
    assert dup["fresh_signal_count"] == result["fresh_spot_rows"]


# ---------------------------------------------------------------------------
# 6. Gap-statistics sanity (informational, no outcome data)
# ---------------------------------------------------------------------------

def test_inter_sample_gap_stats_on_real_spot_prices():
    micro = _ro_micro()
    try:
        spot_ts = audit.fetch_sorted_timestamps(micro, "spot_prices")
    finally:
        micro.close()
    stats = audit.inter_sample_gap_stats(spot_ts)
    assert stats["n_samples"] == len(spot_ts)
    assert stats["min_gap_ms"] >= 0
    assert stats["max_gap_ms"] >= stats["median_gap_ms"] >= stats["min_gap_ms"]
    # the known ~27-day collector outage (2026-06-05 -> 2026-07-02) must
    # show up as the single largest gap, not silently smoothed away
    assert stats["max_gap_ms"] > 20 * 24 * 3600 * 1000


def test_inter_sample_gap_stats_handles_empty_and_singleton():
    assert audit.inter_sample_gap_stats([])["n_gaps"] == 0
    assert audit.inter_sample_gap_stats([100])["n_gaps"] == 0
