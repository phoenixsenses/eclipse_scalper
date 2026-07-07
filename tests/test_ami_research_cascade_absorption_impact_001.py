"""BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1 -- pre-execution validation.

Three groups:
1. Pure statistics (synthetic data, no DB) -- OLS/cluster-robust SE/VIF/
   collinearity policy/verdict rule/listwise deletion/rank-check.
2. Real-data (mode=ro via conftest session isolation) population/split/
   identity reproduction -- proves the module's TEST-outcome-blind functions
   reproduce the frozen preregistration exactly, without ever reading
   endpoint_return_bps/mfe_bps.
3. Disposable-copy integration (conftest session isolation) -- a full dress
   rehearsal of `execute_governed_run` against copies of the real canonical/
   knowledge databases: proves nullifier consumption, registry/results
   write, gate-receipt reissue, and idempotent rerun, without ever touching
   the real files.

Run: pytest tests/test_ami_research_cascade_absorption_impact_001.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

import ami.research.cascade_absorption_impact_001 as m

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_KNOWLEDGE_PATH = "D:/eclipse_scalper/data/ami/knowledge.sqlite"


# ---------------------------------------------------------------------------
# 1. Pure statistics
# ---------------------------------------------------------------------------

def test_ols_exact_recovery_no_noise():
    rng = np.random.default_rng(7)
    n = 60
    X = np.ones((n, 6))
    X[:, 1] = rng.normal(0, 1, n)
    X[:, 2] = rng.normal(0, 1, n)
    X[:, 3] = rng.integers(0, 2, n)
    X[:, 4] = rng.integers(0, 2, n)
    X[:, 5] = rng.normal(0, 1, n)
    true_beta = np.array([1.5, 3.0, -2.0, 0.5, -0.5, 0.75])
    y = X @ true_beta  # no noise
    cluster_ids = [f"c{i}" for i in range(n)]  # G == n, singleton clusters
    fit = m.run_cluster_robust_ols(X, y, cluster_ids)
    assert np.allclose(fit["beta"], true_beta, atol=1e-8)
    assert fit["G"] == n
    assert np.all(fit["se"] >= 0)


def test_ols_cluster_grouping_reduces_effective_g():
    rng = np.random.default_rng(11)
    n = 40
    X = np.ones((n, 2))
    X[:, 1] = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * X[:, 1] + rng.normal(0, 0.1, n)
    cluster_ids = [f"cluster{i // 4}" for i in range(n)]
    fit = m.run_cluster_robust_ols(X, y, cluster_ids)
    assert fit["G"] == 10
    assert fit["n"] == 40
    assert fit["df"] == 9


def test_check_design_rank_full_rank():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(50), rng.normal(0, 1, 50), rng.normal(0, 1, 50)])
    rank = m.check_design_rank(X)
    assert rank["full_rank"] is True
    assert rank["rank"] == 3


def test_check_design_rank_deficient_when_column_duplicated():
    rng = np.random.default_rng(5)
    x1 = rng.normal(0, 1, 50)
    X = np.column_stack([np.ones(50), x1, x1])  # exact duplicate -> rank deficient
    rank = m.check_design_rank(X)
    assert rank["full_rank"] is False
    assert rank["rank"] < rank["k"]


def test_build_design_raises_on_unexpected_europe_session():
    """Frozen policy: EUROPE must never appear (0 TRAIN/TEST observations).
    If it somehow does, build_design must fail closed, not silently encode it."""
    rows = [
        {"cycle_key": "a", "signal_id": "SIG-TEST", "price_response_w300": -5.0, "event_notional": 100_000.0,
         "session": "EUROPE", "day_trend_bps": 1.0, "endpoint_return_bps": 5.0, "missing_predictor": False},
    ]
    with pytest.raises(m.ProtocolInvalidation):
        m.build_design(rows, "endpoint_return_bps")


def test_vif_high_for_collinear_columns_low_for_independent():
    rng = np.random.default_rng(3)
    n = 100
    x1 = rng.normal(0, 1, n)
    x_collinear = x1 * 2.0 + rng.normal(0, 0.001, n)
    x_independent = rng.normal(0, 1, n)
    # build a 6-column design matching DESIGN_NAMES layout: const, x1(predictor),
    # x_collinear(event_notional slot), session_US, session_OFF, x_independent(day_trend slot)
    X = np.column_stack([np.ones(n), x1, x_collinear,
                          rng.integers(0, 2, n), rng.integers(0, 2, n), x_independent])
    vifs = m.compute_vif(X)
    assert vifs["price_response_w300"] > 10
    assert vifs["event_notional_per_100k"] > 10
    assert vifs["day_trend_bps"] < 5


def test_collinearity_policy_drop_order_and_session_never_dropped():
    vifs_both_high = {"day_trend_bps": 15.0, "event_notional_per_100k": 12.0}
    drops = m.apply_collinearity_policy(vifs_both_high)
    assert drops == ["day_trend_bps", "event_notional_per_100k"]

    vifs_only_event = {"day_trend_bps": 2.0, "event_notional_per_100k": 11.0}
    assert m.apply_collinearity_policy(vifs_only_event) == ["event_notional_per_100k"]

    vifs_none_high = {"day_trend_bps": 2.0, "event_notional_per_100k": 3.0}
    assert m.apply_collinearity_policy(vifs_none_high) == []


def test_verdict_rule_underpowered_small_n():
    verdict, _ = m.apply_verdict_rule(n_test=10, coef=1.0, se=0.1, ci_lo=0.8, ci_hi=1.2, p_value=0.001)
    assert verdict == m.VERDICT_UNDERPOWERED


def test_verdict_rule_ci_too_wide_is_underpowered():
    verdict, _ = m.apply_verdict_rule(n_test=40, coef=1.0, se=50.0, ci_lo=-98.0, ci_hi=100.0, p_value=0.9)
    assert verdict == m.VERDICT_UNDERPOWERED


def test_verdict_rule_supports_incremental_association():
    # coef chosen so |coef * PREDICTOR_TRAIN_STDEV| >= 5: stdev~10.7, coef=1.0 -> ~10.7 >= 5
    verdict, reason = m.apply_verdict_rule(n_test=40, coef=1.0, se=0.3, ci_lo=0.4, ci_hi=1.6, p_value=0.001)
    assert verdict == m.VERDICT_SUPPORTS
    assert "CI excludes 0" in reason


def test_verdict_rule_significant_but_below_relevance_floor_is_no_reliable_association():
    # tiny coefficient: |0.1 * 10.7| ~= 1.07 < floor(5), even though CI excludes 0 and p<0.05
    verdict, _ = m.apply_verdict_rule(n_test=40, coef=0.1, se=0.01, ci_lo=0.08, ci_hi=0.12, p_value=0.0001)
    assert verdict == m.VERDICT_NO_RELIABLE


def test_verdict_rule_ci_includes_zero_is_no_reliable_association():
    verdict, _ = m.apply_verdict_rule(n_test=40, coef=1.0, se=1.0, ci_lo=-1.0, ci_hi=3.0, p_value=0.3)
    assert verdict == m.VERDICT_NO_RELIABLE


def test_build_design_applies_listwise_deletion():
    rows = [
        {"cycle_key": "a", "price_response_w300": -5.0, "event_notional": 100_000.0,
         "session": "ASIA", "day_trend_bps": 1.0, "endpoint_return_bps": 5.0, "missing_predictor": False},
        {"cycle_key": "b", "price_response_w300": None, "event_notional": 100_000.0,
         "session": "US", "day_trend_bps": 1.0, "endpoint_return_bps": 5.0, "missing_predictor": True},
        {"cycle_key": "c", "price_response_w300": -2.0, "event_notional": 100_000.0,
         "session": "OFF", "day_trend_bps": 1.0, "endpoint_return_bps": None, "missing_predictor": False},
    ]
    design = m.build_design(rows, "endpoint_return_bps")
    assert design["n"] == 1
    assert design["n_total"] == 3
    assert design["n_dropped"] == 2
    assert design["cluster_ids"] == ["a"]


def test_design_names_exclude_europe():
    assert "session_EUROPE" not in m.DESIGN_NAMES
    assert m.DESIGN_NAMES == ("const", "price_response_w300", "event_notional_per_100k",
                               "session_US", "session_OFF", "day_trend_bps")


# ---------------------------------------------------------------------------
# 2. Population/split reproduction, TEST-outcome-blind. Uses the conftest
#    session-isolated DISPOSABLE copy (writable) rather than a raw mode=ro
#    real-file connection: `resolve_population()` calls
#    `feature_gateway.fetch_lifecycle_signals()`, which appends an (expected,
#    by-design) `researcher_exposure_ledger` audit row on every call -- that
#    write must never land on the real file from a test.
# ---------------------------------------------------------------------------

def test_resolve_population_matches_frozen_preregistration():
    import ami.warehouse.schema as schema_mod

    conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    try:
        pop = m.resolve_population(conn)
    finally:
        conn.close()
    assert pop["long_n"] == 220
    assert pop["eligible_long_n"] == 194
    assert pop["representative_cycle_n"] == 131
    assert len(pop["train_reps"]) == 91
    assert len(pop["test_reps"]) == 40
    assert pop["train_hash"] == m.EXPECTED_TRAIN_HASH
    assert pop["test_hash"] == m.EXPECTED_TEST_HASH


def test_verify_pre_execution_reports_zero_errors_against_real_db():
    import ami.knowledge.store as knowledge_mod
    import ami.warehouse.schema as schema_mod

    canonical_conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    knowledge_conn = sqlite3.connect(str(knowledge_mod.DEFAULT_PATH))
    try:
        result = m.verify_pre_execution(canonical_conn, knowledge_conn)
    finally:
        canonical_conn.close()
        knowledge_conn.close()
    assert result["errors"] == []
    assert result["family_id"] == m.FAMILY_ID
    assert result["nullifier"] == m.EXPECTED_NULLIFIER
    assert result["is_rerun_of_self"] is False  # not yet executed against the real DB
    assert result["already_has_results_before"] == 0
    assert result["schema_version"] == m.EXPECTED_SCHEMA_VERSION


def test_verify_pre_execution_never_selects_outcome_columns():
    """Static guard: resolve_population's own SQL text must never mention
    the outcome columns -- the TEST-outcome-blind boundary is enforced at
    the query level, not just by unused-variable discipline."""
    import inspect
    src = inspect.getsource(m.resolve_population)
    assert "endpoint_return_bps" not in src
    assert "mfe_bps" not in src


# ---------------------------------------------------------------------------
# 3. Disposable-copy integration (conftest session isolation) -- full dress
#    rehearsal of the governed execution against copies of the real data
# ---------------------------------------------------------------------------

def test_execute_governed_run_blocks_on_identity_mismatch(monkeypatch):
    """Must run BEFORE the dress rehearsal below -- both share the same
    session-scoped disposable knowledge.sqlite copy, and this test's own
    assertion (nullifier still unconsumed) would be false once the
    rehearsal has legitimately consumed it."""
    import ami.knowledge.store as knowledge_mod
    import ami.warehouse.schema as schema_mod

    canonical_conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    knowledge_conn = sqlite3.connect(str(knowledge_mod.DEFAULT_PATH))
    try:
        monkeypatch.setattr(m, "FAMILY_ID", "FAMv1:0000000000000000")
        with pytest.raises(m.ProtocolInvalidation):
            m.execute_governed_run(canonical_conn, knowledge_conn)
        n_rows = knowledge_conn.execute(
            "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
            (m.EXPECTED_NULLIFIER,)).fetchone()[0]
        assert n_rows == 0
    finally:
        canonical_conn.close()
        knowledge_conn.close()


def test_governed_execution_dress_rehearsal_on_disposable_copies():
    import ami.knowledge.store as knowledge_mod
    import ami.warehouse.schema as schema_mod

    canonical_conn = schema_mod.connect(schema_mod.DEFAULT_PATH)
    knowledge_conn = sqlite3.connect(str(knowledge_mod.DEFAULT_PATH))
    try:
        pre_verify = m.verify_pre_execution(canonical_conn, knowledge_conn)
        assert pre_verify["errors"] == []
        assert pre_verify["is_rerun_of_self"] is False

        pre_reg_n = canonical_conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
        pre_events_n = canonical_conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
        pre_signals_n = canonical_conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_absorption_n = canonical_conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow").fetchone()[0]

        r1 = m.execute_governed_run(canonical_conn, knowledge_conn)

        assert r1["consume_result"] == "CONSUMED"
        assert r1["nullifier"] == m.EXPECTED_NULLIFIER
        assert r1["registry_result"] == "INSERTED"
        assert r1["results_result"] == "INSERTED"
        assert r1["test_n"] >= 20
        assert r1["train_n"] >= 20
        assert r1["test_design_rank"]["full_rank"] is True
        assert r1["train_design_rank"]["full_rank"] is True
        assert r1["verdict"] in (
            m.VERDICT_SUPPORTS, m.VERDICT_NO_RELIABLE, m.VERDICT_UNDERPOWERED, m.VERDICT_INVALIDATED,
        )

        n_rows = knowledge_conn.execute(
            "SELECT consumed_by_experiment_id FROM epistemic_test_nullifiers WHERE nullifier=?",
            (m.EXPECTED_NULLIFIER,)).fetchall()
        assert n_rows == [(m.EXPERIMENT_ID,)]

        assert canonical_conn.execute(
            "SELECT COUNT(*) FROM experiment_registry").fetchone()[0] == pre_reg_n + 1
        n_results_1 = canonical_conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (m.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_1 > 0
        assert canonical_conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0] == pre_events_n
        assert canonical_conn.execute(
            "SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_signals_n
        assert canonical_conn.execute(
            "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow").fetchone()[0] == pre_absorption_n

        # idempotent rerun: NOOP everywhere, no duplicate rows, same verdict
        r2 = m.execute_governed_run(canonical_conn, knowledge_conn)
        assert r2["consume_result"] == "NOOP_IDENTICAL"
        assert r2["registry_result"] == "NOOP_IDENTICAL"
        assert r2["results_result"] == "NOOP_IDENTICAL"
        assert r2["verdict"] == r1["verdict"]
        n_results_2 = canonical_conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (m.EXPERIMENT_ID,)
        ).fetchone()[0]
        assert n_results_2 == n_results_1
        assert canonical_conn.execute(
            "SELECT COUNT(*) FROM experiment_registry").fetchone()[0] == pre_reg_n + 1
        n_rows_again = knowledge_conn.execute(
            "SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?",
            (m.EXPECTED_NULLIFIER,)).fetchone()[0]
        assert n_rows_again == 1
    finally:
        canonical_conn.close()
        knowledge_conn.close()
