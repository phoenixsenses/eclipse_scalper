"""BATCH-P6-009: W6-RS confound-resolution wave tests.

Run: pytest tests/test_ami_research_w6rs_confound_resolution.py --basetemp <scratchpad> -p no:cacheprovider
"""
import numpy as np

from ami.research.w6rs_confound_resolution import (
    EXPERIMENT_ID,
    _build_design_matrix,
    _cluster_bootstrap_interaction,
    _fit_logistic_irls,
    _overlap_positivity_checks,
    _prepare_rows_and_design,
    classify_result,
    freeze_and_record,
)


def test_irls_recovers_strong_separating_predictor():
    rng = np.random.default_rng(42)
    n = 400
    x1 = rng.normal(size=n)
    logits = 3.0 * x1  # strong, known relationship
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([np.ones(n), x1])
    beta = _fit_logistic_irls(X, y)
    assert beta[1] > 1.5  # recovers a clearly positive, sizeable coefficient


def test_irls_null_predictor_near_zero_coefficient():
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.5).astype(float)  # no real relationship
    X = np.column_stack([np.ones(n), x1])
    beta = _fit_logistic_irls(X, y)
    assert abs(beta[1]) < 0.5  # no strong spurious coefficient on pure noise


def _row(anchor_status, rs_group, path_class, day_trend, session, month, cluster_id):
    return {"anchor_status": anchor_status, "rs_group": rs_group, "path_class": path_class,
            "day_trend_bps": day_trend, "session": session, "month": month, "cluster_id": cluster_id}


def test_build_design_matrix_columns_and_shape():
    rows = [
        _row("ANCHOR", "RS_STRONG", "REVERSAL", 10.0, "US", "2026-03", "C1"),
        _row("ANCHOR", "RS_WEAK", "CONTINUATION", -5.0, "ASIA", "2026-04", "C2"),
        _row("CONTROL", "RS_STRONG", "CHOP", 0.0, "OFF", "2026-03", "CTRL-1"),
    ]
    X, y, names, meta = _prepare_rows_and_design(rows)
    assert X.shape[0] == 3
    assert names[0] == "intercept"
    assert "interaction" in names
    assert y.tolist() == [1.0, 0.0, 0.0]


def test_classify_result_general_regime_when_ci_includes_zero():
    assert classify_result((-0.1, 0.2), june_direction_matches_overall=True) == "GENERAL_REGIME_FEATURE"


def test_classify_result_cascade_specific_when_ci_excludes_zero_and_holdout_stable():
    assert classify_result((0.3, 0.9), june_direction_matches_overall=True) == \
        "CASCADE_SPECIFIC_CANDIDATE_FOR_ECONOMIC_VALIDATION"


def test_classify_result_regime_dependent_when_june_anomaly_unexplained():
    assert classify_result((0.3, 0.9), june_direction_matches_overall=False) == \
        "REGIME_DEPENDENT_CONTINUE_ACCUMULATING"


def test_classify_result_insufficient_sample_when_no_bootstrap_draws():
    assert classify_result((None, None), june_direction_matches_overall=True) == "INSUFFICIENT_SAMPLE"


def test_overlap_positivity_flags_thin_cells():
    rows = [_row("ANCHOR", "RS_STRONG", "REVERSAL", 10.0, "EUROPE", "2026-03", "C1")]
    rows += [_row("ANCHOR", "RS_WEAK", "CONTINUATION", -5.0, "US", "2026-04", f"C{i}") for i in range(10)]
    rows += [_row("CONTROL", "RS_STRONG", "CHOP", 0.0, "US", "2026-03", f"CTRL-{i}") for i in range(10)]
    result = _overlap_positivity_checks(rows)
    assert any("EUROPE" in k for k in result["thin_cells_lt5"])


def test_cluster_bootstrap_returns_ci_for_simple_synthetic_data():
    rows = []
    for i in range(30):
        rows.append(_row("ANCHOR", "RS_STRONG", "REVERSAL", 10.0, "US", "2026-03", f"C{i}"))
        rows.append(_row("ANCHOR", "RS_WEAK", "CONTINUATION", -5.0, "US", "2026-03", f"C{i}b"))
        rows.append(_row("CONTROL", "RS_STRONG", "CONTINUATION", 0.0, "US", "2026-03", f"CTRL-{i}"))
        rows.append(_row("CONTROL", "RS_WEAK", "CONTINUATION", 0.0, "US", "2026-03", f"CTRL-{i}b"))
    cluster_ids = [r["cluster_id"] for r in rows]
    X, y, names, _ = _prepare_rows_and_design(rows)
    beta = _fit_logistic_irls(X, y)
    beta_by_name = dict(zip(names, beta))
    result = _cluster_bootstrap_interaction(rows, cluster_ids, beta_by_name, n_boot=200, seed=1)
    assert result["n_valid_draws"] > 0
    lo, hi = result["ci95"]
    assert lo is not None and hi is not None
    assert lo <= hi


def test_compute_metrics_real_data_smoke_and_freeze_idempotent():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        r1 = freeze_and_record(conn)
        r2 = freeze_and_record(conn)
        n_registry = conn.execute(
            "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert r1["anchor_n"] > 0
    assert r1["control_n"] > 0
    assert r1["independent_cycle_n"] > 0
    assert r1["verdict"] in {
        "GENERAL_REGIME_FEATURE", "CASCADE_SPECIFIC_CANDIDATE_FOR_ECONOMIC_VALIDATION",
        "REGIME_DEPENDENT_CONTINUE_ACCUMULATING", "INSUFFICIENT_SAMPLE",
    }
    assert n_registry == 1
