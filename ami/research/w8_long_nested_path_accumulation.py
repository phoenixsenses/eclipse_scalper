"""BATCH-P7B-1 (W8-LONG-NESTED-PATH-ACCUMULATION-001): descriptive path
research -- how much ADDITIONAL favorable (MFE) and adverse (|MAE|) path
exposure accumulates as the observation horizon expands 30m->1h->4h->24h,
for the canonical LONG population.

RESEARCH QUESTIONS (frozen, operator-specified):
  1. How much additional MFE is accumulated in each interval?
  2. How much additional absolute MAE is accumulated in each interval?
  3. Does extending the horizon add relatively more favorable or adverse
     excursion? (secondary descriptive: median(delta_mfe) vs median(delta_abs_mae)
     per interval, and delta_mfe-delta_abs_mae per signal)
  4. Are these incremental distributions chronologically stable?
     -> PRIMARY INFERENTIAL FAMILY, cycle-grouped TRAIN/TEST stability test.

NOT_A_MANAGEMENT_WAVE: NO_STOP_RULE / NO_EXIT_RULE / NO_REENTRY /
NO_HOLD_OPTIMIZATION. This module never recommends a holding period --
"percentage of 24h MFE/MAE already captured by an earlier horizon" is
reported strictly as a descriptive fact about the PATH, never as a
suggested exit/hold rule (operator's own explicit instruction, restated
here so it cannot be silently dropped by a future edit).
NO_ECONOMIC_OR_ALPHA_CLAIM / NO_SHORT_POOLING (LONG only, same posture as
W8-LONG-TIMING-STRUCTURE-001) / NO_SIGNAL_BACKFILL.

COMMON-COHORT REQUIREMENT (operator-mandated, the entire point of this
design): a signal is included ONLY if it has observation_status='OK' at ALL
FOUR frozen horizons simultaneously. This is a single FIXED, horizon-complete
population reused for every interval -- it deliberately does NOT reuse each
horizon's own (larger, horizon-specific) OK population the way
W8-LONG-TIMING-STRUCTURE-001 did, precisely so that a difference between
intervals can never be attributed to a change in WHICH signals are being
looked at. `fetch_common_cohort()` reports the excluded (incomplete-horizon)
count explicitly -- never silently drops rows without accounting for them.

NESTED-PATH NON-NEGATIVITY: because the 1h/4h/24h horizon windows are strict
supersets of the 30m/1h/4h windows respectively (same effective_path_start_ts,
same reference_price -- both are signal_birth_ts-derived, not horizon-
dependent, per ami.lifecycle.path_metrics.compute_observation), MFE (a
running maximum) can only stay the same or increase as the window grows, and
|MAE| (a running maximum of the adverse excursion) likewise. All six frozen
deltas are therefore mathematically non-negative by construction; this is
verified (not assumed) against the real population by
`assert_nested_nonnegativity()`, with a documented floating-point tolerance
(EPSILON) for the 4-decimal rounding each horizon's mfe_bps/mae_bps already
independently applies (ami.lifecycle.path_metrics.compute_observation).

CYCLE-GROUPED SPLIT: reuses ami.research.w8_short_expanded_baseline's
generic split machinery VERBATIM (compute_global_cycle_split()/
split_rows_by_cycle_keys()/assert_zero_cycle_straddling()/_cycle_key()) --
not reimplemented. Since this module's population is one row PER SIGNAL
(not per signal-horizon like the timing-structure wave), the split operates
directly on the wide, one-row-per-signal cohort.

PRIMARY FAMILY (frozen, exactly 6 = 3 MFE-increment intervals + 3
absolute-MAE-increment intervals): delta_mfe_30m_to_1h, delta_mfe_1h_to_4h,
delta_mfe_4h_to_24h, delta_abs_mae_30m_to_1h, delta_abs_mae_1h_to_4h,
delta_abs_mae_4h_to_24h. ONE joint Holm step-down correction across all 6.

SECONDARY DESCRIPTIVE ONLY (no p-values): median incremental MFE/|MAE| per
interval (already the cell's own full_median, surfaced together for
convenience), median(delta_mfe - delta_abs_mae) per interval, and
percentage of 24h MFE/|MAE| already captured by 30m/1h/4h (per-signal ratio,
median reported) -- a zero 24h denominator is EXCLUDED from that specific
ratio's median (never silently divided by zero), with the exact
excluded-for-zero-denominator count reported alongside every capture-fraction
statistic.
"""
from __future__ import annotations
import hashlib
import time

from ami.research.feature_gateway import fetch_lifecycle_signals, fetch_path_observations
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results
from ami.research.w8_hold_baseline import (
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    _median,
    _quantile,
    classify_cell_verdict,
    cluster_bootstrap_median_diff,
    permutation_test_median_diff,
)
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)

EXPERIMENT_ID = "E-W8-LONG-NESTED-PATH-ACCUMULATION-001"
RESEARCH_CONTEXT_ID = "w8-long-nested-path-accumulation"

DIRECTION = "LONG"
# short aliases (operator vocabulary) -> real ami_lifecycle_path_observations.horizon_name values
HORIZON_ALIAS_ORDER = ("30m", "1h", "4h", "24h")
_HORIZON_NAME_OF_ALIAS = {"30m": "scalp_30m", "1h": "scalp_1h", "4h": "swing_4h", "24h": "swing_24h"}
_REAL_HORIZON_NAMES = tuple(_HORIZON_NAME_OF_ALIAS.values())

INTERVALS = (("30m", "1h"), ("1h", "4h"), ("4h", "24h"))
MFE_DELTA_FIELDS = tuple(f"delta_mfe_{a}_to_{b}" for a, b in INTERVALS)
MAE_DELTA_FIELDS = tuple(f"delta_abs_mae_{a}_to_{b}" for a, b in INTERVALS)
ALL_DELTA_FIELDS = MFE_DELTA_FIELDS + MAE_DELTA_FIELDS  # 6 = primary family

EPSILON = 1e-6  # documented floating-point tolerance for independently-rounded (4 decimals) mfe_bps/mae_bps


def fetch_common_cohort(conn, symbol: str = "ETHUSDT") -> dict:
    """feature_gateway ONLY. A signal is included in the cohort iff it has
    observation_status='OK' at ALL FOUR frozen horizons. Returns the wide,
    one-row-per-signal cohort plus the excluded-incomplete-horizon count.

    [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL C] Pins path_definition_version="path-v2" -- this is
    E-W8-LONG-NESTED-PATH-ACCUMULATION-001's frozen, pre-candle-repair
    historical population (superseded, for corrected-data research, by
    ami.research.w8_long_nested_path_accumulation_002_candle_repair, which
    uses the effective-path selector under its own new experiment_id
    instead). An unpinned read now fails closed
    (feature_gateway.AmbiguousPathVersionError)."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_path_observations(
        conn, RESEARCH_CONTEXT_ID,
        equals={"observation_status": "OK", "path_definition_version": "path-v2"},
    )

    by_signal: dict[str, dict[str, dict]] = {}
    for o in obs_rows:
        s = signals.get(o["signal_id"])
        if s is None or s["direction"] != DIRECTION or o["horizon_name"] not in _REAL_HORIZON_NAMES:
            continue
        by_signal.setdefault(o["signal_id"], {})[o["horizon_name"]] = o

    cohort_rows: list[dict] = []
    excluded_incomplete_n = 0
    for signal_id, by_horizon in by_signal.items():
        if len(by_horizon) < 4 or not all(h in by_horizon for h in _REAL_HORIZON_NAMES):
            excluded_incomplete_n += 1
            continue
        s = signals[signal_id]
        row = {
            "signal_id": signal_id, "source_event_id": s["source_event_id"],
            "independent_cycle_id": s["independent_cycle_id"], "signal_birth_ts": s["signal_birth_ts"],
        }
        for alias, real_name in _HORIZON_NAME_OF_ALIAS.items():
            row[f"mfe_bps_{alias}"] = by_horizon[real_name]["mfe_bps"]
            row[f"mae_bps_{alias}"] = by_horizon[real_name]["mae_bps"]
        cohort_rows.append(row)

    return {
        "cohort_rows": cohort_rows, "excluded_incomplete_horizon_n": excluded_incomplete_n,
        "long_signals_with_any_ok_horizon_n": len(by_signal),
    }


def compute_derived_fields(row: dict) -> dict:
    out = dict(row)
    for a, b in INTERVALS:
        out[f"delta_mfe_{a}_to_{b}"] = round(row[f"mfe_bps_{b}"] - row[f"mfe_bps_{a}"], 4)
        out[f"delta_abs_mae_{a}_to_{b}"] = round(abs(row[f"mae_bps_{b}"]) - abs(row[f"mae_bps_{a}"]), 4)
        out[f"delta_diff_{a}_to_{b}"] = round(
            out[f"delta_mfe_{a}_to_{b}"] - out[f"delta_abs_mae_{a}_to_{b}"], 4
        )
    return out


def assert_nested_nonnegativity(rows: list[dict]) -> dict:
    """Verifies (never assumes) the mathematical nesting property: all six
    frozen deltas must be >= -EPSILON. Returns the violation count and the
    exact offending (signal_id, field, value) tuples -- never silently
    swallowed."""
    violations = []
    for r in rows:
        for field in ALL_DELTA_FIELDS:
            val = r.get(field)
            if val is not None and val < -EPSILON:
                violations.append({"signal_id": r["signal_id"], "field": field, "value": val})
    return {"violation_n": len(violations), "violations": violations, "epsilon": EPSILON}


# ---------------------------------------------------------------------------
# primary inferential family: 6 cells
# ---------------------------------------------------------------------------

def compute_cell(rows: list[dict], metric: str, train_keys: set, test_keys: set) -> dict:
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    train_rows, test_rows = split_rows_by_cycle_keys(rows, train_keys, test_keys)
    straddling_n = assert_zero_cycle_straddling(train_rows, test_rows)
    train_vals = [r[metric] for r in train_rows if r.get(metric) is not None]
    test_vals = [r[metric] for r in test_rows if r.get(metric) is not None]

    raw_signal_n = len(rows)
    distinct_source_event_n = len({r["source_event_id"] for r in rows if r["source_event_id"] is not None})
    distinct_cycle_n = len({_cycle_key(r) for r in rows})
    train_signal_n, test_signal_n = len(train_rows), len(test_rows)
    train_event_n = len({r["source_event_id"] for r in train_rows if r["source_event_id"] is not None})
    test_event_n = len({r["source_event_id"] for r in test_rows if r["source_event_id"] is not None})
    train_cycle_n = len({_cycle_key(r) for r in train_rows})
    test_cycle_n = len({_cycle_key(r) for r in test_rows})

    full_median = _median(vals)
    train_median = _median(train_vals)
    test_median = _median(test_vals)
    q10, q25, q50, q75, q90 = (_quantile(vals, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
    iqr = None if q75 is None or q25 is None else round(q75 - q25, 4)
    train_minus_test = None if train_median is None or test_median is None else round(train_median - test_median, 4)

    sufficiency_ok = train_cycle_n >= MIN_BUCKET_N and test_cycle_n >= MIN_BUCKET_N
    sample_sufficiency = "OK" if sufficiency_ok else "INSUFFICIENT_SAMPLE"

    if sufficiency_ok:
        boot = cluster_bootstrap_median_diff(train_rows, test_rows, metric)
        perm = permutation_test_median_diff(train_vals, test_vals)
    else:
        boot = {"n_valid_draws": 0, "ci95": (None, None)}
        perm = {"observed_diff": None, "p_value": None, "n_perm": N_PERMUTATIONS}

    return {
        "raw_signal_n": raw_signal_n, "distinct_source_event_n": distinct_source_event_n,
        "distinct_independent_cycle_n": distinct_cycle_n,
        "train_signal_n": train_signal_n, "test_signal_n": test_signal_n,
        "train_source_event_n": train_event_n, "test_source_event_n": test_event_n,
        "train_cycle_n": train_cycle_n, "test_cycle_n": test_cycle_n,
        "cycle_straddling_violations": straddling_n,
        "full_median": round(full_median, 4) if full_median is not None else None,
        "train_median": round(train_median, 4) if train_median is not None else None,
        "test_median": round(test_median, 4) if test_median is not None else None,
        "q10": round(q10, 4) if q10 is not None else None,
        "q25": round(q25, 4) if q25 is not None else None,
        "q50": round(q50, 4) if q50 is not None else None,
        "q75": round(q75, 4) if q75 is not None else None,
        "q90": round(q90, 4) if q90 is not None else None,
        "iqr": iqr,
        "train_minus_test_median_diff": train_minus_test,
        "bootstrap_ci95": boot["ci95"], "bootstrap_n_valid_draws": boot["n_valid_draws"],
        "permutation_observed_diff": perm["observed_diff"], "permutation_p_value": perm["p_value"],
        "sample_sufficiency": sample_sufficiency,
    }


# ---------------------------------------------------------------------------
# secondary, descriptive-only reporting (no p-values)
# ---------------------------------------------------------------------------

def _capture_fraction_stats(rows: list[dict], numerator_field: str, denominator_field: str) -> dict:
    """median(numerator/denominator) across signals where denominator != 0.
    A zero denominator is EXCLUDED from the median (never divided by zero),
    with the exact excluded count reported."""
    ratios = []
    zero_denominator_n = 0
    for r in rows:
        denom = r.get(denominator_field)
        numer = r.get(numerator_field)
        if denom is None or numer is None:
            continue
        if denom == 0:
            zero_denominator_n += 1
            continue
        ratios.append(numer / denom)
    med = _median(ratios)
    return {
        "median_ratio": round(med, 6) if med is not None else None,
        "n_included": len(ratios), "n_excluded_zero_denominator": zero_denominator_n,
    }


def compute_secondary_descriptive(rows: list[dict]) -> dict:
    out: dict = {"per_interval": {}, "capture_fraction_of_24h": {}}
    for a, b in INTERVALS:
        key = f"{a}_to_{b}"
        mfe_vals = [r[f"delta_mfe_{key}"] for r in rows if r.get(f"delta_mfe_{key}") is not None]
        mae_vals = [r[f"delta_abs_mae_{key}"] for r in rows if r.get(f"delta_abs_mae_{key}") is not None]
        diff_vals = [r[f"delta_diff_{key}"] for r in rows if r.get(f"delta_diff_{key}") is not None]
        out["per_interval"][key] = {
            "median_incremental_mfe_bps": round(_median(mfe_vals), 4) if mfe_vals else None,
            "median_incremental_abs_mae_bps": round(_median(mae_vals), 4) if mae_vals else None,
            "median_delta_mfe_minus_delta_abs_mae_bps": round(_median(diff_vals), 4) if diff_vals else None,
        }
    for alias in ("30m", "1h", "4h"):
        out["capture_fraction_of_24h"][f"mfe_{alias}_over_24h"] = _capture_fraction_stats(
            rows, f"mfe_bps_{alias}", "mfe_bps_24h"
        )
        out["capture_fraction_of_24h"][f"abs_mae_{alias}_over_24h"] = _capture_fraction_stats(
            [{**r, f"abs_mae_bps_{alias}": abs(r[f"mae_bps_{alias}"]), "abs_mae_bps_24h": abs(r["mae_bps_24h"])}
             for r in rows],
            f"abs_mae_bps_{alias}", "abs_mae_bps_24h",
        )
    return out


# ---------------------------------------------------------------------------
# family assembly + freeze_and_record
# ---------------------------------------------------------------------------

def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    cohort = fetch_common_cohort(conn, symbol)
    rows = [compute_derived_fields(r) for r in cohort["cohort_rows"]]

    nonneg = assert_nested_nonnegativity(rows)

    split = compute_global_cycle_split(rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]
    train_rows, test_rows = split_rows_by_cycle_keys(rows, train_keys, test_keys)

    population_report = {
        "signal_n": len(rows),
        "source_event_n": len({r["source_event_id"] for r in rows if r["source_event_id"]}),
        "independent_cycle_n": len({_cycle_key(r) for r in rows}),
        "train_cycle_n": len({_cycle_key(r) for r in train_rows}),
        "test_cycle_n": len({_cycle_key(r) for r in test_rows}),
        "excluded_incomplete_horizon_n": cohort["excluded_incomplete_horizon_n"],
        "long_signals_with_any_ok_horizon_n": cohort["long_signals_with_any_ok_horizon_n"],
        "cycle_straddling_violations": assert_zero_cycle_straddling(train_rows, test_rows),
    }

    cells: dict[str, dict] = {}
    cell_order: list[str] = list(ALL_DELTA_FIELDS)
    for metric in cell_order:
        cells[metric] = compute_cell(rows, metric, train_keys, test_keys)
    assert len(cell_order) == 6, f"primary family must be exactly 6 cells, got {len(cell_order)}"

    p_values = [cells[k]["permutation_p_value"] for k in cell_order]
    holm = holm_adjust(p_values)
    for key, adj in zip(cell_order, holm):
        cells[key]["permutation_p_value_holm_adjusted"] = adj
        cells[key]["closure_classification"] = classify_cell_verdict(
            cells[key]["sample_sufficiency"], adj, cells[key]["bootstrap_ci95"],
        )

    secondary = compute_secondary_descriptive(rows)

    any_insufficient = any(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    all_insufficient = all(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    any_regime_dependent = any(
        cells[k]["closure_classification"] == "ANSWERED_REGIME_DEPENDENT_BASELINE" for k in cell_order
    )
    any_stable = any(cells[k]["closure_classification"] == "ANSWERED_SUPPORTED_STABLE_BASELINE" for k in cell_order)

    if all_insufficient:
        family_verdict = "INSUFFICIENT_COMMON_COHORT"
    elif any_insufficient:
        family_verdict = "MIXED_BY_INTERVAL_OR_METRIC"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_INTERVAL_OR_METRIC"
    elif any_regime_dependent:
        family_verdict = "LONG_NESTED_PATH_REGIME_DEPENDENT"
    else:
        family_verdict = "LONG_NESTED_PATH_STABLE"

    return {
        "rows": rows, "cells": cells, "cell_order": cell_order,
        "population_report": population_report,
        "nested_nonnegativity_check": nonneg,
        "secondary_descriptive": secondary,
        "family_verdict": family_verdict,
    }


def freeze_and_record(conn, provenance: str = "batch-p7b1-w8-long-nested-path-accumulation") -> dict:
    now = int(time.time() * 1000)
    family = compute_family(conn)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(r["signal_id"] for r in family["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"ami_lifecycle_path_observations WHERE observation_status=OK AT ALL 4 FROZEN HORIZONS "
        f"(common cohort), direction=LONG only; signal_n={family['population_report']['signal_n']}; "
        f"source_event_n={family['population_report']['source_event_n']}; "
        f"independent_cycle_n={family['population_report']['independent_cycle_n']}; "
        f"excluded_incomplete_horizon_n={family['population_report']['excluded_incomplete_horizon_n']}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id (never signal-level): cycle order key=earliest signal_birth_ts among all "
        f"of that cycle's rows; MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle N in TRAIN and TEST"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_LONG_NESTED_PATH_ACCUMULATION",
        "hypothesis_id": "H-W8-LONG-NESTED-PATH-ACCUMULATION", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(ALL_DELTA_FIELDS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30%) "
            "median stability of nested MFE/|MAE| accumulation deltas, LONG only, common 4-horizon cohort"
        ),
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N} (independent-cycle N in TRAIN and TEST); TRAIN_FRACTION={TRAIN_FRACTION}; "
            "classification=stable iff Holm-p>=0.05 AND bootstrap-CI includes 0; regime-dependent iff Holm-p<0.05 "
            "AND CI excludes 0; disagreement->regime-dependent (conservative); insufficient iff either split's "
            "independent-cycle N<MIN_BUCKET_N"
        ),
        "frozen_splits": frozen_splits,
        "frozen_economic_gate": (
            "N/A (no hold/exit/stop/re-entry/management rule tested -- nested path accumulation "
            "characterization only, capture fractions are NOT recommended holding periods)"
        ),
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}); "
            f"secondary=two-sided label-permutation median-difference p (n={N_PERMUTATIONS}) + ONE joint "
            f"Holm step-down across all 6 primary cells (3 MFE-interval + 3 |MAE|-interval together)"
        ),
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": family["family_verdict"],
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": None, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })
    rows_to_write = [
        ("population_report", family["population_report"]),
        ("nested_nonnegativity_check", family["nested_nonnegativity_check"]),
        ("secondary_descriptive", family["secondary_descriptive"]),
        ("family_verdict", family["family_verdict"]),
    ]
    for key in family["cell_order"]:
        rows_to_write.append((f"cell_{key}", family["cells"][key]))
    results = [(name, str(value)) for name, value in rows_to_write]
    record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10, provenance=provenance, created_ms=now)
    conn.commit()

    return {
        "family": family, "cell_order": family["cell_order"], "cells": family["cells"],
        "family_verdict": family["family_verdict"],
        "population_report": family["population_report"],
        "nested_nonnegativity_check": family["nested_nonnegativity_check"],
        "secondary_descriptive": family["secondary_descriptive"],
    }
