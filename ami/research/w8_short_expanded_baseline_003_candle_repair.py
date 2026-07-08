"""BATCH-P7B-1 (W8-SHORT-EXPANDED-BASELINE-003-CANDLE-REPAIR): corrected-data
rerun of E-W8-HOLD-BASELINE-002-SHORT-EXPANDED /
E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED, using the EFFECTIVE (post
candle-gap-repair) path population instead of the frozen pre-repair
"path-v2"-only population those two still use.

IMMUTABLE PRECEDENT: E-W8-HOLD-BASELINE-002-SHORT-EXPANDED and
E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED are NEVER touched by this
module -- two completely separate experiment_ids, registry rows, and result
sets. Their own frozen results remain the historical record of what the
PRE-REPAIR candle data implied for the cycle-grouped SHORT rerun (same
discipline as ami.research.w8_long_nested_path_accumulation_002_candle_repair
did for E-W8-LONG-NESTED-PATH-ACCUMULATION-001).

EFFECTIVE PATH SELECTION (mandatory precondition, verified before any cell is
computed): `verify_effective_path_selection_integrity()` calls
ami.lifecycle.path_candle_repair_correction.effective_path_selection_audit()
and requires physical_row_count_total=1466, duplicate_physical_pair_n=170,
effective_row_count=1296, duplicate_effective_pair_n=0 (i.e. exactly one
effective row per (signal_id, horizon_name)). If ANY of these fail, this
module computes NO cells at all and both new experiment_ids are frozen with
scientific_verdict="BLOCKED_BY_EFFECTIVE_PATH_SELECTION" -- never silently
falls back to a partial or best-guess population.

This module's ONLY population-fetching functions (fetch_raw_bps_population/
fetch_vol_normalized_population) are the two pieces of
w8_short_expanded_baseline NOT reused verbatim -- they are rewritten to call
ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations()
instead of feature_gateway.fetch_path_observations() directly, exactly
mirroring how w8_long_nested_path_accumulation_002_candle_repair.py overrode
only fetch_common_cohort(). Every other function (compute_global_cycle_split,
split_rows_by_cycle_keys, assert_zero_cycle_straddling, _cycle_key,
compute_cell, _cell_rows, compute_composition_diagnostic, classify_cell_verdict,
holm_adjust, cluster_bootstrap_median_diff, permutation_test_median_diff) is
imported from w8_short_expanded_baseline/w8_hold_baseline UNCHANGED -- not
reimplemented, verified by `is` identity in tests.

CYCLE MEMBERSHIP IS NOT REUSED FROM -002 BLINDLY: compute_global_cycle_split()
is recomputed fresh from the corrected (effective-selector) SHORT population,
same "never fed a hardcoded prior membership list" reasoning as the nested-
path -002 rerun.

MIN_BUCKET_N=20 applies to INDEPENDENT-CYCLE N in each split, never
signal-level N (unchanged from -002). Holm step-down is invoked over all 16
permutation p-values together, but ami.research.w7a_state_structure_aging_
market_clocks.holm_adjust() itself only performs work on the non-None subset
-- if every cell is INSUFFICIENT_SAMPLE (all 16 permutation p-values None),
holm_adjust() returns immediately with n=0, i.e. NO Holm computation actually
runs. This module never lowers MIN_BUCKET_N, never pools horizons, never
pools LONG+SHORT, and never reverts to a signal-level split to manufacture
sufficiency.

DESCRIPTIVE_ONLY_NOT_INFERENTIAL: every cell's full-population descriptive
fields (full_median/q10/q25/q50/q75/q90/iqr) are computed by compute_cell()
regardless of sufficiency (already the existing, unchanged behavior). Any
cell classified INSUFFICIENT_SAMPLE additionally carries
cells[key]["descriptive_only_label"]="DESCRIPTIVE_ONLY_NOT_INFERENTIAL" --
these descriptive values are NEVER converted into a stable/regime-dependent/
null/alpha claim; `classify_cell_verdict()` already guarantees an insufficient
cell's closure_classification stays "INSUFFICIENT_SAMPLE", never "stable" or
"regime-dependent".

CORRECTION IMPACT AUDIT: `compute_correction_impact_audit()` independently
re-derives (via ami.lifecycle.path_candle_repair_correction.identify_affected_pairs(),
filtered to direction=SHORT) the dependency-reconciliation's own expected
SHORT-specific figures (45 affected rows / 28 signals / 24 events / 18 cycles
/ 0 volatility-only corrections, concentrated in swing_24h) and reports the
exact actual values plus any mismatch -- never silently substitutes the
expected numbers for the measured ones.

COMPOSITION DIAGNOSTIC (secondary, descriptive-only, NO p-values): reuses
w8_short_expanded_baseline.compute_composition_diagnostic() VERBATIM (already
generic over an SHORT-filtered row list, not tied to any particular
path_definition_version).

NOT_A_MANAGEMENT_WAVE: no stop/exit/partial-exit/time-stop/re-entry/
cancellation rule anywhere in this module. NO_ECONOMIC_CLAIM: no PnL/alpha
claim is made or implied by any verdict this module can produce.
NO_SIGNAL_BACKFILL / NO_CANDLE_OR_PATH_MUTATION: reads the EXISTING,
now-repaired canonical population exactly as-is; triggers no new candle
fetch, no new path computation, no signal/event backfill of any kind.
"""
from __future__ import annotations
import ast
import hashlib
import time

from ami.lifecycle.path_candle_repair_correction import (
    OLD_PATH_DEFINITION_VERSION,
    PATH_DATA_VERSION_CANDLE_REPAIR_R1,
    effective_path_selection_audit,
    fetch_effective_path_observations,
    identify_affected_pairs,
)
from ami.research.feature_gateway import fetch_lifecycle_signals
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.research.w8_hold_baseline import HORIZONS, N_BOOTSTRAP, N_PERMUTATIONS, classify_cell_verdict
from ami.research.w8_short_expanded_baseline import (
    NEW_SETUP_ID,
    RAW_BPS_EXPERIMENT_ID as OLD_RAW_BPS_EXPERIMENT_ID,
    RAW_BPS_METRICS,
    VOL_NORMALIZED_EXPERIMENT_ID as OLD_VOL_NORMALIZED_EXPERIMENT_ID,
    VOL_NORMALIZED_METRICS,
    DIRECTION,
    _cell_rows,
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_cell,
    compute_composition_diagnostic,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results

EXPERIMENT_ID_CANDLE_REPAIR_SUFFIX = "-003-SHORT-EXPANDED-CANDLE-REPAIR"
RAW_BPS_EXPERIMENT_ID = "E-W8-HOLD-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR"
VOL_NORMALIZED_EXPERIMENT_ID = "E-W8-VOL-NORMALIZED-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR"
RESEARCH_CONTEXT_ID = "w8-short-expanded-baseline-003-candle-repair"

CANDLE_DATA_VERSION = "candle-binance-fapi-repair-v1"
PATH_DATA_VERSION = PATH_DATA_VERSION_CANDLE_REPAIR_R1  # "path-v2-candle-repair-r1"
_CORRECTED_DATA_RERUN_OF = {
    RAW_BPS_EXPERIMENT_ID: OLD_RAW_BPS_EXPERIMENT_ID,
    VOL_NORMALIZED_EXPERIMENT_ID: OLD_VOL_NORMALIZED_EXPERIMENT_ID,
}

_METRIC_EXPERIMENT_OF = {
    "mfe_bps": RAW_BPS_EXPERIMENT_ID, "mae_bps": RAW_BPS_EXPERIMENT_ID,
    "mfe_anchor_vol_units": VOL_NORMALIZED_EXPERIMENT_ID, "mae_anchor_vol_units": VOL_NORMALIZED_EXPERIMENT_ID,
}

# ---------------------------------------------------------------------------
# Part 0 -- mandatory effective-path selection integrity gate
# ---------------------------------------------------------------------------

_EXPECTED_PHYSICAL_ROW_TOTAL = 1466
_EXPECTED_DUPLICATE_PHYSICAL_PAIR_N = 170
_EXPECTED_EFFECTIVE_ROW_COUNT = 1296
_EXPECTED_DUPLICATE_EFFECTIVE_PAIR_N = 0


def verify_effective_path_selection_integrity(conn) -> dict:
    """Read-only integrity gate, run BEFORE any population is fetched or any
    cell is computed. Never assumes the checkpoint holds -- always measures
    it fresh via effective_path_selection_audit()."""
    audit = effective_path_selection_audit(conn, RESEARCH_CONTEXT_ID)
    checks = {
        "physical_row_count_total": (audit["physical_row_count_total"], _EXPECTED_PHYSICAL_ROW_TOTAL),
        "duplicate_physical_pair_n": (audit["duplicate_physical_pair_n"], _EXPECTED_DUPLICATE_PHYSICAL_PAIR_N),
        "effective_row_count": (audit["effective_row_count"], _EXPECTED_EFFECTIVE_ROW_COUNT),
        "duplicate_effective_pair_n": (audit["duplicate_effective_pair_n"], _EXPECTED_DUPLICATE_EFFECTIVE_PAIR_N),
    }
    mismatches = {name: {"actual": actual, "expected": expected}
                  for name, (actual, expected) in checks.items() if actual != expected}
    return {"audit": audit, "checks": checks, "mismatches": mismatches, "passed": not mismatches}


# ---------------------------------------------------------------------------
# population assembly -- ONLY these two functions differ from
# w8_short_expanded_baseline (effective selector instead of raw fetch)
# ---------------------------------------------------------------------------

def fetch_raw_bps_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_effective_path_observations(conn, RESEARCH_CONTEXT_ID, equals={"observation_status": "OK"})
    rows = []
    for o in obs_rows:
        s = signals.get(o["signal_id"])
        if s is None or s["direction"] != DIRECTION:
            continue
        rows.append({
            **o, "direction": s["direction"], "signal_birth_ts": s["signal_birth_ts"],
            "source_event_id": s["source_event_id"], "independent_cycle_id": s["independent_cycle_id"],
            "setup_id": s["setup_id"],
        })
    return rows


def fetch_vol_normalized_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_effective_path_observations(
        conn, RESEARCH_CONTEXT_ID, equals={"observation_status": "OK", "volatility_status": "OK"},
    )
    rows = []
    for o in obs_rows:
        s = signals.get(o["signal_id"])
        if s is None or s["direction"] != DIRECTION:
            continue
        rows.append({
            **o, "direction": s["direction"], "signal_birth_ts": s["signal_birth_ts"],
            "source_event_id": s["source_event_id"], "independent_cycle_id": s["independent_cycle_id"],
            "setup_id": s["setup_id"],
        })
    return rows


# ---------------------------------------------------------------------------
# correction impact audit -- verifies the reconciliation report's SHORT-only
# expectations against the real DB, never assumes them
# ---------------------------------------------------------------------------

_EXPECTED_CORRECTION_IMPACT = {
    "affected_physical_row_n": 45, "distinct_signal_n": 28, "distinct_event_n": 24,
    "distinct_cycle_n": 18, "class_b_n": 0,
}


def compute_correction_impact_audit(conn, symbol: str = "ETHUSDT") -> dict:
    result = identify_affected_pairs(conn, symbol=symbol)
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}

    def _short_only(pairs):
        return [(sid, h) for sid, h in pairs if signals.get(sid, {}).get("direction") == "SHORT"]

    class_a_short = _short_only(result["class_a"])
    class_b_short = _short_only(result["class_b"])
    affected_short = class_a_short + class_b_short

    def _cyc(sid):
        s = signals[sid]
        return s["independent_cycle_id"] or f"NOCYCLE-{s['source_event_id']}"

    by_horizon: dict[str, int] = {}
    for sid, h in affected_short:
        by_horizon[h] = by_horizon.get(h, 0) + 1

    actual = {
        "affected_physical_row_n": len(affected_short),
        "distinct_signal_n": len({sid for sid, _ in affected_short}),
        "distinct_event_n": len({signals[sid]["source_event_id"] for sid, _ in affected_short
                                  if signals[sid]["source_event_id"]}),
        "distinct_cycle_n": len({_cyc(sid) for sid, _ in affected_short}),
        "class_b_n": len(class_b_short),
    }
    mismatches = {k: {"actual": actual[k], "expected": v} for k, v in _EXPECTED_CORRECTION_IMPACT.items()
                  if actual[k] != v}
    dominant_horizon = max(by_horizon, key=by_horizon.get) if by_horizon else None

    return {
        "actual": actual, "expected": _EXPECTED_CORRECTION_IMPACT, "mismatches": mismatches,
        "by_horizon": by_horizon, "dominant_horizon": dominant_horizon,
        "concentrated_in_swing_24h": dominant_horizon == "swing_24h",
    }


# ---------------------------------------------------------------------------
# primary family -- 16 cells, cycle-grouped split recomputed fresh
# ---------------------------------------------------------------------------

_EXPECTED_COVERAGE = {"total_cycle_n": 61, "train_cycle_n": 42, "test_cycle_n": 19}


def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    integrity = verify_effective_path_selection_integrity(conn)
    if not integrity["passed"]:
        return {
            "blocked": True, "effective_path_integrity": integrity,
            "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
        }

    raw_bps_rows = fetch_raw_bps_population(conn, symbol)
    vol_norm_rows = fetch_vol_normalized_population(conn, symbol)

    # cycle membership recomputed fresh from the corrected population -- never reused from -002
    split = compute_global_cycle_split(raw_bps_rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    per_horizon_split_report: dict[str, dict] = {}
    for horizon in HORIZONS:
        raw_cell_rows = _cell_rows(raw_bps_rows, horizon)
        vol_cell_rows = _cell_rows(vol_norm_rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(raw_cell_rows, train_keys, test_keys)
        train_cycle_n = len({_cycle_key(r) for r in train_r})
        test_cycle_n = len({_cycle_key(r) for r in test_r})
        per_horizon_split_report[horizon] = {
            "raw_signal_n": len(raw_cell_rows),
            "source_event_n": len({r["source_event_id"] for r in raw_cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in raw_cell_rows}),
            "train_cycle_n": train_cycle_n,
            "test_cycle_n": test_cycle_n,
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
            "sufficiency_verdict": (
                "OK" if train_cycle_n >= MIN_BUCKET_N and test_cycle_n >= MIN_BUCKET_N else "INSUFFICIENT_SAMPLE"
            ),
        }
        for metric in RAW_BPS_METRICS:
            key = f"{metric}|{horizon}"
            cell_order.append(key)
            cells[key] = compute_cell(raw_cell_rows, metric, train_keys, test_keys)
        for metric in VOL_NORMALIZED_METRICS:
            key = f"{metric}|{horizon}"
            cell_order.append(key)
            cells[key] = compute_cell(vol_cell_rows, metric, train_keys, test_keys)

    assert len(cell_order) == 16, f"primary family must be exactly 16 cells, got {len(cell_order)}"

    # ONE joint Holm correction across all 16 p-values -- holm_adjust() itself only computes over the
    # non-None subset; if all 16 are INSUFFICIENT_SAMPLE (all p-values None), it does no work at all
    p_values = [cells[k]["permutation_p_value"] for k in cell_order]
    holm = holm_adjust(p_values)
    for key, adj in zip(cell_order, holm):
        cells[key]["permutation_p_value_holm_adjusted"] = adj
        cells[key]["closure_classification"] = classify_cell_verdict(
            cells[key]["sample_sufficiency"], adj, cells[key]["bootstrap_ci95"],
        )
        if cells[key]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE":
            cells[key]["descriptive_only_label"] = "DESCRIPTIVE_ONLY_NOT_INFERENTIAL"

    any_insufficient = any(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    all_insufficient = all(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    any_regime_dependent = any(
        cells[k]["closure_classification"] == "ANSWERED_REGIME_DEPENDENT_BASELINE" for k in cell_order
    )
    any_stable = any(cells[k]["closure_classification"] == "ANSWERED_SUPPORTED_STABLE_BASELINE" for k in cell_order)

    if all_insufficient:
        family_verdict = "EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT_CORRECTED_DATA"
    elif any_insufficient and (any_regime_dependent or any_stable):
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_DATA"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_DATA"
    elif any_regime_dependent:
        family_verdict = "EXPANDED_SHORT_REGIME_DEPENDENT_CORRECTED_DATA"
    else:
        family_verdict = "EXPANDED_SHORT_STABLE_BASELINE_CORRECTED_DATA"

    coverage_expectation_check = {
        "expected": _EXPECTED_COVERAGE,
        "actual": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "matches_expectation": (
            split["total_cycle_n"] == _EXPECTED_COVERAGE["total_cycle_n"]
            and split["train_cycle_n"] == _EXPECTED_COVERAGE["train_cycle_n"]
            and split["test_cycle_n"] == _EXPECTED_COVERAGE["test_cycle_n"]
        ),
    }

    composition = compute_composition_diagnostic(raw_bps_rows)
    correction_audit = compute_correction_impact_audit(conn, symbol)

    return {
        "blocked": False, "effective_path_integrity": integrity,
        "raw_bps_rows": raw_bps_rows, "vol_norm_rows": vol_norm_rows,
        "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "coverage_expectation_check": coverage_expectation_check,
        "per_horizon_split_report": per_horizon_split_report,
        "composition_diagnostic_descriptive_only": composition,
        "correction_impact_audit": correction_audit,
        "raw_signal_n_population": len({r["signal_id"] for r in raw_bps_rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in raw_bps_rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in raw_bps_rows}),
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# comparison with the immutable -002 (descriptive only -- NOT independent
# replications, the cohorts share the same population definition)
# ---------------------------------------------------------------------------

def _read_old_metric(conn, experiment_id: str, metric_name: str):
    row = conn.execute(
        "SELECT metric_value FROM experiment_results WHERE experiment_id=? AND metric_name=?",
        (experiment_id, metric_name),
    ).fetchone()
    if row is None:
        return None
    try:
        return ast.literal_eval(row[0])
    except (ValueError, SyntaxError):
        return row[0]  # a bare verdict string is not a Python literal


def compare_with_v002(conn, family: dict) -> dict:
    """Reads -002's ALREADY-FROZEN, byte-immutable stored results (never
    recomputes them) and compares descriptively against this rerun's fresh
    computation. No new p-value is computed for the comparison itself."""
    v002_global_split = _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "global_cycle_split")
    v002_per_horizon = _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "per_horizon_split_report") or {}
    v002_family_verdict = _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "family_verdict")

    population_count_changes = {
        "raw_signal_n_population": {
            "v002": _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "raw_signal_n_population"),
            "v003": family["raw_signal_n_population"],
        },
        "distinct_source_event_n_population": {
            "v002": _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "distinct_source_event_n_population"),
            "v003": family["distinct_source_event_n_population"],
        },
        "distinct_independent_cycle_n_population": {
            "v002": _read_old_metric(conn, OLD_RAW_BPS_EXPERIMENT_ID, "distinct_independent_cycle_n_population"),
            "v003": family["distinct_independent_cycle_n_population"],
        },
    }
    split_count_changes = {"v002_global_split": v002_global_split, "v003_global_split": family["global_split"]}

    horizon_sufficiency_changes = {}
    for horizon in HORIZONS:
        v002_h = v002_per_horizon.get(horizon, {})
        v003_h = family["per_horizon_split_report"].get(horizon, {})
        v002_train, v002_test = v002_h.get("train_cycle_n"), v002_h.get("test_cycle_n")
        v002_sufficiency_derived = (
            "OK" if (v002_train or 0) >= MIN_BUCKET_N and (v002_test or 0) >= MIN_BUCKET_N
            else "INSUFFICIENT_SAMPLE" if v002_h else None
        )
        horizon_sufficiency_changes[horizon] = {
            # -002's stored per_horizon_split_report never carried an explicit "sufficiency_verdict"
            # field -- derived here (never invented) purely from its own stored train/test cycle N
            "v002_sufficiency_verdict_derived": v002_sufficiency_derived,
            "v003_sufficiency_verdict": v003_h.get("sufficiency_verdict"),
            "v002_test_cycle_n": v002_test, "v003_test_cycle_n": v003_h.get("test_cycle_n"),
        }

    cell_verdict_changes = {}
    for key in family["cell_order"]:
        metric = key.split("|")[0]
        old_id = OLD_RAW_BPS_EXPERIMENT_ID if metric in RAW_BPS_METRICS else OLD_VOL_NORMALIZED_EXPERIMENT_ID
        v002_c = _read_old_metric(conn, old_id, f"cell_{key}")
        v003_c = family["cells"][key]
        v002_verdict = v002_c.get("closure_classification") if v002_c else None
        v003_verdict = v003_c.get("closure_classification")
        cell_verdict_changes[key] = {
            "v002_verdict": v002_verdict, "v003_verdict": v003_verdict,
            "changed": v002_verdict != v003_verdict,
            "changed_from_insufficient_sample": (
                v002_verdict == "INSUFFICIENT_SAMPLE" and v003_verdict != "INSUFFICIENT_SAMPLE"
            ),
            "v002_full_median": v002_c.get("full_median") if v002_c else None,
            "v003_full_median": v003_c.get("full_median"),
        }

    any_cell_changed_from_insufficient = any(
        v["changed_from_insufficient_sample"] for v in cell_verdict_changes.values()
    )

    return {
        "v002_raw_bps_experiment_id": OLD_RAW_BPS_EXPERIMENT_ID,
        "v002_vol_normalized_experiment_id": OLD_VOL_NORMALIZED_EXPERIMENT_ID,
        "v002_family_verdict": v002_family_verdict, "v003_family_verdict": family["family_verdict"],
        "population_count_changes": population_count_changes,
        "split_count_changes": split_count_changes,
        "horizon_sufficiency_changes": horizon_sufficiency_changes,
        "cell_verdict_changes": cell_verdict_changes,
        "any_cell_changed_from_insufficient_sample": any_cell_changed_from_insufficient,
        "not_independent_replications_note": (
            "-002 and -003 share the identical direction=SHORT/setup-unfiltered population definition "
            "-- -003's population differs only via the effective-path selector picking up 170 "
            "candle-repair-corrected rows. This comparison is a consistency check on a corrected-data "
            "rerun, never treated as an independent confirmation of the same finding."
        ),
    }


# ---------------------------------------------------------------------------
# integrity: -002 experiments must remain byte-unchanged
# ---------------------------------------------------------------------------

def snapshot_old_experiments(conn) -> dict:
    def _snapshot(experiment_id: str) -> tuple:
        reg = conn.execute(
            "SELECT * FROM experiment_registry WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        results = conn.execute(
            "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=? "
            "ORDER BY metric_name", (experiment_id,),
        ).fetchall()
        return (reg, tuple(results))
    return {
        OLD_RAW_BPS_EXPERIMENT_ID: _snapshot(OLD_RAW_BPS_EXPERIMENT_ID),
        OLD_VOL_NORMALIZED_EXPERIMENT_ID: _snapshot(OLD_VOL_NORMALIZED_EXPERIMENT_ID),
    }


def assert_old_experiments_untouched(conn, before: dict) -> bool:
    return snapshot_old_experiments(conn) == before


# ---------------------------------------------------------------------------
# freeze_and_record -- writes to the two NEW experiment_ids only
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-w8-short-expanded-baseline-003-candle-repair") -> dict:
    now = int(time.time() * 1000)
    before_old = snapshot_old_experiments(conn)

    family = compute_family(conn)

    if family["blocked"]:
        dataset_hash = hashlib.sha256(b"BLOCKED_BY_EFFECTIVE_PATH_SELECTION").hexdigest()
        frozen_population = (
            "BLOCKED_BY_EFFECTIVE_PATH_SELECTION -- effective_path_selection_audit() mismatch, "
            "see effective_path_integrity for exact diffs; no population fetched, no cell computed"
        )
        for experiment_id, metrics_group in ((RAW_BPS_EXPERIMENT_ID, RAW_BPS_METRICS),
                                              (VOL_NORMALIZED_EXPERIMENT_ID, VOL_NORMALIZED_METRICS)):
            record_experiment_registry(conn, {
                "experiment_id": experiment_id, "question_ids": "FAM_W8_SHORT_EXPANDED_BASELINE",
                "hypothesis_id": "H-W8-SHORT-EXPANDED-BASELINE", "preregistered_at": now,
                "frozen_population": frozen_population, "frozen_features": ",".join(metrics_group),
                "frozen_target": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any target could be computed",
                "frozen_thresholds": (
                    f"MIN_BUCKET_N={MIN_BUCKET_N}; effective-path integrity required before any threshold applies"
                ),
                "frozen_splits": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any split could be computed",
                "frozen_economic_gate": "N/A (no management/exit/stop/re-entry rule tested)",
                "frozen_statistical_gate": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any statistic ran",
                "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
                "software_verdict": "PASSED", "scientific_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
                "mutation_test_count": 0, "mutation_test_passed": 1,
                "supersedes_experiment_id": _CORRECTED_DATA_RERUN_OF[experiment_id], "report_artifact_id": None,
                "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
            })
            results = [
                ("family_verdict", "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"),
                ("effective_path_integrity", str(family["effective_path_integrity"])),
                ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
                ("corrected_data_rerun_of", _CORRECTED_DATA_RERUN_OF[experiment_id]),
            ]
            record_experiment_results(conn, experiment_id, results, schema_version=10,
                                       provenance=provenance, created_ms=now)
        conn.commit()
        return {
            "family": family, "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
            "old_experiments_untouched": assert_old_experiments_untouched(conn, before_old),
        }

    comparison = compare_with_v002(conn, family)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(
            f"{r['signal_id']}|{r['horizon_name']}" for r in family["raw_bps_rows"] + family["vol_norm_rows"]
        )).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"EFFECTIVE ami_lifecycle_path_observations (path-v2-candle-repair-r1 where corrected, else "
        f"path-v2) WHERE observation_status=OK (raw-bps) / observation_status=OK AND volatility_status=OK "
        f"(vol-normalized), direction=SHORT only; raw_signal_n={family['raw_signal_n_population']}; "
        f"distinct_source_event_n={family['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={family['distinct_independent_cycle_n_population']}; "
        f"candle_data_version={CANDLE_DATA_VERSION}; path_data_version={PATH_DATA_VERSION}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id, RECOMPUTED FRESH from the corrected (effective-selector) population "
        f"(never reused from -002's partition); MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle "
        f"N per split/cell"
    )

    for experiment_id, metrics_group in ((RAW_BPS_EXPERIMENT_ID, RAW_BPS_METRICS),
                                          (VOL_NORMALIZED_EXPERIMENT_ID, VOL_NORMALIZED_METRICS)):
        record_experiment_registry(conn, {
            "experiment_id": experiment_id, "question_ids": "FAM_W8_SHORT_EXPANDED_BASELINE",
            "hypothesis_id": "H-W8-SHORT-EXPANDED-BASELINE", "preregistered_at": now,
            "frozen_population": frozen_population, "frozen_features": ",".join(metrics_group),
            "frozen_target": (
                "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30% of "
                "independent cycles) median stability, SHORT only, by horizon, CORRECTED (post "
                "candle-gap-repair) data"
            ),
            "frozen_thresholds": (
                f"MIN_BUCKET_N={MIN_BUCKET_N} (independent-cycle N per split/cell, NOT signal-level N); "
                f"TRAIN_FRACTION={TRAIN_FRACTION}; classification=stable iff Holm-p>=0.05 AND bootstrap-CI "
                "includes 0; regime-dependent iff Holm-p<0.05 AND CI excludes 0; disagreement->regime-dependent "
                "(conservative); insufficient iff either split's independent-cycle N<MIN_BUCKET_N"
            ),
            "frozen_splits": frozen_splits,
            "frozen_economic_gate": "N/A (no management/exit/stop/re-entry rule tested)",
            "frozen_statistical_gate": (
                f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}, "
                f"same seed convention as W8-HOLD-BASELINE); secondary=two-sided label-permutation median-"
                f"difference p (n={N_PERMUTATIONS}) + ONE joint Holm step-down across all 16 primary cells "
                f"(both raw-bps and vol-normalized metric groups together, not two separate 8-cell "
                f"corrections; Holm only performs work if >=1 cell is sufficient)"
            ),
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": family["family_verdict"],
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": _CORRECTED_DATA_RERUN_OF[experiment_id], "report_artifact_id": None,
            "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
        })
        rows_to_write = [
            ("raw_signal_n_population", family["raw_signal_n_population"]),
            ("distinct_source_event_n_population", family["distinct_source_event_n_population"]),
            ("distinct_independent_cycle_n_population", family["distinct_independent_cycle_n_population"]),
            ("global_cycle_split", family["global_split"]),
            ("coverage_expectation_check", family["coverage_expectation_check"]),
            ("per_horizon_split_report", family["per_horizon_split_report"]),
            ("composition_diagnostic_descriptive_only", family["composition_diagnostic_descriptive_only"]),
            ("correction_impact_audit", family["correction_impact_audit"]),
            ("effective_path_integrity", family["effective_path_integrity"]),
            ("comparison_with_v002", comparison),
            ("family_verdict", family["family_verdict"]),
            ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
            ("corrected_data_rerun_of", _CORRECTED_DATA_RERUN_OF[experiment_id]),
        ]
        for key in family["cell_order"]:
            metric = key.split("|")[0]
            if _METRIC_EXPERIMENT_OF[metric] != experiment_id:
                continue
            rows_to_write.append((f"cell_{key}", family["cells"][key]))
        results = [(name, str(value)) for name, value in rows_to_write]
        record_experiment_results(conn, experiment_id, results, schema_version=10,
                                   provenance=provenance, created_ms=now)
    conn.commit()

    old_experiments_untouched = assert_old_experiments_untouched(conn, before_old)

    return {
        "family": family, "comparison_with_v002": comparison,
        "old_experiments_untouched": old_experiments_untouched,
        "cell_order": family["cell_order"], "cells": family["cells"],
        "family_verdict": family["family_verdict"],
        "raw_signal_n_population": family["raw_signal_n_population"],
        "distinct_source_event_n_population": family["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": family["distinct_independent_cycle_n_population"],
        "global_split": family["global_split"],
        "coverage_expectation_check": family["coverage_expectation_check"],
        "per_horizon_split_report": family["per_horizon_split_report"],
        "correction_impact_audit": family["correction_impact_audit"],
        "effective_path_integrity": family["effective_path_integrity"],
    }
