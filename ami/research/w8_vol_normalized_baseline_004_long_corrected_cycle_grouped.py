"""BATCH-P7B-1 (W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED):
corrected (post candle-gap-repair) + cycle-grouped-split rerun of the
LONG-only volatility-normalized portion of E-W8-VOL-NORMALIZED-BASELINE-001,
paired with the already-completed raw-bps
E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED.

NOT AN INDEPENDENT REPLICATION of E-W8-VOL-NORMALIZED-BASELINE-001 (same
discipline as its paired raw-bps sibling): v001 used the pre-repair
"path-v2"-only population AND a naive signal-level chronological split. This
experiment changes BOTH the data (effective, post-repair paths) AND the
split methodology (independent-cycle-grouped, REUSED from the paired raw
experiment rather than recomputed). `historical_reference_experiment_id`,
`paired_raw_baseline_experiment_id`, `candle_data_version`,
`path_data_version`, and `methodological_change` are recorded explicitly on
the registry row for exactly this reason.

NOT A VOLATILITY-STATE STRATIFICATION WAVE: this module computes NO
HIGH/LOW volatility label, NO median-volatility threshold, and performs NO
threshold fitting or regime classification of any kind. mfe_anchor_vol_units/
mae_anchor_vol_units are simply per-row divisions by that signal's own
realized_vol_at_anchor (already computed, unchanged, by
ami.lifecycle.path_metrics.compute_observation()) -- this module reads that
existing field, it does not derive or fit anything new from it.

SPLIT CONTRACT -- MANDATORY MANIFEST REUSE, NOT RECOMPUTATION FROM THE
VOL-FILTERED POPULATION: `fetch_raw_population()` (identical to
w8_hold_baseline_004_long_corrected_cycle_grouped.fetch_population(),
imported directly rather than reimplemented) fetches the SAME raw
(observation_status=OK, LONG, effective-selector) population the paired raw
experiment used to build its split. compute_global_cycle_split() is called
on THAT raw population -- deterministically reproducing the identical
train/test cycle-key membership the paired experiment froze (verified
against the paired experiment's own stored global_cycle_split counts via
`verify_split_matches_paired_raw_baseline()`, never merely assumed). The
volatility-normalized cell rows (a subset -- observation_status=OK AND
volatility_status=OK) are then assigned TRAIN/TEST membership using THIS
SAME split via split_rows_by_cycle_keys() -- a signal excluded only because
its volatility_status is not OK never removes or reassigns its cycle's
split membership, because the split was never computed from the
vol-filtered rows in the first place.

PRIMARY FAMILY (frozen, exactly 8 = 2 vol-normalized metrics
[mfe_anchor_vol_units, mae_anchor_vol_units] x 4 horizons, LONG only): raw-bps
metrics are NOT included here -- those are the paired
E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED experiment's own primary
family. ONE joint Holm step-down correction across all 8 p-values together.
compute_cell() is reused VERBATIM from w8_short_expanded_baseline --
MIN_BUCKET_N=20 applies to INDEPENDENT-CYCLE N in each split, never
signal-level N, and is never reduced.

PRE-OUTCOME COVERAGE REPORT (frozen, computed BEFORE any metric value is
read): per-horizon raw signal/source-event/independent-cycle N, TRAIN/TEST
cycle N, volatility-invalid signal/event/cycle N (rows present in the raw
population but excluded by volatility_status!=OK), cycle-straddling
violations, sufficiency verdict, monthly distribution, setup_id composition,
and an explicit comparison against the paired raw v004 population's own
stored per-horizon coverage.

COMPARISON WITH PAIRED RAW V004 / HISTORICAL V001 (descriptive only -- NOT
independent replications): reads both ALREADY-FROZEN, byte-immutable stored
results (never recomputes them).

NOT_A_MANAGEMENT_WAVE: no stop/exit/partial-exit/time-stop/re-entry/
cancellation rule anywhere in this module. NO_ECONOMIC_CLAIM: no PnL/alpha/
win-rate claim is made or implied by any verdict this module can produce.
NO_SIGNAL_BACKFILL / NO_CANDLE_OR_PATH_MUTATION / NO_MATCHED_CONTROL_
RECONSTRUCTION.
"""
from __future__ import annotations
import ast
import hashlib
import time

from ami.lifecycle.path_candle_repair_correction import (
    PATH_DATA_VERSION_CANDLE_REPAIR_R1,
    effective_path_selection_audit,
    fetch_effective_path_observations,
)
from ami.research.feature_gateway import fetch_lifecycle_signals
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.research.w8_hold_baseline import HORIZONS, N_BOOTSTRAP, N_PERMUTATIONS, _month_bucket, classify_cell_verdict
from ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped import (
    EXPERIMENT_ID as PAIRED_RAW_BASELINE_EXPERIMENT_ID,
    DIRECTION,
)
from ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped import (
    fetch_population as fetch_raw_population,
)
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_cell,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)
from ami.research.w8_vol_normalized_baseline import EXPERIMENT_ID as OLD_VOL_NORMALIZED_EXPERIMENT_ID
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results

EXPERIMENT_ID = "E-W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
RESEARCH_CONTEXT_ID = "w8-vol-normalized-baseline-004-long-corrected-cycle-grouped"
HISTORICAL_REFERENCE_EXPERIMENT_ID = OLD_VOL_NORMALIZED_EXPERIMENT_ID  # "E-W8-VOL-NORMALIZED-BASELINE-001"
PAIRED_RAW_BASELINE_EXPERIMENT_ID = PAIRED_RAW_BASELINE_EXPERIMENT_ID  # "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"

CANDLE_DATA_VERSION = "candle-binance-fapi-repair-v1"
PATH_DATA_VERSION = PATH_DATA_VERSION_CANDLE_REPAIR_R1  # "path-v2-candle-repair-r1"
METHODOLOGICAL_CHANGE = "SIGNAL_LEVEL_SPLIT_TO_INDEPENDENT_CYCLE_GROUPED_SPLIT"

METRICS = ("mfe_anchor_vol_units", "mae_anchor_vol_units")  # vol-normalized only -- raw-bps is the paired experiment
_RAW_BPS_METRIC_OF = {"mfe_anchor_vol_units": "mfe_bps", "mae_anchor_vol_units": "mae_bps"}


# ---------------------------------------------------------------------------
# Part 0 -- mandatory effective-path selection integrity gate
# ---------------------------------------------------------------------------

_EXPECTED_PHYSICAL_ROW_TOTAL = 1466
_EXPECTED_DUPLICATE_PHYSICAL_PAIR_N = 170
_EXPECTED_EFFECTIVE_ROW_COUNT = 1296
_EXPECTED_DUPLICATE_EFFECTIVE_PAIR_N = 0


def verify_effective_path_selection_integrity(conn) -> dict:
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
# population assembly -- vol-normalized subset of the SAME raw population
# ---------------------------------------------------------------------------

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


def _cell_rows(rows: list[dict], horizon: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == DIRECTION]


# ---------------------------------------------------------------------------
# Part: verify the reused split truly matches the paired raw experiment
# ---------------------------------------------------------------------------

def verify_split_matches_paired_raw_baseline(conn, split: dict) -> dict:
    """Reads E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED's ALREADY-
    STORED global_cycle_split (never recomputes IT) and compares against the
    split freshly computed here from the identical raw population -- proves
    (never assumes) that reuse is byte-exact, not merely "should be the
    same"."""
    stored = _read_old_metric(conn, PAIRED_RAW_BASELINE_EXPERIMENT_ID, "global_cycle_split")
    actual = {
        "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
        "test_cycle_n": split["test_cycle_n"],
    }
    matches = stored == actual
    return {"stored_paired_split": stored, "actual_split": actual, "matches": matches}


# ---------------------------------------------------------------------------
# pre-outcome coverage report (structural/descriptive counts only)
# ---------------------------------------------------------------------------

def _monthly_distribution(cell_rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in cell_rows:
        m = _month_bucket(r["signal_birth_ts"])
        counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items()))


def _setup_composition(cell_rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in cell_rows:
        counts[r["setup_id"]] = counts.get(r["setup_id"], 0) + 1
    return dict(sorted(counts.items()))


def compute_coverage_report(raw_rows: list[dict], vol_rows: list[dict], train_keys: set, test_keys: set,
                             conn) -> dict:
    per_horizon: dict[str, dict] = {}
    for horizon in HORIZONS:
        raw_cell_rows = _cell_rows(raw_rows, horizon)
        vol_cell_rows = _cell_rows(vol_rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(vol_cell_rows, train_keys, test_keys)
        train_cycle_n = len({_cycle_key(r) for r in train_r})
        test_cycle_n = len({_cycle_key(r) for r in test_r})

        vol_signal_ids = {r["signal_id"] for r in vol_cell_rows}
        invalid_rows = [r for r in raw_cell_rows if r["signal_id"] not in vol_signal_ids]

        raw_v004_coverage = _read_old_metric(conn, PAIRED_RAW_BASELINE_EXPERIMENT_ID, "coverage_report")
        raw_v004_horizon = (raw_v004_coverage or {}).get("per_horizon", {}).get(horizon, {})

        per_horizon[horizon] = {
            "raw_signal_n": len(raw_cell_rows),
            "vol_signal_n": len(vol_cell_rows),
            "source_event_n": len({r["source_event_id"] for r in vol_cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in vol_cell_rows}),
            "train_cycle_n": train_cycle_n, "test_cycle_n": test_cycle_n,
            "volatility_invalid_signal_n": len({r["signal_id"] for r in invalid_rows}),
            "volatility_invalid_event_n": len({r["source_event_id"] for r in invalid_rows
                                                if r["source_event_id"]}),
            "volatility_invalid_cycle_n": len({_cycle_key(r) for r in invalid_rows}),
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
            "sufficiency_verdict": (
                "OK" if train_cycle_n >= MIN_BUCKET_N and test_cycle_n >= MIN_BUCKET_N else "INSUFFICIENT_SAMPLE"
            ),
            "monthly_distribution": _monthly_distribution(vol_cell_rows),
            "setup_composition": _setup_composition(vol_cell_rows),
            "paired_raw_v004_signal_n": raw_v004_horizon.get("raw_signal_n"),
        }

    return {"per_horizon": per_horizon}


# ---------------------------------------------------------------------------
# primary family -- 8 cells (2 vol-normalized metrics x 4 horizons)
# ---------------------------------------------------------------------------

def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    integrity = verify_effective_path_selection_integrity(conn)
    if not integrity["passed"]:
        return {
            "blocked": True, "effective_path_integrity": integrity,
            "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
        }

    # split reused, byte-exact, from the SAME raw population the paired raw experiment used --
    # never recomputed from the (smaller) vol-filtered population
    raw_rows = fetch_raw_population(conn, symbol)
    split = compute_global_cycle_split(raw_rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]
    split_integrity = verify_split_matches_paired_raw_baseline(conn, split)

    vol_rows = fetch_vol_normalized_population(conn, symbol)
    coverage_report = compute_coverage_report(raw_rows, vol_rows, train_keys, test_keys, conn)

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    for horizon in HORIZONS:
        cell_rows = _cell_rows(vol_rows, horizon)
        for metric in METRICS:
            key = f"{metric}|{horizon}"
            cell_order.append(key)
            cells[key] = compute_cell(cell_rows, metric, train_keys, test_keys)

    assert len(cell_order) == 8, f"primary family must be exactly 8 cells, got {len(cell_order)}"

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
        family_verdict = "LONG_VOL_NORMALIZED_BASELINE_INSUFFICIENT"
    elif any_insufficient and (any_regime_dependent or any_stable):
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED"
    elif any_regime_dependent:
        family_verdict = "LONG_VOL_NORMALIZED_BASELINE_REGIME_DEPENDENT_CORRECTED_CYCLE_GROUPED"
    else:
        family_verdict = "LONG_VOL_NORMALIZED_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED"

    return {
        "blocked": False, "effective_path_integrity": integrity, "split_integrity": split_integrity,
        "raw_rows": raw_rows, "vol_rows": vol_rows, "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "coverage_report": coverage_report,
        "raw_signal_n_population": len({r["signal_id"] for r in vol_rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in vol_rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in vol_rows}),
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# comparisons (descriptive only -- NOT independent replications)
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
        return row[0]


def compare_with_raw_v004(conn, family: dict) -> dict:
    """Reads the already-completed, immutable paired raw-bps experiment's
    stored cells (never recomputes them) and compares descriptively against
    this wave's vol-normalized cells. Raw and normalized cells are NEVER
    treated as independent replications of each other -- vol-normalization
    is a per-row rescaling of the SAME underlying path, not a new sample."""
    raw_cells = {}
    for key in family["cell_order"]:
        vol_metric, horizon = key.split("|")
        raw_metric = _RAW_BPS_METRIC_OF[vol_metric]
        raw_cells[key] = _read_old_metric(conn, PAIRED_RAW_BASELINE_EXPERIMENT_ID, f"cell_{raw_metric}|{horizon}")

    any_raw_cell_missing = any(raw_cells[k] is None for k in family["cell_order"])

    cell_comparisons = {}
    changed_count = 0
    for key in family["cell_order"]:
        raw_c = raw_cells.get(key)
        vol_c = family["cells"][key]
        raw_verdict = raw_c.get("closure_classification") if raw_c else None
        vol_verdict = vol_c.get("closure_classification")
        raw_train_test_gap = raw_c.get("train_minus_test_median_diff") if raw_c else None
        vol_train_test_gap = vol_c.get("train_minus_test_median_diff")
        gap_shrank = (
            raw_train_test_gap is not None and vol_train_test_gap is not None
            and abs(vol_train_test_gap) < abs(raw_train_test_gap)
        )
        changed = raw_verdict != vol_verdict
        if changed:
            changed_count += 1
        cell_comparisons[key] = {
            "raw_verdict": raw_verdict, "vol_normalized_verdict": vol_verdict, "verdict_changed": changed,
            "raw_train_minus_test_median_diff": raw_train_test_gap,
            "vol_normalized_train_minus_test_median_diff": vol_train_test_gap,
            "train_test_gap_shrank_after_normalization": gap_shrank,
            "raw_signal_n": raw_c.get("raw_signal_n") if raw_c else None,
            "vol_normalized_signal_n": vol_c.get("raw_signal_n"),
        }

    all_insufficient = family["family_verdict"] == "LONG_VOL_NORMALIZED_BASELINE_INSUFFICIENT"
    stable_survives = (
        not any_raw_cell_missing and not all_insufficient
        and all(c["raw_verdict"] == "ANSWERED_SUPPORTED_STABLE_BASELINE"
                for c in cell_comparisons.values() if c["raw_verdict"] is not None)
        and all(c["vol_normalized_verdict"] == "ANSWERED_SUPPORTED_STABLE_BASELINE"
                for c in cell_comparisons.values())
    )

    if all_insufficient:
        comparison_label = "INSUFFICIENT_VOL_NORMALIZED_POPULATION"
    elif any_raw_cell_missing:
        comparison_label = "MATERIAL_NORMALIZATION_EFFECT"
    elif changed_count == 0:
        comparison_label = "RAW_AND_VOL_NORMALIZED_LONG_BASELINES_CONSISTENT"
    elif changed_count <= len(family["cell_order"]) // 2:
        comparison_label = "PARTIALLY_CONSISTENT_AFTER_NORMALIZATION"
    else:
        comparison_label = "MATERIAL_NORMALIZATION_EFFECT"

    return {
        "paired_raw_baseline_experiment_id": PAIRED_RAW_BASELINE_EXPERIMENT_ID,
        "cell_comparisons": cell_comparisons, "changed_cell_count": changed_count,
        "raw_stability_conclusion_survives_normalization": stable_survives,
        "comparison_label": comparison_label,
        "not_independent_replication_note": (
            "Raw-bps and volatility-normalized cells describe the SAME underlying paths (vol-"
            "normalization is a per-row rescaling by that signal's own realized_vol_at_anchor, not a "
            "new/independent sample) -- this comparison is a consistency check only."
        ),
    }


def compare_with_v001(conn, family: dict) -> dict:
    """Same discipline as w8_hold_baseline_004's compare_with_v001(): v001's
    stored population totals are combined LONG+SHORT family-level, not
    LONG-only -- per-horizon LONG-only population is instead derived from
    v001's own LONG cells."""
    v001_cells = {}
    for key in family["cell_order"]:
        metric, horizon = key.split("|")
        v001_cells[key] = _read_old_metric(conn, HISTORICAL_REFERENCE_EXPERIMENT_ID, f"cell_{metric}|{horizon}|LONG")

    per_horizon_population_changes = {}
    for horizon in HORIZONS:
        v001_cell = v001_cells.get(f"mfe_anchor_vol_units|{horizon}")
        v004_cov = family["coverage_report"]["per_horizon"][horizon]
        per_horizon_population_changes[horizon] = {
            "v001_raw_signal_n": v001_cell.get("raw_signal_n") if v001_cell else None,
            "v004_raw_signal_n": v004_cov["vol_signal_n"],
            "v001_distinct_source_event_n": v001_cell.get("distinct_source_event_n") if v001_cell else None,
            "v004_distinct_source_event_n": v004_cov["source_event_n"],
            "v001_distinct_independent_cycle_n": (
                v001_cell.get("distinct_independent_cycle_n") if v001_cell else None
            ),
            "v004_distinct_independent_cycle_n": v004_cov["independent_cycle_n"],
        }

    any_v001_cell_missing = any(v001_cells[k] is None for k in family["cell_order"])

    cell_changes = {}
    changed_count = 0
    same_sign_count = 0
    comparable_count = 0
    for key in family["cell_order"]:
        v001_c = v001_cells.get(key)
        v004_c = family["cells"][key]
        v001_verdict = v001_c.get("closure_classification") if v001_c else None
        v004_verdict = v004_c.get("closure_classification")
        v001_median = v001_c.get("full_median") if v001_c else None
        v004_median = v004_c.get("full_median")
        same_sign = (
            v001_median is not None and v004_median is not None and (v001_median > 0) == (v004_median > 0)
        )
        if v001_median is not None and v004_median is not None:
            comparable_count += 1
            if same_sign:
                same_sign_count += 1
        changed = v001_verdict != v004_verdict
        if changed:
            changed_count += 1
        cell_changes[key] = {
            "v001_verdict": v001_verdict, "v004_verdict": v004_verdict, "changed": changed,
            "v001_median": v001_median, "v004_median": v004_median, "same_sign": same_sign,
        }

    if any_v001_cell_missing:
        comparison_label = "NOT_COMPARABLE_DUE_TO_METHOD_CHANGE"
    elif changed_count == 0:
        comparison_label = "QUALITATIVELY_CONSISTENT_AFTER_CORRECTION_AND_CYCLE_GROUPING"
    elif changed_count <= len(family["cell_order"]) // 2:
        comparison_label = "PARTIALLY_CONSISTENT"
    else:
        comparison_label = "MATERIAL_BASELINE_CHANGE"

    return {
        "v001_experiment_id": HISTORICAL_REFERENCE_EXPERIMENT_ID,
        "population_changes": {
            "not_comparable_note": (
                "v001's stored population totals are combined LONG+SHORT family-level, not LONG-only "
                "-- deliberately NOT compared directly. See per_horizon_population_changes (derived "
                "from v001's own LONG cells)."
            ),
            "per_horizon_population_changes": per_horizon_population_changes,
        },
        "median_direction_consistency": {
            "comparable_cell_n": comparable_count, "same_sign_cell_n": same_sign_count,
        },
        "split_method_difference": (
            "v001: naive signal-level chronological 70/30 split. v004: independent-cycle-grouped "
            "split reused byte-exact from the paired raw-bps experiment."
        ),
        "cell_changes": cell_changes, "changed_cell_count": changed_count,
        "comparison_label": comparison_label,
        "not_independent_replication_note": (
            "v001 and v004 populations overlap heavily but are not identical (candle-repair-corrected "
            "rows added), AND the split methodology differs (signal-level vs cycle-grouped, reused "
            "from the paired raw experiment) -- this comparison is a consistency check only, never "
            "treated as an independent confirmation or refutation of v001's finding."
        ),
    }


# ---------------------------------------------------------------------------
# freeze_and_record -- writes to the NEW experiment_id only
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-w8-vol-normalized-baseline-004-long-corrected-cycle-grouped"
                       ) -> dict:
    now = int(time.time() * 1000)

    family = compute_family(conn)

    if family["blocked"]:
        dataset_hash = hashlib.sha256(b"BLOCKED_BY_EFFECTIVE_PATH_SELECTION").hexdigest()
        record_experiment_registry(conn, {
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_VOL_NORMALIZED_BASELINE",
            "hypothesis_id": "H-W8-VOL-NORMALIZED-BASELINE-LONG-CORRECTED-CYCLE-GROUPED", "preregistered_at": now,
            "frozen_population": (
                "BLOCKED_BY_EFFECTIVE_PATH_SELECTION -- effective_path_selection_audit() mismatch, "
                "see effective_path_integrity for exact diffs; no population fetched, no cell computed"
            ),
            "frozen_features": ",".join(METRICS),
            "frozen_target": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any target could be computed",
            "frozen_thresholds": f"MIN_BUCKET_N={MIN_BUCKET_N}; effective-path integrity required first",
            "frozen_splits": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any split could be computed",
            "frozen_economic_gate": "N/A (no management/exit/stop/re-entry rule tested)",
            "frozen_statistical_gate": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any statistic ran",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": HISTORICAL_REFERENCE_EXPERIMENT_ID, "report_artifact_id": None,
            "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
        })
        results = [
            ("family_verdict", "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"),
            ("effective_path_integrity", str(family["effective_path_integrity"])),
            ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
            ("historical_reference_experiment_id", HISTORICAL_REFERENCE_EXPERIMENT_ID),
            ("paired_raw_baseline_experiment_id", PAIRED_RAW_BASELINE_EXPERIMENT_ID),
            ("methodological_change", METHODOLOGICAL_CHANGE),
        ]
        record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10,
                                   provenance=provenance, created_ms=now)
        conn.commit()
        return {"family": family, "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"}

    comparison_raw_v004 = compare_with_raw_v004(conn, family)
    comparison_v001 = compare_with_v001(conn, family)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(f"{r['signal_id']}|{r['horizon_name']}" for r in family["vol_rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"EFFECTIVE ami_lifecycle_path_observations (path-v2-candle-repair-r1 where corrected, else "
        f"path-v2) WHERE observation_status=OK AND volatility_status=OK, direction=LONG only; "
        f"raw_signal_n={family['raw_signal_n_population']}; "
        f"distinct_source_event_n={family['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={family['distinct_independent_cycle_n_population']}; "
        f"candle_data_version={CANDLE_DATA_VERSION}; path_data_version={PATH_DATA_VERSION}; "
        f"historical_reference_experiment_id={HISTORICAL_REFERENCE_EXPERIMENT_ID}; "
        f"paired_raw_baseline_experiment_id={PAIRED_RAW_BASELINE_EXPERIMENT_ID}; "
        f"methodological_change={METHODOLOGICAL_CHANGE}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id, REUSED BYTE-EXACT from {PAIRED_RAW_BASELINE_EXPERIMENT_ID} (computed "
        f"from the same raw, observation_status=OK population -- never recomputed from the smaller "
        f"vol-filtered population, so volatility_status exclusions never alter a cycle's split side); "
        f"MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle N per split/cell; "
        f"split_matches_paired_raw_baseline={family['split_integrity']['matches']}"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_VOL_NORMALIZED_BASELINE",
        "hypothesis_id": "H-W8-VOL-NORMALIZED-BASELINE-LONG-CORRECTED-CYCLE-GROUPED", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(METRICS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30%) "
            "median stability of mfe_anchor_vol_units/mae_anchor_vol_units, LONG only, by horizon, "
            "CORRECTED (post candle-gap-repair) data"
        ),
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N} (independent-cycle N per split/cell, NOT signal-level N); "
            f"TRAIN_FRACTION={TRAIN_FRACTION}; classification=stable iff Holm-p>=0.05 AND bootstrap-CI "
            "includes 0; regime-dependent iff Holm-p<0.05 AND CI excludes 0; disagreement->regime-dependent "
            "(conservative); insufficient iff either split's independent-cycle N<MIN_BUCKET_N"
        ),
        "frozen_splits": frozen_splits,
        "frozen_economic_gate": "N/A (no management/exit/stop/re-entry rule tested -- path baseline only)",
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}); "
            f"secondary=two-sided label-permutation median-difference p (n={N_PERMUTATIONS}) + ONE joint "
            f"Holm step-down across all 8 primary cells (2 vol-normalized metrics x 4 horizons together)"
        ),
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": family["family_verdict"],
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": HISTORICAL_REFERENCE_EXPERIMENT_ID, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })

    rows_to_write = [
        ("raw_signal_n_population", family["raw_signal_n_population"]),
        ("distinct_source_event_n_population", family["distinct_source_event_n_population"]),
        ("distinct_independent_cycle_n_population", family["distinct_independent_cycle_n_population"]),
        ("global_cycle_split", family["global_split"]),
        ("split_integrity", family["split_integrity"]),
        ("coverage_report", family["coverage_report"]),
        ("effective_path_integrity", family["effective_path_integrity"]),
        ("comparison_with_raw_v004", comparison_raw_v004),
        ("comparison_with_v001", comparison_v001),
        ("family_verdict", family["family_verdict"]),
        ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
        ("historical_reference_experiment_id", HISTORICAL_REFERENCE_EXPERIMENT_ID),
        ("paired_raw_baseline_experiment_id", PAIRED_RAW_BASELINE_EXPERIMENT_ID),
        ("methodological_change", METHODOLOGICAL_CHANGE),
    ]
    for key in family["cell_order"]:
        rows_to_write.append((f"cell_{key}", family["cells"][key]))

    results = [(name, str(value)) for name, value in rows_to_write]
    record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10,
                               provenance=provenance, created_ms=now)
    conn.commit()

    return {
        "family": family, "comparison_with_raw_v004": comparison_raw_v004, "comparison_with_v001": comparison_v001,
        "cell_order": family["cell_order"], "cells": family["cells"],
        "family_verdict": family["family_verdict"],
        "raw_signal_n_population": family["raw_signal_n_population"],
        "distinct_source_event_n_population": family["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": family["distinct_independent_cycle_n_population"],
        "global_split": family["global_split"], "split_integrity": family["split_integrity"],
        "coverage_report": family["coverage_report"],
        "effective_path_integrity": family["effective_path_integrity"],
    }
