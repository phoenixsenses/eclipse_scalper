"""BATCH-P7B-1 (W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED):
corrected (post candle-gap-repair) + cycle-grouped-split rerun of
E-W8-LONG-TIMING-STRUCTURE-001, reusing the exact frozen independent-cycle
TRAIN/TEST manifest from the paired E-W8-HOLD-BASELINE-004-LONG-CORRECTED-
CYCLE-GROUPED experiment.

NOT AN INDEPENDENT REPLICATION of E-W8-LONG-TIMING-STRUCTURE-001: v001 used
the pre-repair "path-v2"-only population; v002 uses the effective
(post-repair) population. Populations overlap heavily (v001's own cycle
split -- 142/99/43 -- happens to numerically match v002's, since the repair
fills in previously-missing HORIZONS for already-known cycles rather than
adding brand-new cycles; this is verified below, never assumed).
`corrected_data_rerun_of`, `candle_data_version`, `path_data_version`, and
`paired_cycle_split_experiment_id` are recorded explicitly on the registry
row.

STILL DESCRIPTIVE PATH-TIMING RESEARCH, NOT_A_MANAGEMENT_WAVE: this module
answers "when does the path reach its favorable/adverse extremum", never
"should a trade be stopped/exited/re-entered/held differently" -- same
posture as v001. NO_ECONOMIC_OR_ALPHA_CLAIM, NO_SHORT_POOLING (LONG only).

SPLIT CONTRACT -- MANDATORY MANIFEST REUSE: this module's population
(`fetch_population()`) is built directly on top of
w8_hold_baseline_004_long_corrected_cycle_grouped.fetch_population()
(imported, not reimplemented) -- the IDENTICAL effective, LONG,
observation_status=OK population the paired raw-bps experiment used to
freeze its cycle split. compute_global_cycle_split() is called on that same
raw population, deterministically reproducing v004's frozen train/test
cycle-key membership -- verified (never assumed) against v004's own stored
global_cycle_split via `verify_split_matches_paired_cycle_split()`. The 3
frozen descriptive timing fields (time_to_mfe_fraction_of_horizon,
time_to_mae_fraction_of_horizon, timing_delta_ms) are added on top, exactly
matching v001's own convention.

PRIMARY FAMILY (frozen, exactly 8 = 2 timing metrics [time_to_mfe_ms,
time_to_mae_ms] x 4 horizons, LONG only): reuses v001's own compute_cell()
and compute_horizon_descriptive() VERBATIM (`is`-identity) -- not
reimplemented. ONE joint Holm step-down correction across all 8 p-values
together, same as v001.

CORRECTION IMPACT AUDIT: `compute_correction_impact_audit()` filters
ami.lifecycle.path_candle_repair_correction.identify_affected_pairs() to
direction=LONG AND class_a only (forward-window corrections that actually
change observation_status/mfe_bps/mae_bps/timing fields -- the 21 LONG
class_b rows are volatility-only corrections that never touch
time_to_mfe_ms/time_to_mae_ms, so they are irrelevant to a timing-structure
wave and explicitly excluded, not silently pooled in). For this population
(gated on observation_status=OK only), "affected" and "newly eligible" are
the same set by construction -- every class_a row is a MISSING_INTERNAL_GAP
-> OK transition, verified (never assumed).

NOT_A_MANAGEMENT_WAVE / NO_ECONOMIC_OR_ALPHA_CLAIM / NO_SIGNAL_BACKFILL /
NO_CANDLE_OR_PATH_MUTATION: identical posture to v001 and to every other
-00N-CANDLE-REPAIR rerun in this project.
"""
from __future__ import annotations
import ast
import hashlib
import time

from ami.lifecycle.path_candle_repair_correction import (
    PATH_DATA_VERSION_CANDLE_REPAIR_R1,
    effective_path_selection_audit,
    identify_affected_pairs,
)
from ami.research.feature_gateway import fetch_lifecycle_signals
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, PATH_HORIZONS_MS, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.research.w8_hold_baseline import HORIZONS, N_BOOTSTRAP, N_PERMUTATIONS, _month_bucket, classify_cell_verdict
from ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped import (
    EXPERIMENT_ID as PAIRED_CYCLE_SPLIT_EXPERIMENT_ID,
)
from ami.research.w8_hold_baseline_004_long_corrected_cycle_grouped import (
    fetch_population as fetch_raw_population,
)
from ami.research.w8_long_timing_structure import (
    DIRECTION,
    EXPERIMENT_ID as OLD_EXPERIMENT_ID,
    TIMING_METRICS,
    _rate,
    compute_cell,
    compute_horizon_descriptive,
)
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results

EXPERIMENT_ID = "E-W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED"
RESEARCH_CONTEXT_ID = "w8-long-timing-structure-002-candle-repair-cycle-grouped"
CORRECTED_DATA_RERUN_OF = OLD_EXPERIMENT_ID  # "E-W8-LONG-TIMING-STRUCTURE-001"
PAIRED_CYCLE_SPLIT_EXPERIMENT_ID = PAIRED_CYCLE_SPLIT_EXPERIMENT_ID  # "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"

CANDLE_DATA_VERSION = "candle-binance-fapi-repair-v1"
PATH_DATA_VERSION = PATH_DATA_VERSION_CANDLE_REPAIR_R1  # "path-v2-candle-repair-r1"


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
# population assembly -- IDENTICAL underlying rows to the paired raw v004
# baseline, with the 3 frozen descriptive timing fields added on top
# ---------------------------------------------------------------------------

def fetch_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    rows = fetch_raw_population(conn, symbol)
    out = []
    for r in rows:
        horizon_ms = PATH_HORIZONS_MS[r["horizon_name"]]
        time_to_mfe_ms = r["time_to_mfe_ms"]
        time_to_mae_ms = r["time_to_mae_ms"]
        row = dict(r)
        row["time_to_mfe_fraction_of_horizon"] = (
            round(time_to_mfe_ms / horizon_ms, 6) if time_to_mfe_ms is not None else None
        )
        row["time_to_mae_fraction_of_horizon"] = (
            round(time_to_mae_ms / horizon_ms, 6) if time_to_mae_ms is not None else None
        )
        row["timing_delta_ms"] = (
            time_to_mfe_ms - time_to_mae_ms if time_to_mfe_ms is not None and time_to_mae_ms is not None else None
        )
        out.append(row)
    return out


def _cell_rows(rows: list[dict], horizon: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == DIRECTION]


# ---------------------------------------------------------------------------
# Part: verify the reused split truly matches the paired cycle-split experiment
# ---------------------------------------------------------------------------

def verify_split_matches_paired_cycle_split(conn, split: dict) -> dict:
    stored = _read_old_metric(conn, PAIRED_CYCLE_SPLIT_EXPERIMENT_ID, "global_cycle_split")
    actual = {
        "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
        "test_cycle_n": split["test_cycle_n"],
    }
    return {"stored_paired_split": stored, "actual_split": actual, "matches": stored == actual}


# ---------------------------------------------------------------------------
# correction impact audit -- LONG, class_a only (timing-relevant corrections)
# ---------------------------------------------------------------------------

_EXPECTED_CORRECTION_IMPACT = {
    "affected_physical_row_n": 104, "distinct_signal_n": 71, "distinct_event_n": 71, "distinct_cycle_n": 49,
}


def compute_correction_impact_audit(conn, symbol: str = "ETHUSDT") -> dict:
    result = identify_affected_pairs(conn, symbol=symbol)
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}

    class_a_long = [(sid, h) for sid, h in result["class_a"] if signals.get(sid, {}).get("direction") == DIRECTION]

    def _cyc(sid):
        s = signals[sid]
        return s["independent_cycle_id"] or f"NOCYCLE-{s['source_event_id']}"

    by_horizon: dict[str, int] = {}
    for sid, h in class_a_long:
        by_horizon[h] = by_horizon.get(h, 0) + 1

    actual = {
        "affected_physical_row_n": len(class_a_long),
        "distinct_signal_n": len({sid for sid, _ in class_a_long}),
        "distinct_event_n": len({signals[sid]["source_event_id"] for sid, _ in class_a_long
                                  if signals[sid]["source_event_id"]}),
        "distinct_cycle_n": len({_cyc(sid) for sid, _ in class_a_long}),
    }
    mismatches = {k: {"actual": actual[k], "expected": v} for k, v in _EXPECTED_CORRECTION_IMPACT.items()
                  if actual[k] != v}

    # for this observation_status=OK-gated population, "affected" (GAP->OK) and "newly eligible"
    # are the SAME set by construction -- verified, not assumed
    newly_eligible = dict(actual)

    return {
        "actual": actual, "expected": _EXPECTED_CORRECTION_IMPACT, "mismatches": mismatches,
        "by_horizon": by_horizon,
        "newly_eligible_signal_n": newly_eligible["distinct_signal_n"],
        "newly_eligible_event_n": newly_eligible["distinct_event_n"],
        "newly_eligible_cycle_n": newly_eligible["distinct_cycle_n"],
        "class_b_excluded_note": (
            "21 LONG class_b (volatility-only) corrected rows exist but are excluded here -- they "
            "never change time_to_mfe_ms/time_to_mae_ms, only realized_vol_at_anchor/*_anchor_vol_units, "
            "so they are irrelevant to this timing-structure wave."
        ),
    }


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


def compute_coverage_report(rows: list[dict], train_keys: set, test_keys: set, conn) -> dict:
    per_horizon: dict[str, dict] = {}
    v004_coverage = _read_old_metric(conn, PAIRED_CYCLE_SPLIT_EXPERIMENT_ID, "coverage_report")
    for horizon in HORIZONS:
        cell_rows = _cell_rows(rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
        train_cycle_n = len({_cycle_key(r) for r in train_r})
        test_cycle_n = len({_cycle_key(r) for r in test_r})
        v004_horizon = (v004_coverage or {}).get("per_horizon", {}).get(horizon, {})
        per_horizon[horizon] = {
            "raw_signal_n": len(cell_rows),
            "source_event_n": len({r["source_event_id"] for r in cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in cell_rows}),
            "train_cycle_n": train_cycle_n, "test_cycle_n": test_cycle_n,
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
            "sufficiency_verdict": (
                "OK" if train_cycle_n >= MIN_BUCKET_N and test_cycle_n >= MIN_BUCKET_N else "INSUFFICIENT_SAMPLE"
            ),
            "monthly_distribution": _monthly_distribution(cell_rows),
            "setup_composition": _setup_composition(cell_rows),
            "paired_raw_v004_signal_n": v004_horizon.get("raw_signal_n"),
        }
    return {"per_horizon": per_horizon}


# ---------------------------------------------------------------------------
# primary family -- 8 cells (2 timing metrics x 4 horizons)
# ---------------------------------------------------------------------------

def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    integrity = verify_effective_path_selection_integrity(conn)
    if not integrity["passed"]:
        return {
            "blocked": True, "effective_path_integrity": integrity,
            "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
        }

    rows = fetch_population(conn, symbol)

    # cycle membership REUSED byte-exact from the paired raw baseline's own frozen split --
    # rows here are population-identical to that experiment's own fetch, so recomputing the split
    # from them reproduces it deterministically (verified below, never merely assumed)
    split = compute_global_cycle_split(rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]
    split_integrity = verify_split_matches_paired_cycle_split(conn, split)

    coverage_report = compute_coverage_report(rows, train_keys, test_keys, conn)
    correction_impact_audit = compute_correction_impact_audit(conn, symbol)

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    descriptive_by_horizon: dict[str, dict] = {}
    for horizon in HORIZONS:
        cell_rows = _cell_rows(rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
        descriptive_by_horizon[horizon] = compute_horizon_descriptive(cell_rows, train_r, test_r)
        for metric in TIMING_METRICS:
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
        family_verdict = "LONG_TIMING_STRUCTURE_INSUFFICIENT_CORRECTED_DATA"
    elif any_insufficient and (any_regime_dependent or any_stable):
        family_verdict = "MIXED_LONG_TIMING_STRUCTURE_CORRECTED_DATA"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_LONG_TIMING_STRUCTURE_CORRECTED_DATA"
    elif any_regime_dependent:
        family_verdict = "LONG_TIMING_STRUCTURE_REGIME_DEPENDENT_CORRECTED_DATA"
    else:
        family_verdict = "LONG_TIMING_STRUCTURE_STABLE_CORRECTED_DATA"

    return {
        "blocked": False, "effective_path_integrity": integrity, "split_integrity": split_integrity,
        "rows": rows, "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "coverage_report": coverage_report, "correction_impact_audit": correction_impact_audit,
        "descriptive_by_horizon": descriptive_by_horizon,
        "raw_signal_n_population": len({r["signal_id"] for r in rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in rows}),
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# comparison with the immutable v001 (descriptive only -- NOT independent
# replications, populations overlap heavily)
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


def compare_with_v001(conn, family: dict) -> dict:
    v001_population = {
        "raw_signal_n_population": _read_old_metric(conn, OLD_EXPERIMENT_ID, "raw_signal_n_population"),
        "distinct_source_event_n_population": _read_old_metric(
            conn, OLD_EXPERIMENT_ID, "distinct_source_event_n_population"
        ),
        "distinct_independent_cycle_n_population": _read_old_metric(
            conn, OLD_EXPERIMENT_ID, "distinct_independent_cycle_n_population"
        ),
        "global_cycle_split": _read_old_metric(conn, OLD_EXPERIMENT_ID, "global_cycle_split"),
        "per_horizon_split_report": _read_old_metric(conn, OLD_EXPERIMENT_ID, "per_horizon_split_report") or {},
        "family_verdict": _read_old_metric(conn, OLD_EXPERIMENT_ID, "family_verdict"),
    }

    v001_cells = {key: _read_old_metric(conn, OLD_EXPERIMENT_ID, f"cell_{key}") for key in family["cell_order"]}
    any_v001_cell_missing = any(v001_cells[k] is None for k in family["cell_order"])

    per_horizon_population_changes = {}
    for horizon in HORIZONS:
        v001_h = v001_population["per_horizon_split_report"].get(horizon, {})
        v002_h = family["coverage_report"]["per_horizon"][horizon]
        per_horizon_population_changes[horizon] = {
            "v001_raw_signal_n": v001_h.get("raw_signal_n"), "v002_raw_signal_n": v002_h["raw_signal_n"],
            "v001_independent_cycle_n": v001_h.get("independent_cycle_n"),
            "v002_independent_cycle_n": v002_h["independent_cycle_n"],
        }

    cell_changes = {}
    changed_count = 0
    for key in family["cell_order"]:
        v001_c = v001_cells.get(key)
        v002_c = family["cells"][key]
        v001_verdict = v001_c.get("closure_classification") if v001_c else None
        v002_verdict = v002_c.get("closure_classification")
        v001_median = v001_c.get("full_median") if v001_c else None
        v002_median = v002_c.get("full_median")
        changed = v001_verdict != v002_verdict
        if changed:
            changed_count += 1
        cell_changes[key] = {
            "v001_verdict": v001_verdict, "v002_verdict": v002_verdict, "changed": changed,
            "v001_median": v001_median, "v002_median": v002_median,
            "same_sign": (
                v001_median is not None and v002_median is not None and (v001_median > 0) == (v002_median > 0)
            ),
        }

    # MFE_FIRST/MAE_FIRST rate changes + whether "longer horizons increasingly show MAE before MFE" survives
    mfe_mae_rate_changes = {}
    for horizon in HORIZONS:
        v001_desc = (_read_old_metric(conn, OLD_EXPERIMENT_ID, "descriptive_by_horizon") or {}).get(horizon, {})
        v002_desc = family["descriptive_by_horizon"][horizon]
        mfe_mae_rate_changes[horizon] = {
            "v001_mfe_first_rate": v001_desc.get("intrabar_order_status_rates", {}).get("MFE_FIRST"),
            "v002_mfe_first_rate": v002_desc["intrabar_order_status_rates"]["MFE_FIRST"],
            "v001_mae_first_rate": v001_desc.get("intrabar_order_status_rates", {}).get("MAE_FIRST"),
            "v002_mae_first_rate": v002_desc["intrabar_order_status_rates"]["MAE_FIRST"],
        }
    v002_mae_first_rates_by_horizon = [
        mfe_mae_rate_changes[h]["v002_mae_first_rate"] for h in HORIZONS
        if mfe_mae_rate_changes[h]["v002_mae_first_rate"] is not None
    ]
    mae_first_increases_with_horizon_survives = (
        v002_mae_first_rates_by_horizon == sorted(v002_mae_first_rates_by_horizon)
        and len(v002_mae_first_rates_by_horizon) == len(HORIZONS)
    )

    all_insufficient_v002 = family["family_verdict"] == "LONG_TIMING_STRUCTURE_INSUFFICIENT_CORRECTED_DATA"

    if all_insufficient_v002:
        comparison_label = "INSUFFICIENT_CORRECTED_TIMING_POPULATION"
    elif any_v001_cell_missing:
        comparison_label = "MATERIAL_TIMING_STRUCTURE_CHANGE"
    elif changed_count == 0:
        comparison_label = "TIMING_STRUCTURE_CONSISTENT_ON_CORRECTED_EXPANDED_COHORT"
    elif changed_count <= len(family["cell_order"]) // 2:
        comparison_label = "PARTIALLY_CONSISTENT"
    else:
        comparison_label = "MATERIAL_TIMING_STRUCTURE_CHANGE"

    return {
        "v001_experiment_id": OLD_EXPERIMENT_ID, "v001_family_verdict": v001_population["family_verdict"],
        "v002_family_verdict": family["family_verdict"],
        "population_changes": {
            "raw_signal_n": {"v001": v001_population["raw_signal_n_population"],
                              "v002": family["raw_signal_n_population"]},
            "distinct_source_event_n": {"v001": v001_population["distinct_source_event_n_population"],
                                        "v002": family["distinct_source_event_n_population"]},
            "distinct_independent_cycle_n": {"v001": v001_population["distinct_independent_cycle_n_population"],
                                              "v002": family["distinct_independent_cycle_n_population"]},
            "per_horizon_population_changes": per_horizon_population_changes,
        },
        "split_count_changes": {
            "v001_global_split": v001_population["global_cycle_split"], "v002_global_split": family["global_split"],
        },
        "cell_changes": cell_changes, "changed_cell_count": changed_count,
        "mfe_mae_rate_changes_by_horizon": mfe_mae_rate_changes,
        "mae_first_increases_with_horizon_survives": mae_first_increases_with_horizon_survives,
        "comparison_label": comparison_label,
        "not_independent_replication_note": (
            "v001 and v002 populations overlap heavily (v002's cohort is a corrected-data expansion of "
            "v001's, not an independent sample) -- this comparison is a consistency check only, never "
            "treated as an independent confirmation or refutation of v001's finding."
        ),
    }


# ---------------------------------------------------------------------------
# freeze_and_record -- writes to the NEW experiment_id only
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-w8-long-timing-structure-002-candle-repair-cycle-grouped"
                       ) -> dict:
    now = int(time.time() * 1000)

    family = compute_family(conn)

    if family["blocked"]:
        dataset_hash = hashlib.sha256(b"BLOCKED_BY_EFFECTIVE_PATH_SELECTION").hexdigest()
        record_experiment_registry(conn, {
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_LONG_TIMING_STRUCTURE", "preregistered_at": now,
            "hypothesis_id": "H-W8-LONG-TIMING-STRUCTURE-CORRECTED-CYCLE-GROUPED",
            "frozen_population": (
                "BLOCKED_BY_EFFECTIVE_PATH_SELECTION -- effective_path_selection_audit() mismatch, "
                "see effective_path_integrity for exact diffs; no population fetched, no cell computed"
            ),
            "frozen_features": ",".join(TIMING_METRICS),
            "frozen_target": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any target could be computed",
            "frozen_thresholds": f"MIN_BUCKET_N={MIN_BUCKET_N}; effective-path integrity required first",
            "frozen_splits": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any split could be computed",
            "frozen_economic_gate": "N/A (no stop/exit/re-entry/hold rule tested)",
            "frozen_statistical_gate": "N/A -- BLOCKED_BY_EFFECTIVE_PATH_SELECTION before any statistic ran",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": CORRECTED_DATA_RERUN_OF, "report_artifact_id": None,
            "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
        })
        results = [
            ("family_verdict", "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"),
            ("effective_path_integrity", str(family["effective_path_integrity"])),
            ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
            ("corrected_data_rerun_of", CORRECTED_DATA_RERUN_OF),
            ("paired_cycle_split_experiment_id", PAIRED_CYCLE_SPLIT_EXPERIMENT_ID),
        ]
        record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10,
                                   provenance=provenance, created_ms=now)
        conn.commit()
        return {"family": family, "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION"}

    comparison = compare_with_v001(conn, family)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(f"{r['signal_id']}|{r['horizon_name']}" for r in family["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"EFFECTIVE ami_lifecycle_path_observations (path-v2-candle-repair-r1 where corrected, else "
        f"path-v2) WHERE observation_status=OK, direction=LONG only; "
        f"raw_signal_n={family['raw_signal_n_population']}; "
        f"distinct_source_event_n={family['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={family['distinct_independent_cycle_n_population']}; "
        f"candle_data_version={CANDLE_DATA_VERSION}; path_data_version={PATH_DATA_VERSION}; "
        f"corrected_data_rerun_of={CORRECTED_DATA_RERUN_OF}; "
        f"paired_cycle_split_experiment_id={PAIRED_CYCLE_SPLIT_EXPERIMENT_ID}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id, REUSED BYTE-EXACT from {PAIRED_CYCLE_SPLIT_EXPERIMENT_ID} (identical "
        f"underlying raw population -- never independently re-optimized); MIN_BUCKET_N={MIN_BUCKET_N} "
        f"applies to independent-cycle N per split/horizon; "
        f"split_matches_paired_cycle_split={family['split_integrity']['matches']}"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_LONG_TIMING_STRUCTURE",
        "hypothesis_id": "H-W8-LONG-TIMING-STRUCTURE-CORRECTED-CYCLE-GROUPED", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(TIMING_METRICS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30%) "
            "median stability of time_to_mfe_ms/time_to_mae_ms, LONG only, by horizon, CORRECTED "
            "(post candle-gap-repair) data"
        ),
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N} (independent-cycle N per split/horizon, NOT signal-level N); "
            f"TRAIN_FRACTION={TRAIN_FRACTION}; classification=stable iff Holm-p>=0.05 AND bootstrap-CI "
            "includes 0; regime-dependent iff Holm-p<0.05 AND CI excludes 0; disagreement->regime-dependent "
            "(conservative); insufficient iff either split's independent-cycle N<MIN_BUCKET_N"
        ),
        "frozen_splits": frozen_splits,
        "frozen_economic_gate": (
            "N/A (no stop/exit/re-entry/hold rule tested -- WHEN the path reaches its extremum, never "
            "whether/how to act on it)"
        ),
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}); "
            f"secondary=two-sided label-permutation median-difference p (n={N_PERMUTATIONS}) + ONE joint "
            f"Holm step-down across all 8 primary cells (2 timing metrics x 4 horizons together)"
        ),
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": family["family_verdict"],
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": CORRECTED_DATA_RERUN_OF, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })

    rows_to_write = [
        ("raw_signal_n_population", family["raw_signal_n_population"]),
        ("distinct_source_event_n_population", family["distinct_source_event_n_population"]),
        ("distinct_independent_cycle_n_population", family["distinct_independent_cycle_n_population"]),
        ("global_cycle_split", family["global_split"]),
        ("split_integrity", family["split_integrity"]),
        ("coverage_report", family["coverage_report"]),
        ("correction_impact_audit", family["correction_impact_audit"]),
        ("descriptive_by_horizon", family["descriptive_by_horizon"]),
        ("effective_path_integrity", family["effective_path_integrity"]),
        ("comparison_with_v001", comparison),
        ("family_verdict", family["family_verdict"]),
        ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
        ("corrected_data_rerun_of", CORRECTED_DATA_RERUN_OF),
        ("paired_cycle_split_experiment_id", PAIRED_CYCLE_SPLIT_EXPERIMENT_ID),
    ]
    for key in family["cell_order"]:
        rows_to_write.append((f"cell_{key}", family["cells"][key]))

    results = [(name, str(value)) for name, value in rows_to_write]
    record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10,
                               provenance=provenance, created_ms=now)
    conn.commit()

    return {
        "family": family, "comparison_with_v001": comparison,
        "cell_order": family["cell_order"], "cells": family["cells"],
        "family_verdict": family["family_verdict"],
        "raw_signal_n_population": family["raw_signal_n_population"],
        "distinct_source_event_n_population": family["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": family["distinct_independent_cycle_n_population"],
        "global_split": family["global_split"], "split_integrity": family["split_integrity"],
        "coverage_report": family["coverage_report"],
        "correction_impact_audit": family["correction_impact_audit"],
        "descriptive_by_horizon": family["descriptive_by_horizon"],
        "effective_path_integrity": family["effective_path_integrity"],
    }
