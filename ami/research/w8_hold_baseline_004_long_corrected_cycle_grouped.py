"""BATCH-P7B-1 (W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED): corrected
(post candle-gap-repair) + cycle-grouped-split rerun of the LONG-only raw
mfe_bps/mae_bps portion of E-W8-HOLD-BASELINE-001.

NOT AN INDEPENDENT REPLICATION OF E-W8-HOLD-BASELINE-001 (operator's own
explicit instruction, restated here so it cannot be silently dropped by a
future edit): v001 used the pre-repair "path-v2"-only population AND a naive
signal-level chronological 70/30 split. This experiment changes BOTH the
data (effective, post-repair paths) AND the split methodology (independent-
cycle-grouped). It is therefore a methodologically-distinct rerun, never
described as a byte-for-byte reproduction or an independent confirmation of
v001's finding. `historical_reference_experiment_id`, `candle_data_version`,
`path_data_version`, and `methodological_change` are recorded explicitly on
the registry row for exactly this reason.

IMMUTABLE PRECEDENT: E-W8-HOLD-BASELINE-001 is NEVER touched by this module
-- a completely separate experiment_id, registry row, and result set (same
discipline as every prior -00N-CANDLE-REPAIR/-CORRECTED rerun in this
project).

EFFECTIVE PATH SELECTION (mandatory precondition, verified before any cell is
computed): `verify_effective_path_selection_integrity()` requires
physical_row_count_total=1466, duplicate_physical_pair_n=170,
effective_row_count=1296, duplicate_effective_pair_n=0 (exactly one
effective row per (signal_id, horizon_name)). If any of these fail, this
module computes NO population and NO cell -- scientific_verdict is frozen as
"BLOCKED_BY_EFFECTIVE_PATH_SELECTION".

CYCLE-GROUPED SPLIT: reuses ami.research.w8_short_expanded_baseline's
generic (not SHORT-specific) split machinery VERBATIM --
compute_global_cycle_split()/split_rows_by_cycle_keys()/
assert_zero_cycle_straddling()/_cycle_key() operate on plain row dicts keyed
by independent_cycle_id/source_event_id/signal_birth_ts, with no
SHORT-specific logic baked in (same reuse precedent as
w8_long_timing_structure.py/w8_long_nested_path_accumulation.py). Not
reimplemented here. compute_cell() is reused VERBATIM from
w8_short_expanded_baseline too -- MIN_BUCKET_N=20 applies to
INDEPENDENT-CYCLE N in each split, never signal-level N.

PRIMARY FAMILY (frozen, exactly 8 = 2 raw-bps metrics [mfe_bps, mae_bps] x 4
horizons, LONG only): NO volatility-normalized metrics in this batch (that is
explicitly a separate, subsequent immutable experiment per operator
instruction) -- NO_SHORT_POOLING (LONG only, matches
BATCH-W8-SHORT-EXPANDED-BASELINE-003-CANDLE-REPAIR's own finding that SHORT
remains an accumulation/new-event-family branch). ONE joint Holm step-down
correction across all 8 p-values together.

PRE-OUTCOME COVERAGE REPORT (frozen, computed BEFORE any metric value is
read): per-horizon raw signal/source-event/independent-cycle N, TRAIN/TEST
cycle N, cycle-straddling violations, sufficiency verdict, monthly
distribution (by signal_birth_ts), and setup_id composition -- plus global
cycle-split totals, signals sharing an independent cycle, and source events
carrying multiple LONG signals. All of these are structural/descriptive
counts, never an outcome value.

COMPARISON WITH V001 (descriptive only -- NOT an independent replication):
reads E-W8-HOLD-BASELINE-001's ALREADY-FROZEN, byte-immutable stored LONG
cells (never recomputes them) and reports population/split/verdict/median
changes. Never claims agreement or disagreement is forced.

NOT_A_MANAGEMENT_WAVE: no stop/exit/partial-exit/time-stop/re-entry/
cancellation rule anywhere in this module. NO_ECONOMIC_CLAIM: no PnL/alpha/
win-rate claim is made or implied by any verdict this module can produce.
NO_SIGNAL_BACKFILL / NO_CANDLE_OR_PATH_MUTATION / NO_MATCHED_CONTROL_
RECONSTRUCTION: this module never touches, reopens, or fabricates the
matched-control direction-assignment question left unresolved by
E-W8-HOLD-BASELINE-001's own negative control.
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
from ami.research.w8_hold_baseline import (
    EXPERIMENT_ID as OLD_EXPERIMENT_ID,
    HORIZONS,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    _median,
    _month_bucket,
    _quantile,
    classify_cell_verdict,
)
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_cell,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results

EXPERIMENT_ID = "E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED"
RESEARCH_CONTEXT_ID = "w8-hold-baseline-004-long-corrected-cycle-grouped"
HISTORICAL_REFERENCE_EXPERIMENT_ID = OLD_EXPERIMENT_ID  # "E-W8-HOLD-BASELINE-001"

CANDLE_DATA_VERSION = "candle-binance-fapi-repair-v1"
PATH_DATA_VERSION = PATH_DATA_VERSION_CANDLE_REPAIR_R1  # "path-v2-candle-repair-r1"
METHODOLOGICAL_CHANGE = "SIGNAL_LEVEL_SPLIT_TO_INDEPENDENT_CYCLE_GROUPED_SPLIT"

DIRECTION = "LONG"
METRICS = ("mfe_bps", "mae_bps")  # raw-bps only -- vol-normalized is a separate, subsequent experiment


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
# population assembly -- effective selector, LONG only, observation_status=OK
# ---------------------------------------------------------------------------

def fetch_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
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


def _cell_rows(rows: list[dict], horizon: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == DIRECTION]


def _anchor_profile(rows: list[dict]) -> list[dict]:
    """One row per DISTINCT signal (not per signal x horizon)."""
    by_signal: dict[str, dict] = {}
    for r in rows:
        by_signal.setdefault(r["signal_id"], r)
    return list(by_signal.values())


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


def compute_coverage_report(rows: list[dict], train_keys: set, test_keys: set) -> dict:
    """Per-horizon coverage (N/sufficiency/monthly/setup) PLUS global cycle-
    split structural counts -- computed and returned entirely independently
    of any MFE/MAE value (no metric is read anywhere in this function)."""
    per_horizon: dict[str, dict] = {}
    for horizon in HORIZONS:
        cell_rows = _cell_rows(rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
        train_cycle_n = len({_cycle_key(r) for r in train_r})
        test_cycle_n = len({_cycle_key(r) for r in test_r})
        per_horizon[horizon] = {
            "raw_signal_n": len(cell_rows),
            "source_event_n": len({r["source_event_id"] for r in cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in cell_rows}),
            "train_cycle_n": train_cycle_n,
            "test_cycle_n": test_cycle_n,
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
            "sufficiency_verdict": (
                "OK" if train_cycle_n >= MIN_BUCKET_N and test_cycle_n >= MIN_BUCKET_N else "INSUFFICIENT_SAMPLE"
            ),
            "monthly_distribution": _monthly_distribution(cell_rows),
            "setup_composition": _setup_composition(cell_rows),
        }

    anchor_rows = _anchor_profile(rows)
    by_cycle: dict[str, set] = {}
    for r in anchor_rows:
        by_cycle.setdefault(_cycle_key(r), set()).add(r["signal_id"])
    cycles_with_multiple_signals_n = sum(1 for sigs in by_cycle.values() if len(sigs) > 1)
    signals_in_multi_signal_cycles_n = sum(len(sigs) for sigs in by_cycle.values() if len(sigs) > 1)

    by_event: dict[str, set] = {}
    for r in anchor_rows:
        if r["source_event_id"] is not None:
            by_event.setdefault(r["source_event_id"], set()).add(r["signal_id"])
    events_with_multiple_long_signals_n = sum(1 for sigs in by_event.values() if len(sigs) > 1)

    return {
        "per_horizon": per_horizon,
        "signals_sharing_independent_cycle": {
            "cycles_with_multiple_signals_n": cycles_with_multiple_signals_n,
            "signals_in_multi_signal_cycles_n": signals_in_multi_signal_cycles_n,
        },
        "source_events_carrying_multiple_long_signals_n": events_with_multiple_long_signals_n,
    }


# ---------------------------------------------------------------------------
# primary family -- 8 cells (2 raw-bps metrics x 4 horizons), cycle-grouped split
# ---------------------------------------------------------------------------

def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    integrity = verify_effective_path_selection_integrity(conn)
    if not integrity["passed"]:
        return {
            "blocked": True, "effective_path_integrity": integrity,
            "family_verdict": "BLOCKED_BY_EFFECTIVE_PATH_SELECTION",
        }

    rows = fetch_population(conn, symbol)

    # cycle membership recomputed fresh from the corrected population -- never fed a prior list
    split = compute_global_cycle_split(rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]

    coverage_report = compute_coverage_report(rows, train_keys, test_keys)

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    for horizon in HORIZONS:
        cell_rows = _cell_rows(rows, horizon)
        for metric in METRICS:
            key = f"{metric}|{horizon}"
            cell_order.append(key)
            cells[key] = compute_cell(cell_rows, metric, train_keys, test_keys)

    assert len(cell_order) == 8, f"primary family must be exactly 8 cells, got {len(cell_order)}"

    # ONE joint Holm correction across all 8 p-values -- holm_adjust() only performs work over the
    # non-None subset; if every cell were INSUFFICIENT_SAMPLE it would do no work at all
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
        family_verdict = "LONG_RAW_HOLD_BASELINE_INSUFFICIENT"
    elif any_insufficient and (any_regime_dependent or any_stable):
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC_CORRECTED_CYCLE_GROUPED"
    elif any_regime_dependent:
        family_verdict = "LONG_RAW_HOLD_BASELINE_REGIME_DEPENDENT_CORRECTED_CYCLE_GROUPED"
    else:
        family_verdict = "LONG_RAW_HOLD_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED"

    return {
        "blocked": False, "effective_path_integrity": integrity,
        "rows": rows, "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "coverage_report": coverage_report,
        "raw_signal_n_population": len({r["signal_id"] for r in rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in rows}),
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# comparison with the immutable v001 (descriptive only -- NOT an independent
# replication; population, candle data, AND split methodology all changed)
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
    """NOTE: E-W8-HOLD-BASELINE-001 stored its population totals
    (raw_signal_n_population/distinct_source_event_n_population/
    distinct_independent_cycle_n_population) at the FAMILY level across BOTH
    LONG and SHORT combined (v001 pooled both directions in one 16-cell
    family) -- comparing those combined totals directly against this LONG-
    only experiment's population would be a like-for-unlike comparison and
    is deliberately NOT done. Per-horizon LONG-only population is instead
    derived from each of v001's own LONG cells (raw_signal_n/
    distinct_source_event_n/distinct_independent_cycle_n are already
    direction-specific at the cell level -- mfe_bps and mae_bps share the
    identical row population per horizon, so the mfe_bps cell is used as the
    representative)."""
    v001_cells = {}
    for key in family["cell_order"]:
        metric, horizon = key.split("|")
        v001_cells[key] = _read_old_metric(conn, HISTORICAL_REFERENCE_EXPERIMENT_ID, f"cell_{metric}|{horizon}|LONG")

    per_horizon_population_changes = {}
    for horizon in HORIZONS:
        v001_cell = v001_cells.get(f"mfe_bps|{horizon}")
        v004_cov = family["coverage_report"]["per_horizon"][horizon]
        per_horizon_population_changes[horizon] = {
            "v001_raw_signal_n": v001_cell.get("raw_signal_n") if v001_cell else None,
            "v004_raw_signal_n": v004_cov["raw_signal_n"],
            "v001_distinct_source_event_n": v001_cell.get("distinct_source_event_n") if v001_cell else None,
            "v004_distinct_source_event_n": v004_cov["source_event_n"],
            "v001_distinct_independent_cycle_n": (
                v001_cell.get("distinct_independent_cycle_n") if v001_cell else None
            ),
            "v004_distinct_independent_cycle_n": v004_cov["independent_cycle_n"],
        }

    population_changes = {
        "not_comparable_note": (
            "v001's stored raw_signal_n_population/distinct_source_event_n_population/"
            "distinct_independent_cycle_n_population are combined LONG+SHORT family totals, not "
            "LONG-only -- deliberately NOT compared here to avoid a like-for-unlike figure. See "
            "per_horizon_population_changes for the correct LONG-only, per-horizon comparison "
            "(derived from v001's own LONG cells)."
        ),
        "per_horizon_population_changes": per_horizon_population_changes,
    }

    any_v001_cell_missing = any(v001_cells[k] is None for k in family["cell_order"])

    cell_changes = {}
    changed_count = 0
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
        changed = v001_verdict != v004_verdict
        if changed:
            changed_count += 1
        cell_changes[key] = {
            "v001_verdict": v001_verdict, "v004_verdict": v004_verdict, "changed": changed,
            "v001_median": v001_median, "v004_median": v004_median, "same_sign": same_sign,
            "median_delta": (round(v004_median - v001_median, 4)
                              if v001_median is not None and v004_median is not None else None),
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
        "population_changes": population_changes,
        "split_method_difference": (
            "v001: naive signal-level chronological 70/30 split. v004: independent-cycle-grouped "
            "chronological 70/30 split (one cycle assigned to exactly one side, zero straddling)."
        ),
        "candle_repair_caused_population_growth_by_horizon": {
            horizon: (
                (per_horizon_population_changes[horizon]["v004_raw_signal_n"]
                 - per_horizon_population_changes[horizon]["v001_raw_signal_n"])
                if per_horizon_population_changes[horizon]["v001_raw_signal_n"] is not None else None
            )
            for horizon in HORIZONS
        },
        "cell_changes": cell_changes, "changed_cell_count": changed_count,
        "comparison_label": comparison_label,
        "not_independent_replication_note": (
            "v001 and v004 populations overlap heavily but are not identical (candle-repair-corrected "
            "rows added), AND the split methodology differs (signal-level vs cycle-grouped) -- this "
            "comparison is a consistency check only, never treated as an independent confirmation or "
            "refutation of v001's finding. Agreement is reported, never forced."
        ),
    }


# ---------------------------------------------------------------------------
# freeze_and_record -- writes to the NEW experiment_id only
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-w8-hold-baseline-004-long-corrected-cycle-grouped") -> dict:
    now = int(time.time() * 1000)

    family = compute_family(conn)

    if family["blocked"]:
        dataset_hash = hashlib.sha256(b"BLOCKED_BY_EFFECTIVE_PATH_SELECTION").hexdigest()
        record_experiment_registry(conn, {
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_HOLD_BASELINE",
            "hypothesis_id": "H-W8-HOLD-BASELINE-LONG-CORRECTED-CYCLE-GROUPED", "preregistered_at": now,
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
            ("methodological_change", METHODOLOGICAL_CHANGE),
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
        f"historical_reference_experiment_id={HISTORICAL_REFERENCE_EXPERIMENT_ID}; "
        f"methodological_change={METHODOLOGICAL_CHANGE}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id (never signal-level, unlike v001): cycle order key=earliest "
        f"signal_birth_ts among all of that cycle's eligible LONG rows; MIN_BUCKET_N={MIN_BUCKET_N} "
        f"applies to independent-cycle N per split/cell"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_HOLD_BASELINE",
        "hypothesis_id": "H-W8-HOLD-BASELINE-LONG-CORRECTED-CYCLE-GROUPED", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(METRICS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30%) "
            "median stability of mfe_bps/mae_bps, LONG only, by horizon, CORRECTED (post "
            "candle-gap-repair) data"
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
            f"Holm step-down across all 8 primary cells (2 raw-bps metrics x 4 horizons together)"
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
        ("coverage_report", family["coverage_report"]),
        ("effective_path_integrity", family["effective_path_integrity"]),
        ("comparison_with_v001", comparison),
        ("family_verdict", family["family_verdict"]),
        ("candle_data_version", CANDLE_DATA_VERSION), ("path_data_version", PATH_DATA_VERSION),
        ("historical_reference_experiment_id", HISTORICAL_REFERENCE_EXPERIMENT_ID),
        ("methodological_change", METHODOLOGICAL_CHANGE),
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
        "global_split": family["global_split"],
        "coverage_report": family["coverage_report"],
        "effective_path_integrity": family["effective_path_integrity"],
    }
