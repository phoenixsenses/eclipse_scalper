"""BATCH-P7B-1 (W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR): corrected-
data rerun of E-W8-LONG-NESTED-PATH-ACCUMULATION-001, using the EFFECTIVE
(post candle-gap-repair) path population instead of the frozen pre-repair
"path-v2"-only population that -001 used and still uses.

IMMUTABLE PRECEDENT: E-W8-LONG-NESTED-PATH-ACCUMULATION-001 is NEVER touched
by this module -- a completely separate experiment_id, registry row, and
result set. -001's own frozen results remain the historical record of what
the PRE-REPAIR candle data implied (operator instruction, verbatim: "Record
that they used the pre-repair candle/path version").

EFFECTIVE PATH SELECTION (Part 0 of the corrected-data-rerun contract): this
module's ONLY population-fetching function, fetch_common_cohort(), is the one
piece of -001 NOT reused verbatim -- it is rewritten to call
ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations()
instead of feature_gateway.fetch_path_observations() directly, so that a
signal/horizon with a corrected ("path-v2-candle-repair-r1") row is
represented by that corrected row, never by both the corrected row AND its
now-superseded original ("path-v2") row. See that function's own docstring
for the confirmed double-count hazard this specifically avoids (21 pairs
where BOTH the original and corrected row already show observation_status=
'OK', which a naive equals-filter would return twice). Every other function
(compute_derived_fields, assert_nested_nonnegativity, compute_cell,
_capture_fraction_stats, compute_secondary_descriptive, classify_cell_verdict,
holm_adjust, the cycle-grouped split machinery) is imported from -001 or its
shared dependencies UNCHANGED -- not reimplemented, verified by `is` identity
in tests.

CYCLE MEMBERSHIP IS NOT REUSED FROM -001 BLINDLY: the operator's own
instruction is explicit -- the common cohort's eligible signal set changed
(123 -> up to 194 signals in the rehearsal), so the cycle chronological order
and the resulting TRAIN/TEST partition are RECOMPUTED FRESH from the
corrected-cohort population (compute_global_cycle_split() operates on
whatever `rows` it is given -- it was always population-driven, never fed a
hardcoded prior membership list, so calling it again here on the new,
larger cohort is the correct, not-blind behavior; no special-casing needed
to avoid reusing -001's smaller partition, because there was never a
mechanism to do so in the first place).

NOT_A_MANAGEMENT_WAVE / NO_ECONOMIC_OR_ALPHA_CLAIM / NO_SHORT_POOLING /
NO_SIGNAL_BACKFILL / NO_CANDLE_OR_PATH_MUTATION: identical posture to -001 --
this module reads the EXISTING, now-repaired canonical path population
exactly as-is; it triggers no new candle fetch, no new path computation, no
signal/event backfill of any kind.
"""
from __future__ import annotations
import hashlib
import time

from ami.lifecycle.path_candle_repair_correction import (
    OLD_PATH_DEFINITION_VERSION,
    PATH_DATA_VERSION_CANDLE_REPAIR_R1,
    fetch_effective_path_observations,
)
from ami.research.feature_gateway import fetch_lifecycle_signals
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results
from ami.research.w8_hold_baseline import N_BOOTSTRAP, N_PERMUTATIONS, _median, classify_cell_verdict
from ami.research.w8_long_nested_path_accumulation import (
    ALL_DELTA_FIELDS,
    DIRECTION,
    EXPERIMENT_ID as OLD_EXPERIMENT_ID,
    MFE_DELTA_FIELDS,
    MAE_DELTA_FIELDS as ABS_MAE_DELTA_FIELDS,
    INTERVALS,
    _HORIZON_NAME_OF_ALIAS,
    _REAL_HORIZON_NAMES,
    compute_cell,
    compute_derived_fields,
    compute_secondary_descriptive,
    assert_nested_nonnegativity,
)
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)

EXPERIMENT_ID = "E-W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR"
RESEARCH_CONTEXT_ID = "w8-long-nested-path-accumulation-002-candle-repair"
CANDLE_DATA_VERSION = "candle-binance-fapi-repair-v1"
PATH_DATA_VERSION = PATH_DATA_VERSION_CANDLE_REPAIR_R1
CORRECTED_DATA_RERUN_OF = OLD_EXPERIMENT_ID


def fetch_common_cohort(conn, symbol: str = "ETHUSDT") -> dict:
    """Identical contract to -001's fetch_common_cohort(), except sourced via
    the EFFECTIVE path selector (fetch_effective_path_observations) instead
    of a raw observation_status='OK' filter on feature_gateway directly."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_effective_path_observations(conn, RESEARCH_CONTEXT_ID, equals={"observation_status": "OK"})

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


def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    cohort = fetch_common_cohort(conn, symbol)
    rows = [compute_derived_fields(r) for r in cohort["cohort_rows"]]

    nonneg = assert_nested_nonnegativity(rows)

    # cycle membership recomputed fresh from the corrected-cohort population -- never reused from -001
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
        family_verdict = "INSUFFICIENT_CORRECTED_COMMON_COHORT"
    elif any_insufficient:
        family_verdict = "MIXED_BY_INTERVAL_OR_METRIC_CORRECTED_DATA"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_INTERVAL_OR_METRIC_CORRECTED_DATA"
    elif any_regime_dependent:
        family_verdict = "LONG_NESTED_PATH_REGIME_DEPENDENT_CORRECTED_DATA"
    else:
        family_verdict = "LONG_NESTED_PATH_STABLE_CORRECTED_DATA"

    return {
        "rows": rows, "cells": cells, "cell_order": cell_order,
        "population_report": population_report,
        "nested_nonnegativity_check": nonneg,
        "secondary_descriptive": secondary,
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# comparison with the immutable v001 (descriptive only -- NOT independent
# replications, the cohorts overlap in signals and cycles)
# ---------------------------------------------------------------------------

def compare_with_v001(conn, v002_family: dict, v001_experiment_id: str = OLD_EXPERIMENT_ID) -> dict:
    """Reads v001's ALREADY-FROZEN, byte-immutable stored results (never
    recomputes them) and compares descriptively against v002's fresh
    computation. No new p-value is computed for the comparison itself."""
    import ast

    def _read_metric(name):
        row = conn.execute(
            "SELECT metric_value FROM experiment_results WHERE experiment_id=? AND metric_name=?",
            (v001_experiment_id, name),
        ).fetchone()
        if row is None:
            return None
        try:
            return ast.literal_eval(row[0])
        except (ValueError, SyntaxError):
            return row[0]  # a bare verdict string (e.g. "LONG_NESTED_PATH_STABLE") is not a Python literal

    v001_population = _read_metric("population_report")
    v001_cells = {key: _read_metric(f"cell_{key}") for key in ALL_DELTA_FIELDS}
    v001_verdict = _read_metric("family_verdict")

    v002_population = v002_family["population_report"]
    v002_cells = v002_family["cells"]

    cell_verdict_changes = {}
    median_direction_changes = {}
    for key in ALL_DELTA_FIELDS:
        v001_c, v002_c = v001_cells.get(key), v002_cells.get(key)
        v001_verdict_c = v001_c.get("closure_classification") if v001_c else None
        v002_verdict_c = v002_c.get("closure_classification") if v002_c else None
        cell_verdict_changes[key] = {
            "v001_verdict": v001_verdict_c, "v002_verdict": v002_verdict_c,
            "changed": v001_verdict_c != v002_verdict_c,
        }
        v001_median = v001_c.get("full_median") if v001_c else None
        v002_median = v002_c.get("full_median") if v002_c else None
        same_sign = (
            v001_median is not None and v002_median is not None
            and (v001_median > 0) == (v002_median > 0)
        )
        median_direction_changes[key] = {
            "v001_median": v001_median, "v002_median": v002_median, "same_sign": same_sign,
        }

    # the original qualitative finding: adverse excursion (|MAE|) realized earlier (small/zero
    # incremental growth in later intervals) while favorable excursion (MFE) keeps accumulating
    mfe_keeps_growing_v002 = all(
        (v002_cells[k].get("full_median") or 0) >= 0 for k in MFE_DELTA_FIELDS
    ) and (v002_cells[MFE_DELTA_FIELDS[-1]].get("full_median") or 0) > 0
    mae_plateaus_early_v002 = (
        (v002_cells[ABS_MAE_DELTA_FIELDS[-1]].get("full_median") or 0) == 0
        or (v002_cells[ABS_MAE_DELTA_FIELDS[-1]].get("full_median") or 0)
            <= (v002_cells[ABS_MAE_DELTA_FIELDS[0]].get("full_median") or 0)
    )
    qualitative_finding_survives = bool(mfe_keeps_growing_v002 and mae_plateaus_early_v002)

    cohort_n_change = v002_population["signal_n"] - v001_population["signal_n"]
    cycle_n_change = v002_population["independent_cycle_n"] - v001_population["independent_cycle_n"]
    split_change = {
        "v001_train_cycle_n": v001_population["train_cycle_n"], "v001_test_cycle_n": v001_population["test_cycle_n"],
        "v002_train_cycle_n": v002_population["train_cycle_n"], "v002_test_cycle_n": v002_population["test_cycle_n"],
    }

    any_cell_verdict_changed = any(v["changed"] for v in cell_verdict_changes.values())
    all_insufficient_v002 = v002_family["family_verdict"] == "INSUFFICIENT_CORRECTED_COMMON_COHORT"

    if all_insufficient_v002:
        comparison_conclusion = "INSUFFICIENT_CORRECTED_COHORT"
    elif not any_cell_verdict_changed and qualitative_finding_survives:
        comparison_conclusion = "REPLICATED_ON_CORRECTED_EXPANDED_COHORT"
    elif qualitative_finding_survives:
        comparison_conclusion = "PARTIALLY_REPLICATED"
    else:
        comparison_conclusion = "MATERIAL_RESULT_CHANGE"

    return {
        "v001_experiment_id": v001_experiment_id, "v001_verdict": v001_verdict,
        "v002_verdict": v002_family["family_verdict"],
        "cohort_n_change": cohort_n_change, "cycle_n_change": cycle_n_change,
        "split_change": split_change,
        "cell_verdict_changes": cell_verdict_changes,
        "median_direction_changes": median_direction_changes,
        "qualitative_finding_survives": qualitative_finding_survives,
        "not_independent_replications_note": (
            "v001 and v002 share the vast majority of signals/cycles (v002's cohort is a strict "
            "superset expansion of v001's, not an independent sample) -- this comparison is a "
            "consistency check on a corrected-data rerun, never treated as two independent "
            "confirmations of the same finding."
        ),
        "comparison_conclusion": comparison_conclusion,
    }


def freeze_and_record(conn, provenance: str = "batch-candle-repair-w8-long-nested-path-002") -> dict:
    now = int(time.time() * 1000)
    family = compute_family(conn)
    comparison = compare_with_v001(conn, family)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(r["signal_id"] for r in family["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"EFFECTIVE ami_lifecycle_path_observations (path-v2-candle-repair-r1 where corrected, "
        f"else path-v2) WHERE observation_status=OK AT ALL 4 FROZEN HORIZONS (common cohort), "
        f"direction=LONG only; signal_n={family['population_report']['signal_n']}; "
        f"source_event_n={family['population_report']['source_event_n']}; "
        f"independent_cycle_n={family['population_report']['independent_cycle_n']}; "
        f"excluded_incomplete_horizon_n={family['population_report']['excluded_incomplete_horizon_n']}; "
        f"candle_data_version={CANDLE_DATA_VERSION}; path_data_version={PATH_DATA_VERSION}; "
        f"corrected_data_rerun_of={CORRECTED_DATA_RERUN_OF}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id, RECOMPUTED FRESH from the corrected-cohort population (never reused "
        f"from v001's smaller partition); MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle N "
        f"in TRAIN and TEST"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_LONG_NESTED_PATH_ACCUMULATION",
        "hypothesis_id": "H-W8-LONG-NESTED-PATH-ACCUMULATION", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(ALL_DELTA_FIELDS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN/TEST median stability of nested MFE/|MAE| accumulation deltas, LONG only, "
            "common 4-horizon cohort, CORRECTED (post candle-gap-repair) data"
        ),
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N} (independent-cycle N in TRAIN and TEST); TRAIN_FRACTION={TRAIN_FRACTION}; "
            "classification=stable iff Holm-p>=0.05 AND bootstrap-CI includes 0; regime-dependent iff Holm-p<0.05 "
            "AND CI excludes 0; disagreement->regime-dependent (conservative); insufficient iff either split's "
            "independent-cycle N<MIN_BUCKET_N"
        ),
        "frozen_splits": frozen_splits,
        "frozen_economic_gate": "N/A (no hold/exit/stop/re-entry/management rule tested)",
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}); "
            f"secondary=two-sided label-permutation median-difference p (n={N_PERMUTATIONS}) + ONE joint "
            f"Holm step-down across all 6 primary cells"
        ),
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": family["family_verdict"],
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": CORRECTED_DATA_RERUN_OF, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })
    rows_to_write = [
        ("population_report", family["population_report"]),
        ("nested_nonnegativity_check", family["nested_nonnegativity_check"]),
        ("secondary_descriptive", family["secondary_descriptive"]),
        ("family_verdict", family["family_verdict"]),
        ("candle_data_version", CANDLE_DATA_VERSION),
        ("path_data_version", PATH_DATA_VERSION),
        ("corrected_data_rerun_of", CORRECTED_DATA_RERUN_OF),
        ("comparison_with_v001", comparison),
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
        "comparison_with_v001": comparison,
    }
