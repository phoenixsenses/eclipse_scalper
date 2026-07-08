"""BATCH-P7B-1 (W8-LONG-TIMING-STRUCTURE-001): descriptive + chronologically-
stability-tested characterization of WHEN (not whether) MFE/MAE occur within
the fixed horizon window, for the canonical LONG population.

RESEARCH QUESTIONS (frozen, operator-specified):
  1. Does MFE tend to occur before MAE, or MAE before MFE?
     -> intrabar_order_status categorical breakdown (MFE_FIRST/MAE_FIRST/
        SAME_CANDLE_UNKNOWN) + the ZERO_AT_REFERENCE sub-case (both times
        are 0 -- price never left the reference point in either direction
        within the horizon) + timing_delta_ms descriptive quantiles.
  2. How long does it take to reach MFE and MAE?
     -> time_to_mfe_ms / time_to_mae_ms quantiles (q10-q90), and their
        horizon-relative fraction (time_to_*_fraction_of_horizon).
  3. Are these timing distributions chronologically stable?
     -> the PRIMARY INFERENTIAL FAMILY: cycle-grouped chronological 70/30
        TRAIN/TEST median-difference test (cluster bootstrap CI + label
        permutation p + ONE joint Holm correction), reusing the EXACT same
        machinery as W8-HOLD-BASELINE/W8-SHORT-EXPANDED-BASELINE.
  4. How does timing differ across the four frozen horizons?
     -> every statistic above is reported separately per horizon; no
        cross-horizon pooling anywhere.

NOT_A_MANAGEMENT_WAVE: NO_STOP_RULE / NO_EXIT_RULE / NO_REENTRY /
NO_HOLD_OPTIMIZATION -- this module answers exactly "when does the path reach
its favorable/adverse extremum", never "should a trade be stopped/exited/
re-entered/held differently". NO_ECONOMIC_OR_ALPHA_CLAIM: no verdict this
module can produce is a tradeability or profitability claim.

NO_SHORT_POOLING: population is direction='LONG' ONLY (frozen, matches
BATCH-W8-SHORT-EXPANDED-BASELINE's explicit finding that SHORT remains an
accumulation/new-event-family branch, not ready for combined analysis).
LONG's own population (142 independent cycles, comfortably >=20 cycles on
both sides of a 70/30 split in every horizon) is unaffected by that finding.

CYCLE-GROUPED SPLIT: reuses ami.research.w8_short_expanded_baseline's
generic (not SHORT-specific) split machinery VERBATIM --
compute_global_cycle_split()/split_rows_by_cycle_keys()/
assert_zero_cycle_straddling() operate on plain row dicts keyed by
independent_cycle_id/source_event_id/signal_birth_ts, with no SHORT-specific
logic baked in. Not reimplemented here.

PRIMARY INFERENTIAL FAMILY (frozen, 8 = 2 timing metrics x 4 horizons,
LONG only): time_to_mfe_ms, time_to_mae_ms. ONE joint Holm step-down
correction across all 8 cells (not per-horizon, not per-metric separately).
MIN_BUCKET_N=20 applies to independent-cycle N in EACH split (TRAIN/TEST),
per horizon -- since both timing metrics for a given horizon share the same
row population and the same global cycle partition, this is equivalently
"per split and horizon" (operator's exact phrasing) and "per split and cell"
(this module's literal cell-level check) at once.

SECONDARY DESCRIPTIVE REPORTING (no p-values, not part of the Holm family):
per-horizon MFE_FIRST/MAE_FIRST/SAME_CANDLE_UNKNOWN counts+rates, the
ZERO_AT_REFERENCE sub-case count+rate, time-to-MFE/MAE/fraction-of-horizon
quantiles (full population), timing_delta_ms quantiles, and a TRAIN-vs-TEST
rate comparison for the categorical breakdown (reported as two rate numbers
side by side, never as an inferential test -- no proportion test/p-value is
computed for the categorical breakdown, only for the two continuous timing
metrics in the primary family).
"""
from __future__ import annotations
import hashlib
import time

from ami.research.feature_gateway import fetch_lifecycle_signals, fetch_path_observations
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, PATH_HORIZONS_MS, TRAIN_FRACTION
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

EXPERIMENT_ID = "E-W8-LONG-TIMING-STRUCTURE-001"
RESEARCH_CONTEXT_ID = "w8-long-timing-structure"

DIRECTION = "LONG"
HORIZONS = ("scalp_30m", "scalp_1h", "swing_4h", "swing_24h")
TIMING_METRICS = ("time_to_mfe_ms", "time_to_mae_ms")  # primary inferential family metrics


def fetch_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    """feature_gateway ONLY (fetch_lifecycle_signals + fetch_path_observations),
    observation_status='OK', direction='LONG'. Adds the 3 frozen derived
    descriptive fields (fraction-of-horizon x2, timing_delta_ms) -- never
    interpreted as entry/exit/management rules, purely descriptive.

    [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL C] Pins path_definition_version="path-v2" -- E-W8-LONG-TIMING-
    STRUCTURE-001's frozen, pre-candle-repair historical population. An
    unpinned read now fails closed (feature_gateway.AmbiguousPathVersionError)."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_path_observations(
        conn, RESEARCH_CONTEXT_ID,
        equals={"observation_status": "OK", "path_definition_version": "path-v2"},
    )
    rows = []
    for o in obs_rows:
        s = signals.get(o["signal_id"])
        if s is None or s["direction"] != DIRECTION:
            continue
        horizon_ms = PATH_HORIZONS_MS[o["horizon_name"]]
        time_to_mfe_ms = o["time_to_mfe_ms"]
        time_to_mae_ms = o["time_to_mae_ms"]
        row = {
            **o, "direction": s["direction"], "signal_birth_ts": s["signal_birth_ts"],
            "source_event_id": s["source_event_id"], "independent_cycle_id": s["independent_cycle_id"],
            "setup_id": s["setup_id"],
        }
        row["time_to_mfe_fraction_of_horizon"] = (
            round(time_to_mfe_ms / horizon_ms, 6) if time_to_mfe_ms is not None else None
        )
        row["time_to_mae_fraction_of_horizon"] = (
            round(time_to_mae_ms / horizon_ms, 6) if time_to_mae_ms is not None else None
        )
        row["timing_delta_ms"] = (
            time_to_mfe_ms - time_to_mae_ms if time_to_mfe_ms is not None and time_to_mae_ms is not None else None
        )
        rows.append(row)
    return rows


def _cell_rows(rows: list[dict], horizon: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == DIRECTION]


# ---------------------------------------------------------------------------
# secondary, descriptive-only reporting (no p-values)
# ---------------------------------------------------------------------------

def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total > 0 else None


def compute_horizon_descriptive(cell_rows: list[dict], train_rows: list[dict], test_rows: list[dict]) -> dict:
    total_n = len(cell_rows)
    status_counts = {"MFE_FIRST": 0, "MAE_FIRST": 0, "SAME_CANDLE_UNKNOWN": 0}
    zero_at_reference_n = 0
    for r in cell_rows:
        status = r.get("intrabar_order_status")
        if status in status_counts:
            status_counts[status] += 1
        if r.get("time_to_mfe_ms") == 0 and r.get("time_to_mae_ms") == 0:
            zero_at_reference_n += 1

    def _train_test_rate(status: str) -> dict:
        train_total = len(train_rows)
        test_total = len(test_rows)
        train_count = sum(1 for r in train_rows if r.get("intrabar_order_status") == status)
        test_count = sum(1 for r in test_rows if r.get("intrabar_order_status") == status)
        return {"train_rate": _rate(train_count, train_total), "test_rate": _rate(test_count, test_total)}

    time_to_mfe_vals = [r["time_to_mfe_ms"] for r in cell_rows if r.get("time_to_mfe_ms") is not None]
    time_to_mae_vals = [r["time_to_mae_ms"] for r in cell_rows if r.get("time_to_mae_ms") is not None]
    frac_mfe_vals = [r["time_to_mfe_fraction_of_horizon"] for r in cell_rows
                      if r.get("time_to_mfe_fraction_of_horizon") is not None]
    frac_mae_vals = [r["time_to_mae_fraction_of_horizon"] for r in cell_rows
                      if r.get("time_to_mae_fraction_of_horizon") is not None]
    delta_vals = [r["timing_delta_ms"] for r in cell_rows if r.get("timing_delta_ms") is not None]

    def _quantile_set(vals: list[float]) -> dict:
        return {
            f"q{int(q*100)}": (round(_quantile(vals, q), 4) if vals and _quantile(vals, q) is not None else None)
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        }

    return {
        "raw_signal_n": total_n,
        "intrabar_order_status_counts": status_counts,
        "intrabar_order_status_rates": {k: _rate(v, total_n) for k, v in status_counts.items()},
        "zero_at_reference_n": zero_at_reference_n,
        "zero_at_reference_rate": _rate(zero_at_reference_n, total_n),
        "train_vs_test_rate_by_status": {
            status: _train_test_rate(status) for status in ("MFE_FIRST", "MAE_FIRST", "SAME_CANDLE_UNKNOWN")
        },
        "time_to_mfe_ms_quantiles": _quantile_set(time_to_mfe_vals),
        "time_to_mae_ms_quantiles": _quantile_set(time_to_mae_vals),
        "time_to_mfe_fraction_of_horizon_quantiles": _quantile_set(frac_mfe_vals),
        "time_to_mae_fraction_of_horizon_quantiles": _quantile_set(frac_mae_vals),
        "timing_delta_ms_quantiles": _quantile_set(delta_vals),
        "timing_delta_ms_median": round(_median(delta_vals), 4) if delta_vals else None,
    }


# ---------------------------------------------------------------------------
# primary inferential family: 8 cells (2 timing metrics x 4 horizons)
# ---------------------------------------------------------------------------

def compute_cell(cell_rows: list[dict], metric: str, train_keys: set, test_keys: set) -> dict:
    vals = [r[metric] for r in cell_rows if r.get(metric) is not None]
    train_rows, test_rows = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
    straddling_n = assert_zero_cycle_straddling(train_rows, test_rows)
    train_vals = [r[metric] for r in train_rows if r.get(metric) is not None]
    test_vals = [r[metric] for r in test_rows if r.get(metric) is not None]

    raw_signal_n = len(cell_rows)
    distinct_source_event_n = len({r["source_event_id"] for r in cell_rows if r["source_event_id"] is not None})
    distinct_cycle_n = len({_cycle_key(r) for r in cell_rows})
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

    # MANDATORY: sufficiency is CYCLE-count based (per split, per horizon), never signal-count based
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


def compute_family(conn, symbol: str = "ETHUSDT") -> dict:
    rows = fetch_population(conn, symbol)
    split = compute_global_cycle_split(rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    per_horizon_split_report: dict[str, dict] = {}
    descriptive_by_horizon: dict[str, dict] = {}

    for horizon in HORIZONS:
        cell_rows = _cell_rows(rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
        per_horizon_split_report[horizon] = {
            "raw_signal_n": len(cell_rows),
            "source_event_n": len({r["source_event_id"] for r in cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in cell_rows}),
            "train_cycle_n": len({_cycle_key(r) for r in train_r}),
            "test_cycle_n": len({_cycle_key(r) for r in test_r}),
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
            "sufficiency_verdict": (
                "OK" if (len({_cycle_key(r) for r in train_r}) >= MIN_BUCKET_N
                         and len({_cycle_key(r) for r in test_r}) >= MIN_BUCKET_N)
                else "INSUFFICIENT_SAMPLE"
            ),
        }
        descriptive_by_horizon[horizon] = compute_horizon_descriptive(cell_rows, train_r, test_r)
        for metric in TIMING_METRICS:
            key = f"{metric}|{horizon}"
            cell_order.append(key)
            cells[key] = compute_cell(cell_rows, metric, train_keys, test_keys)

    assert len(cell_order) == 8, f"primary family must be exactly 8 cells, got {len(cell_order)}"

    # ONE joint Holm correction across all 8 p-values together
    p_values = [cells[k]["permutation_p_value"] for k in cell_order]
    holm = holm_adjust(p_values)
    for key, adj in zip(cell_order, holm):
        cells[key]["permutation_p_value_holm_adjusted"] = adj
        cells[key]["closure_classification"] = classify_cell_verdict(
            cells[key]["sample_sufficiency"], adj, cells[key]["bootstrap_ci95"],
        )

    any_insufficient = any(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    all_insufficient = all(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    any_regime_dependent = any(
        cells[k]["closure_classification"] == "ANSWERED_REGIME_DEPENDENT_BASELINE" for k in cell_order
    )
    any_stable = any(cells[k]["closure_classification"] == "ANSWERED_SUPPORTED_STABLE_BASELINE" for k in cell_order)

    # per-horizon verdict consistency check -- if different horizons land on different
    # verdicts, the family is MIXED_BY_HORIZON, never silently averaged/pooled
    horizon_verdicts: dict[str, set] = {}
    for key in cell_order:
        metric, horizon = key.split("|")
        horizon_verdicts.setdefault(horizon, set()).add(cells[key]["closure_classification"])
    distinct_horizon_level_verdicts = {next(iter(v)) if len(v) == 1 else "MIXED_WITHIN_HORIZON"
                                       for v in horizon_verdicts.values()}

    if all_insufficient:
        family_verdict = "INSUFFICIENT_SAMPLE"
    elif any_insufficient:
        family_verdict = "MIXED_BY_HORIZON"
    elif len(distinct_horizon_level_verdicts) > 1 or "MIXED_WITHIN_HORIZON" in distinct_horizon_level_verdicts:
        family_verdict = "MIXED_BY_HORIZON"
    elif any_regime_dependent:
        family_verdict = "LONG_TIMING_STRUCTURE_REGIME_DEPENDENT"
    else:
        family_verdict = "LONG_TIMING_STRUCTURE_STABLE"

    return {
        "rows": rows, "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "per_horizon_split_report": per_horizon_split_report,
        "descriptive_by_horizon": descriptive_by_horizon,
        "raw_signal_n_population": len({r["signal_id"] for r in rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in rows}),
        "family_verdict": family_verdict,
    }


def freeze_and_record(conn, provenance: str = "batch-p7b1-w8-long-timing-structure") -> dict:
    now = int(time.time() * 1000)
    family = compute_family(conn)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(f"{r['signal_id']}|{r['horizon_name']}" for r in family["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"ami_lifecycle_path_observations WHERE observation_status=OK, direction=LONG only; "
        f"raw_signal_n={family['raw_signal_n_population']}; "
        f"distinct_source_event_n={family['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={family['distinct_independent_cycle_n_population']}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id (never signal-level): cycle order key=earliest signal_birth_ts among all "
        f"of that cycle's rows; MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle N per split/horizon"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_LONG_TIMING_STRUCTURE",
        "hypothesis_id": "H-W8-LONG-TIMING-STRUCTURE", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": ",".join(TIMING_METRICS),
        "frozen_target": (
            "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30%) "
            "median stability of time_to_mfe_ms/time_to_mae_ms, LONG only, by horizon"
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
        "supersedes_experiment_id": None, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })
    rows_to_write = [
        ("raw_signal_n_population", family["raw_signal_n_population"]),
        ("distinct_source_event_n_population", family["distinct_source_event_n_population"]),
        ("distinct_independent_cycle_n_population", family["distinct_independent_cycle_n_population"]),
        ("global_cycle_split", family["global_split"]),
        ("per_horizon_split_report", family["per_horizon_split_report"]),
        ("descriptive_by_horizon", family["descriptive_by_horizon"]),
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
        "raw_signal_n_population": family["raw_signal_n_population"],
        "distinct_source_event_n_population": family["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": family["distinct_independent_cycle_n_population"],
        "global_split": family["global_split"],
        "per_horizon_split_report": family["per_horizon_split_report"],
        "descriptive_by_horizon": family["descriptive_by_horizon"],
    }
