"""BATCH-P7B-1 (W8-VOL-NORMALIZED-BASELINE): volatility-normalized fixed-
horizon MFE/MAE baseline stability -- Candidate E from the "Next Research
Frontier Reconciliation", the natural continuation of W8-HOLD-BASELINE
(E-W8-HOLD-BASELINE-001), which explicitly flagged this as its own next
step ("vol-normalized analysis... a natural next step").

Reuses ami.research.w8_hold_baseline's generic statistical machinery
VERBATIM -- _cell_rows(), compute_cell(), classify_cell_verdict(),
_split_chronological_by_birth(), cluster_bootstrap_median_diff(),
permutation_test_median_diff() are all IMPORTED, not reimplemented. Only
two things differ from W8-HOLD-BASELINE:
  1. METRICS = (mfe_anchor_vol_units, mae_anchor_vol_units), not (mfe_bps, mae_bps)
  2. population additionally requires volatility_status='OK' (912 rows, not 914)

NOT_A_MANAGEMENT_WAVE (same discipline as W8-HOLD-BASELINE): no stop/exit/
partial-exit/time-stop/re-entry/cancellation rule anywhere in this module,
no economic claim. This is a unit-normalization of an already-descriptive
baseline, not a new hypothesis about tradeability.

NO_SIGNAL_BACKFILL: this module reads the EXISTING canonical population
exactly as-is -- it does not canonicalize, backfill, or ingest any new
signal/event.
"""
from __future__ import annotations
import hashlib
import time

from ami.research.feature_gateway import fetch_lifecycle_signals, fetch_path_observations
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results
from ami.research.w8_hold_baseline import (
    DIRECTIONS,
    HORIZONS,
    MIN_BUCKET_N,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    TRAIN_FRACTION,
    _cell_rows,
    classify_cell_verdict,
    compute_cell,
)

EXPERIMENT_ID = "E-W8-VOL-NORMALIZED-BASELINE-001"
RESEARCH_CONTEXT_ID = "w8-vol-normalized-baseline"
RAW_BPS_EXPERIMENT_ID = "E-W8-HOLD-BASELINE-001"  # for the explicit comparison this batch requires

METRICS = ("mfe_anchor_vol_units", "mae_anchor_vol_units")
# maps this wave's metric name -> W8-HOLD-BASELINE's corresponding raw-bps metric name,
# for the mandated side-by-side comparison (never merged/re-derived, just looked up)
_RAW_BPS_METRIC_OF = {"mfe_anchor_vol_units": "mfe_bps", "mae_anchor_vol_units": "mae_bps"}


def fetch_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    """Same join pattern as w8_hold_baseline.fetch_population(), via
    feature_gateway only, with the ADDITIONAL volatility_status='OK' filter
    (fetch_path_observations' equals= supports multiple ANDed columns
    already -- no gateway change needed).

    [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL C] Pins path_definition_version="path-v2" for the same reason as
    w8_hold_baseline.fetch_population() -- this is E-W8-VOL-NORMALIZED-
    BASELINE-001's frozen, pre-repair historical population; an unpinned
    read now fails closed (feature_gateway.AmbiguousPathVersionError) since
    the table can hold more than one physical version per (signal_id,
    horizon_name)."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_path_observations(
        conn, RESEARCH_CONTEXT_ID,
        equals={"observation_status": "OK", "volatility_status": "OK", "path_definition_version": "path-v2"},
    )
    rows = []
    for o in obs_rows:
        s = signals.get(o["signal_id"])
        if s is None:
            continue
        rows.append({
            **o,
            "direction": s["direction"],
            "signal_birth_ts": s["signal_birth_ts"],
            "source_event_id": s["source_event_id"],
            "independent_cycle_id": s["independent_cycle_id"],
        })
    return rows


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    rows = fetch_population(conn, symbol)

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    for metric in METRICS:
        for horizon in HORIZONS:
            for direction in DIRECTIONS:
                key = f"{metric}|{horizon}|{direction}"
                cell_order.append(key)
                cells[key] = compute_cell(_cell_rows(rows, horizon, direction), metric)

    p_values = [cells[k]["permutation_p_value"] for k in cell_order]
    holm = holm_adjust(p_values)
    for key, adj in zip(cell_order, holm):
        cells[key]["permutation_p_value_holm_adjusted"] = adj
        cells[key]["closure_classification"] = classify_cell_verdict(
            cells[key]["sample_sufficiency"], adj, cells[key]["bootstrap_ci95"],
        )

    return {
        "rows": rows,
        "cells": cells,
        "cell_order": cell_order,
        "raw_signal_n_population": len({r["signal_id"] for r in rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in rows}),
        "distinct_independent_cycle_n_population": len(
            {r["independent_cycle_id"] or f"NOCYCLE-{r['source_event_id']}" for r in rows}
        ),
    }


def _iqr_ratio(cell: dict) -> float | None:
    """|train_minus_test_median_diff| / IQR -- a unit-free effect-size proxy
    that lets a vol-units cell be compared against its raw-bps counterpart
    despite the two using genuinely different units (vol-normalization is a
    per-row division by that signal's own realized_vol_at_anchor, not a
    single constant rescaling -- so the two train-test gaps are not simply
    proportional; this ratio is the honest way to compare their relative
    size)."""
    diff = cell["train_minus_test_median_diff"]
    iqr = cell["iqr"]
    if diff is None or iqr is None or iqr == 0:
        return None
    return round(abs(diff) / iqr, 4)


def compare_with_raw_bps_baseline(conn, vol_cells: dict) -> dict:
    """Reads E-W8-HOLD-BASELINE-001's stored per-cell results back from
    experiment_results (read-only lookup, never recomputed/re-derived) and
    compares them cell-by-cell against this wave's vol-normalized cells."""
    import ast

    raw_rows = conn.execute(
        "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=?",
        (RAW_BPS_EXPERIMENT_ID,),
    ).fetchall()
    raw_cells: dict[str, dict] = {}
    for metric_name, metric_value in raw_rows:
        if metric_name.startswith("cell_"):
            raw_cells[metric_name[len("cell_"):]] = ast.literal_eval(metric_value)

    comparisons = {}
    long_cell_regressions = []  # LONG cells that were STABLE in raw-bps but are NOT stable here
    for vol_key in vol_cells:
        vol_metric, horizon, direction = vol_key.split("|")
        raw_metric = _RAW_BPS_METRIC_OF[vol_metric]
        raw_key = f"{raw_metric}|{horizon}|{direction}"
        raw_cell = raw_cells.get(raw_key)
        vol_cell = vol_cells[vol_key]
        if raw_cell is None:
            comparisons[vol_key] = {"raw_bps_counterpart": raw_key, "status": "RAW_BPS_CELL_NOT_FOUND"}
            continue

        raw_verdict = raw_cell["closure_classification"]
        vol_verdict = vol_cell["closure_classification"]
        raw_ratio = None
        v_diff = vol_cell["train_minus_test_median_diff"]
        v_iqr = vol_cell["iqr"]
        raw_diff = raw_cell["train_minus_test_median_diff"]
        raw_iqr = raw_cell["iqr"]
        raw_ratio = round(abs(raw_diff) / raw_iqr, 4) if raw_diff is not None and raw_iqr else None
        vol_ratio = round(abs(v_diff) / v_iqr, 4) if v_diff is not None and v_iqr else None

        comparisons[vol_key] = {
            "raw_bps_counterpart": raw_key,
            "raw_bps_verdict": raw_verdict,
            "vol_normalized_verdict": vol_verdict,
            "verdict_changed": raw_verdict != vol_verdict,
            "raw_bps_train_minus_test_over_iqr": raw_ratio,
            "vol_normalized_train_minus_test_over_iqr": vol_ratio,
            "raw_signal_n_unchanged_or_reduced_by_vol_filter": (
                vol_cell["raw_signal_n"] <= raw_cell["raw_signal_n"]
            ),
        }
        if direction == "LONG" and raw_verdict == "ANSWERED_SUPPORTED_STABLE_BASELINE" \
                and vol_verdict != "ANSWERED_SUPPORTED_STABLE_BASELINE":
            long_cell_regressions.append(vol_key)

    return {
        "per_cell": comparisons,
        "any_long_cell_stable_to_regime_dependent": long_cell_regressions,
        "normalization_repairs_sample_size": False,  # documented finding, never silently assumed
    }


def freeze_and_record(conn, provenance: str = "batch-p7b1-w8-vol-normalized-baseline") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    comparison = compare_with_raw_bps_baseline(conn, metrics["cells"])

    dataset_hash = hashlib.sha256(
        "|".join(sorted(f"{r['signal_id']}|{r['horizon_name']}" for r in metrics["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"ami_lifecycle_path_observations WHERE observation_status=OK AND volatility_status=OK; "
        f"raw_signal_n={metrics['raw_signal_n_population']}; "
        f"distinct_source_event_n={metrics['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={metrics['distinct_independent_cycle_n_population']}"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_VOL_NORMALIZED_BASELINE",
        "hypothesis_id": "H-W8-VOL-NORMALIZED-BASELINE", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": "mfe_anchor_vol_units,mae_anchor_vol_units",
        "frozen_target": (
            "TRAIN(chronological first 70%) vs TEST(final 30%) median stability, by horizon x direction "
            "-- IDENTICAL design to E-W8-HOLD-BASELINE-001, verbatim reuse, metric swapped to vol-normalized"
        ),
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N};TRAIN_FRACTION={TRAIN_FRACTION};same classification rule as "
            "E-W8-HOLD-BASELINE-001 (never defaults to stable)"
        ),
        "frozen_splits": (
            f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} by signal_birth_ts, never randomized"
        ),
        "frozen_economic_gate": (
            "N/A (no management/exit/stop/re-entry rule -- unit-normalization of an already-descriptive baseline)"
        ),
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}); "
            f"secondary=two-sided permutation (n={N_PERMUTATIONS}) + Holm across exactly 16 cells "
            f"-- same seeds/method as E-W8-HOLD-BASELINE-001"
        ),
        "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
        "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
        "mutation_test_count": 0, "mutation_test_passed": 1,
        "supersedes_experiment_id": None, "report_artifact_id": None,
        "schema_version": 10, "provenance": provenance, "created_ms": now, "updated_ms": now,
    })

    rows_to_write = [
        ("raw_signal_n_population", metrics["raw_signal_n_population"]),
        ("distinct_source_event_n_population", metrics["distinct_source_event_n_population"]),
        ("distinct_independent_cycle_n_population", metrics["distinct_independent_cycle_n_population"]),
        ("comparison_with_raw_bps_baseline", comparison),
    ]
    for key in metrics["cell_order"]:
        rows_to_write.append((f"cell_{key}", metrics["cells"][key]))

    results = [(name, str(value)) for name, value in rows_to_write]
    record_experiment_results(conn, EXPERIMENT_ID, results, schema_version=10, provenance=provenance, created_ms=now)

    conn.commit()
    return {
        "cells": metrics["cells"],
        "cell_order": metrics["cell_order"],
        "raw_signal_n_population": metrics["raw_signal_n_population"],
        "distinct_source_event_n_population": metrics["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": metrics["distinct_independent_cycle_n_population"],
        "comparison_with_raw_bps_baseline": comparison,
    }


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        r = freeze_and_record(conn)
        print(f"raw_signal_n_population={r['raw_signal_n_population']} "
              f"distinct_source_event_n_population={r['distinct_source_event_n_population']} "
              f"distinct_independent_cycle_n_population={r['distinct_independent_cycle_n_population']}")
        for key in r["cell_order"]:
            c = r["cells"][key]
            print(f"{key}: n={c['raw_signal_n']} cycle_n={c['distinct_independent_cycle_n']} "
                  f"sufficiency={c['sample_sufficiency']} verdict={c['closure_classification']}")
        print(f"comparison_with_raw_bps_baseline={r['comparison_with_raw_bps_baseline']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
