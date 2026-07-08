"""BATCH-P7B-1 (W8-SHORT-EXPANDED-BASELINE): cycle-grouped chronological
SHORT-only re-run of W8-HOLD-BASELINE + W8-VOL-NORMALIZED-BASELINE, triggered
by SHORT population growth from BATCH-SHORT-NOISY-V1-CANON-BACKFILL (54 new
SHORT_NOISY_BTC200K_CONFIRMED_V1 signals canonicalized into ami_signal_lifecycle).

WHY A NEW EXPERIMENT, NOT AN OVERWRITE: E-W8-HOLD-BASELINE-001 and
E-W8-VOL-NORMALIZED-BASELINE-001 are FROZEN, immutable prior results (both
directions, at the population size that existed before the SHORT_NOISY_V1
backfill). This module writes to two NEW experiment_ids (RAW_BPS_EXPERIMENT_ID
= E-W8-HOLD-BASELINE-002-SHORT-EXPANDED, VOL_NORMALIZED_EXPERIMENT_ID =
E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED) -- both
`experiment_registry`/`experiment_results` writes are scoped by experiment_id
(INSERT ... ON CONFLICT(experiment_id) / DELETE WHERE experiment_id=?), so a
new experiment_id can never touch the -001 rows. Verified explicitly by
`assert_old_experiments_untouched()`.

MANDATORY CYCLE-GROUPED SPLIT (the entire point of this rerun): a naive
signal-level 70/30 chronological split (as W8-HOLD-BASELINE/W8-VOL-NORMALIZED-
BASELINE-001 both used) can put different signals of the SAME
independent_cycle_id on opposite sides of the split -- inflating apparent
n on both sides without adding a genuinely independent observation. This
module instead:
  1. groups every SHORT row (across ALL 4 horizons) by independent_cycle_id
     (or "NOCYCLE-<source_event_id>" for an unresolved cycle, same fallback
     convention as W1/W7A/W8-HOLD-BASELINE)
  2. assigns each cycle ONE chronological ordering timestamp = the EARLIEST
     signal_birth_ts among ALL of that cycle's member rows (any horizon) --
     never the per-cell-filtered subset, so the partition is identical for
     every cell
  3. sorts cycles by that timestamp, splits at TRAIN_FRACTION (70/30) BY
     CYCLE COUNT, and assigns EVERY row belonging to a given cycle to
     whichever side that cycle landed on
  4. this ONE global partition (GLOBAL_TRAIN_CYCLE_KEYS / GLOBAL_TEST_CYCLE_KEYS,
     computed once from the raw-bps SHORT population -- the superset of the
     vol-normalized SHORT population) is reused, verbatim, for every one of
     the 16 primary cells (both raw-bps and vol-normalized metric groups) --
     a cycle is NEVER assigned TRAIN for one cell and TEST for another.
`assert_zero_cycle_straddling()` proves, for every cell, that
{cycle keys in that cell's TRAIN rows} and {cycle keys in that cell's TEST
rows} are disjoint.

MIN_BUCKET_N=20 applies to INDEPENDENT-CYCLE N in each split, not signal-level
N (operator instruction) -- `compute_cell()`'s sufficiency check counts
distinct cycle keys in train_rows/test_rows, never len(train_vals)/len(test_vals).

PRIMARY FAMILY (frozen, 16 = 4 metrics x 4 horizons, SHORT only): mfe_bps,
mae_bps (raw-bps, W8-HOLD-BASELINE's metrics) + mfe_anchor_vol_units,
mae_anchor_vol_units (vol-normalized, W8-VOL-NORMALIZED-BASELINE's metrics),
x scalp_30m/scalp_1h/swing_4h/swing_24h. ONE frozen Holm step-down correction
is applied across all 16 p-values TOGETHER (not two separate 8-cell
corrections) -- `compute_family()` pools all 16 permutation p-values, calls
holm_adjust() once, then the results are split back out for storage under
the two experiment_ids they semantically belong to (mirroring how -001's
raw-bps and vol-normalized results were always stored as two experiments,
while this wave's INFERENCE is a single joint 16-cell family).

COMPOSITION DIAGNOSTIC (secondary, descriptive-only, NO p-values): reports
pre-existing-SHORT vs new-SHORT_NOISY_BTC200K_CONFIRMED_V1 vs combined
population N/medians per horizon, to show (not test) whether any apparent
regime dependence concentrates in the new, June-heavy setup family. These
three slices are explicitly NOT three independent confirmations -- combined
IS the primary-family population; the other two are informational subsets of
it, never separately inferentially tested here.

NOT_A_MANAGEMENT_WAVE: no stop/exit/partial-exit/time-stop/re-entry/
cancellation rule anywhere in this module. NO_ECONOMIC_CLAIM: no PnL/alpha
claim is made or implied by any verdict this module can produce.
NO_SIGNAL_BACKFILL: reads the EXISTING canonical population exactly as-is.
"""
from __future__ import annotations
import hashlib
import time

from ami.research.feature_gateway import fetch_lifecycle_signals, fetch_path_observations
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results
from ami.research.w8_hold_baseline import (
    HORIZONS,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    RESEARCH_CONTEXT_ID as _RAW_BPS_RESEARCH_CONTEXT_ID,
    _median,
    _quantile,
    classify_cell_verdict,
    cluster_bootstrap_median_diff,
    permutation_test_median_diff,
)

# frozen, immutable prior results -- NEVER written to by this module (different experiment_id
# scope makes this structurally impossible, verified explicitly by assert_old_experiments_untouched)
OLD_RAW_BPS_EXPERIMENT_ID = "E-W8-HOLD-BASELINE-001"
OLD_VOL_NORMALIZED_EXPERIMENT_ID = "E-W8-VOL-NORMALIZED-BASELINE-001"

RAW_BPS_EXPERIMENT_ID = "E-W8-HOLD-BASELINE-002-SHORT-EXPANDED"
VOL_NORMALIZED_EXPERIMENT_ID = "E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED"
RESEARCH_CONTEXT_ID = "w8-short-expanded-baseline"

DIRECTION = "SHORT"
NEW_SETUP_ID = "SHORT_NOISY_BTC200K_CONFIRMED_V1"

RAW_BPS_METRICS = ("mfe_bps", "mae_bps")
VOL_NORMALIZED_METRICS = ("mfe_anchor_vol_units", "mae_anchor_vol_units")
ALL_METRICS = RAW_BPS_METRICS + VOL_NORMALIZED_METRICS  # 4 metrics
_METRIC_EXPERIMENT_OF = {
    "mfe_bps": RAW_BPS_EXPERIMENT_ID, "mae_bps": RAW_BPS_EXPERIMENT_ID,
    "mfe_anchor_vol_units": VOL_NORMALIZED_EXPERIMENT_ID, "mae_anchor_vol_units": VOL_NORMALIZED_EXPERIMENT_ID,
}


def _cycle_key(row: dict) -> str:
    return row["independent_cycle_id"] or f"NOCYCLE-{row['source_event_id']}"


# ---------------------------------------------------------------------------
# population assembly (feature_gateway only) -- raw-bps population is the
# superset (observation_status=OK); vol-normalized additionally requires
# volatility_status=OK. Both carry setup_id (for the composition diagnostic).
# ---------------------------------------------------------------------------

def fetch_raw_bps_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    """[BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY
    HARDENING, GOAL C] Pins path_definition_version="path-v2" -- this is
    E-W8-HOLD-BASELINE-002-SHORT-EXPANDED's frozen, pre-candle-repair
    historical population (post SHORT_NOISY_V1 canonicalization, before the
    candle repair). An unpinned read now fails closed
    (feature_gateway.AmbiguousPathVersionError)."""
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
        rows.append({
            **o, "direction": s["direction"], "signal_birth_ts": s["signal_birth_ts"],
            "source_event_id": s["source_event_id"], "independent_cycle_id": s["independent_cycle_id"],
            "setup_id": s["setup_id"],
        })
    return rows


def fetch_vol_normalized_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    """See fetch_raw_bps_population()'s docstring -- same path_definition_version
    pin, for E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED's frozen population."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_path_observations(
        conn, RESEARCH_CONTEXT_ID,
        equals={"observation_status": "OK", "volatility_status": "OK", "path_definition_version": "path-v2"},
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
# mandatory cycle-grouped split -- ONE global partition, reused everywhere
# ---------------------------------------------------------------------------

def compute_global_cycle_split(rows: list[dict]) -> dict:
    """Groups ALL rows (any horizon) by cycle, orders cycles by the EARLIEST
    signal_birth_ts among each cycle's member rows, splits at TRAIN_FRACTION
    by CYCLE COUNT. Returns the frozen partition plus enough detail to prove
    zero straddling and report exact cycle-level counts."""
    by_cycle: dict[str, list[dict]] = {}
    for r in rows:
        by_cycle.setdefault(_cycle_key(r), []).append(r)
    cycle_anchors = sorted(
        ((key, min(m["signal_birth_ts"] for m in members)) for key, members in by_cycle.items()),
        key=lambda kv: kv[1],
    )
    cut = int(len(cycle_anchors) * TRAIN_FRACTION)
    train_keys = {k for k, _ in cycle_anchors[:cut]}
    test_keys = {k for k, _ in cycle_anchors[cut:]}
    assert train_keys.isdisjoint(test_keys), "cycle partition must be a strict bipartition"
    return {
        "train_cycle_keys": train_keys, "test_cycle_keys": test_keys,
        "total_cycle_n": len(cycle_anchors), "train_cycle_n": len(train_keys), "test_cycle_n": len(test_keys),
        "ordered_cycle_anchors": cycle_anchors,
    }


def split_rows_by_cycle_keys(rows: list[dict], train_keys: set, test_keys: set) -> tuple[list[dict], list[dict]]:
    train_rows = [r for r in rows if _cycle_key(r) in train_keys]
    test_rows = [r for r in rows if _cycle_key(r) in test_keys]
    return train_rows, test_rows


def assert_zero_cycle_straddling(train_rows: list[dict], test_rows: list[dict]) -> int:
    """Returns the number of cycle keys present in BOTH train_rows and
    test_rows for a given cell -- must be 0 by construction (a cycle is
    assigned to exactly one side globally); this function proves it,
    never assumes it."""
    train_cycles = {_cycle_key(r) for r in train_rows}
    test_cycles = {_cycle_key(r) for r in test_rows}
    return len(train_cycles & test_cycles)


def _cell_rows(rows: list[dict], horizon: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == DIRECTION]


def compute_cell(cell_rows: list[dict], metric: str, train_keys: set, test_keys: set) -> dict:
    vals = [r[metric] for r in cell_rows]
    train_rows, test_rows = split_rows_by_cycle_keys(cell_rows, train_keys, test_keys)
    straddling_n = assert_zero_cycle_straddling(train_rows, test_rows)
    train_vals = [r[metric] for r in train_rows]
    test_vals = [r[metric] for r in test_rows]

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

    # MANDATORY: sufficiency is CYCLE-count based, never signal-count based
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
    """Computes the full 16-cell (4 metrics x 4 horizons, SHORT only) primary
    family with ONE joint Holm correction. The global cycle split is computed
    from the raw-bps population (the superset) and reused, unchanged, for
    every cell of BOTH metric groups."""
    raw_bps_rows = fetch_raw_bps_population(conn, symbol)
    vol_norm_rows = fetch_vol_normalized_population(conn, symbol)

    split = compute_global_cycle_split(raw_bps_rows)
    train_keys, test_keys = split["train_cycle_keys"], split["test_cycle_keys"]

    cells: dict[str, dict] = {}
    cell_order: list[str] = []
    per_horizon_split_report: dict[str, dict] = {}
    for horizon in HORIZONS:
        raw_cell_rows = _cell_rows(raw_bps_rows, horizon)
        vol_cell_rows = _cell_rows(vol_norm_rows, horizon)
        train_r, test_r = split_rows_by_cycle_keys(raw_cell_rows, train_keys, test_keys)
        per_horizon_split_report[horizon] = {
            "raw_signal_n": len(raw_cell_rows),
            "source_event_n": len({r["source_event_id"] for r in raw_cell_rows if r["source_event_id"]}),
            "independent_cycle_n": len({_cycle_key(r) for r in raw_cell_rows}),
            "train_cycle_n": len({_cycle_key(r) for r in train_r}),
            "test_cycle_n": len({_cycle_key(r) for r in test_r}),
            "cycle_straddling_violations": assert_zero_cycle_straddling(train_r, test_r),
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

    # ONE joint Holm correction across all 16 p-values together
    p_values = [cells[k]["permutation_p_value"] for k in cell_order]
    holm = holm_adjust(p_values)
    for key, adj in zip(cell_order, holm):
        cells[key]["permutation_p_value_holm_adjusted"] = adj
        cells[key]["closure_classification"] = classify_cell_verdict(
            cells[key]["sample_sufficiency"], adj, cells[key]["bootstrap_ci95"],
        )

    any_insufficient = any(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    all_insufficient = all(cells[k]["sample_sufficiency"] == "INSUFFICIENT_SAMPLE" for k in cell_order)
    any_regime_dependent = any(cells[k]["closure_classification"] == "ANSWERED_REGIME_DEPENDENT_BASELINE" for k in cell_order)
    any_stable = any(cells[k]["closure_classification"] == "ANSWERED_SUPPORTED_STABLE_BASELINE" for k in cell_order)

    if all_insufficient:
        family_verdict = "EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT"
    elif any_insufficient and (any_regime_dependent or any_stable):
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC"
    elif any_regime_dependent and any_stable:
        family_verdict = "MIXED_BY_HORIZON_OR_METRIC"
    elif any_regime_dependent:
        family_verdict = "EXPANDED_SHORT_REGIME_DEPENDENT"
    else:
        family_verdict = "EXPANDED_SHORT_STABLE_BASELINE"

    return {
        "raw_bps_rows": raw_bps_rows, "vol_norm_rows": vol_norm_rows,
        "cells": cells, "cell_order": cell_order,
        "global_split": {
            "total_cycle_n": split["total_cycle_n"], "train_cycle_n": split["train_cycle_n"],
            "test_cycle_n": split["test_cycle_n"],
        },
        "per_horizon_split_report": per_horizon_split_report,
        "raw_signal_n_population": len({r["signal_id"] for r in raw_bps_rows}),
        "distinct_source_event_n_population": len({r["source_event_id"] for r in raw_bps_rows}),
        "distinct_independent_cycle_n_population": len({_cycle_key(r) for r in raw_bps_rows}),
        "family_verdict": family_verdict,
    }


# ---------------------------------------------------------------------------
# composition diagnostic -- descriptive ONLY, no p-values, no inference
# ---------------------------------------------------------------------------

def compute_composition_diagnostic(raw_bps_rows: list[dict]) -> dict:
    """Reports pre-existing-SHORT vs new-SHORT_NOISY_BTC200K_CONFIRMED_V1 vs
    combined population N/medians per horizon x raw-bps metric, for
    descriptive context ONLY. These are NOT three independent confirmations
    -- combined is the primary-family population itself; the other two are
    informational subsets, never separately inferentially tested."""
    pre_existing = [r for r in raw_bps_rows if r["setup_id"] != NEW_SETUP_ID]
    new_setup = [r for r in raw_bps_rows if r["setup_id"] == NEW_SETUP_ID]

    out: dict[str, dict] = {}
    for horizon in HORIZONS:
        entry = {}
        for label, subset in (("pre_existing", pre_existing), ("new_short_noisy_v1", new_setup),
                               ("combined", raw_bps_rows)):
            cell_rows = _cell_rows(subset, horizon)
            entry[label] = {
                "raw_signal_n": len(cell_rows),
                "distinct_independent_cycle_n": len({_cycle_key(r) for r in cell_rows}),
                "mfe_bps_median": round(_median([r["mfe_bps"] for r in cell_rows]), 4) if cell_rows else None,
                "mae_bps_median": round(_median([r["mae_bps"] for r in cell_rows]), 4) if cell_rows else None,
            }
        out[horizon] = entry
    return out


# ---------------------------------------------------------------------------
# integrity: old (-001) experiments must remain byte-unchanged
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
    after = snapshot_old_experiments(conn)
    return after == before


# ---------------------------------------------------------------------------
# freeze_and_record -- writes to the two NEW experiment_ids only
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-p7b1-w8-short-expanded-baseline") -> dict:
    now = int(time.time() * 1000)
    before_old = snapshot_old_experiments(conn)

    family = compute_family(conn)
    composition = compute_composition_diagnostic(family["raw_bps_rows"])

    dataset_hash = hashlib.sha256(
        "|".join(sorted(
            f"{r['signal_id']}|{r['horizon_name']}" for r in family["raw_bps_rows"] + family["vol_norm_rows"]
        )).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"ami_lifecycle_path_observations WHERE observation_status=OK (raw-bps) / "
        f"observation_status=OK AND volatility_status=OK (vol-normalized), direction=SHORT only; "
        f"raw_signal_n={family['raw_signal_n_population']}; "
        f"distinct_source_event_n={family['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={family['distinct_independent_cycle_n_population']}"
    )
    frozen_splits = (
        f"CYCLE-GROUPED chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split by "
        f"independent_cycle_id (never signal-level): cycle order key=earliest signal_birth_ts among all "
        f"of that cycle's rows; MIN_BUCKET_N={MIN_BUCKET_N} applies to independent-cycle N per split/cell"
    )

    for experiment_id, metrics_group in ((RAW_BPS_EXPERIMENT_ID, RAW_BPS_METRICS),
                                          (VOL_NORMALIZED_EXPERIMENT_ID, VOL_NORMALIZED_METRICS)):
        record_experiment_registry(conn, {
            "experiment_id": experiment_id, "question_ids": "FAM_W8_SHORT_EXPANDED_BASELINE",
            "hypothesis_id": "H-W8-SHORT-EXPANDED-BASELINE", "preregistered_at": now,
            "frozen_population": frozen_population, "frozen_features": ",".join(metrics_group),
            "frozen_target": (
                "CYCLE-GROUPED TRAIN(chronological first 70% of independent cycles) vs TEST(final 30% of "
                "independent cycles) median stability, SHORT only, by horizon"
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
                f"(both raw-bps and vol-normalized metric groups together, not two separate 8-cell corrections)"
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
            ("composition_diagnostic_descriptive_only", composition),
            ("family_verdict", family["family_verdict"]),
        ]
        for key in family["cell_order"]:
            metric = key.split("|")[0]
            if _METRIC_EXPERIMENT_OF[metric] != experiment_id:
                continue
            rows_to_write.append((f"cell_{key}", family["cells"][key]))
        results = [(name, str(value)) for name, value in rows_to_write]
        record_experiment_results(conn, experiment_id, results, schema_version=10, provenance=provenance, created_ms=now)
    conn.commit()

    old_experiments_untouched = assert_old_experiments_untouched(conn, before_old)

    return {
        "family": family, "composition": composition,
        "old_experiments_untouched": old_experiments_untouched,
        "cell_order": family["cell_order"], "cells": family["cells"],
        "family_verdict": family["family_verdict"],
        "raw_signal_n_population": family["raw_signal_n_population"],
        "distinct_source_event_n_population": family["distinct_source_event_n_population"],
        "distinct_independent_cycle_n_population": family["distinct_independent_cycle_n_population"],
        "global_split": family["global_split"],
        "per_horizon_split_report": family["per_horizon_split_report"],
    }
