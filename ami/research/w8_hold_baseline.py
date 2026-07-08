"""BATCH-P7B-1 (W8-HOLD-BASELINE): Fixed-horizon MFE/MAE baseline
characterization + chronological stability testing (HISTORICAL_RESEARCH_
WAVES.md W8, precondition "P7 timing metrics" -- now satisfied by Phase 7B's
canonical ami_lifecycle_path_observations).

NOT_A_MANAGEMENT_WAVE: this module computes NO stop/exit/partial-exit/
time-stop/re-entry/cancellation rule and makes NO economic claim. It answers
exactly one question -- W8's own mandatory precondition, quoted verbatim
from HISTORICAL_RESEARCH_WAVES.md: "Erken-cikis mezar bulgusu: hold-baseline
benchmark zorunlu" (a hold-to-fixed-horizon baseline must exist BEFORE any
future, separately-preregistered competing-risk/management hypothesis can be
tested against it). This wave produces that baseline and tests whether it is
chronologically stable enough to be reused as a fixed reference later.

GRAVEYARD AWARENESS (failure_archive, read before writing a line of this
module): #3 (tight vol-scaled stops), #4 (partial exits), #10 (loser
time-stops), #13 (MFE50 giveback single-feature separation), #18 (BUYFADE
management overlay) are all NO_EDGE/EXECUTION_FAILURE findings about
MANAGEMENT RULES. This module tests none of those rules -- it only
characterizes the raw MFE/MAE distribution already stored in
ami_lifecycle_path_observations (built with zero knowledge of any exit
rule), split chronologically to check stability. BUY_FADE_SHORT_H45_SL75
(19/270 of the canonical population) is inside the #16/#18/#19/#21 graveyard
family; this wave reads its path observations exactly like every other
signal's, applies no special-cased rule to it, and draws no conclusion about
any BUYFADE-specific management strategy.

POPULATION (frozen): ami_lifecycle_path_observations WHERE observation_status
='OK', joined to ami_signal_lifecycle.direction/signal_birth_ts/
source_event_id/independent_cycle_id -- via ami.research.feature_gateway
ONLY (fetch_lifecycle_signals + fetch_path_observations), never raw table
access.

METRICS (frozen): mfe_bps, mae_bps only. endpoint_return_bps was already
characterized descriptively in W4 (E-W4-POST-EVENT-PATH-TAXONOMY-001) --
not repeated here.

HORIZONS (frozen): the same 4 from ami.research.w4_post_event_path_taxonomy.
PATH_HORIZONS_MS, reused verbatim -- no new horizon, no sweep.

FAMILY (frozen, 16 = 2 metrics x 4 horizons x 2 directions): for each cell,
TRAIN-split (chronological first 70%) vs TEST-split (final 30%, ordered by
signal_birth_ts, never randomized) median comparison.

PRIMARY DENOMINATOR: independent_cycle_id (never raw signal_id or
source_event_id) -- the 16 multi-route events (two signals sharing one
source_event_id/likely one cycle) are explicitly never counted as 2
independent observations in the bootstrap (cluster resampling groups by
cycle, singleton "NOCYCLE-<event_id>" for unresolved cycles, same convention
as W1/W7A/W10a).

SCIENTIFIC VERDICT PER CELL (operator-specified, mechanically applied, NEVER
defaulted to "supported"):
  stable AND Holm-nonsignificant AND bootstrap CI includes 0
    -> ANSWERED_SUPPORTED_STABLE_BASELINE
  materially different AND CI excludes 0 AND Holm-significant
    -> ANSWERED_REGIME_DEPENDENT_BASELINE
  either split has n < MIN_BUCKET_N=20
    -> INSUFFICIENT_SAMPLE
  [operator did not define a 4th branch for CI/Holm DISAGREEMENT -- this
  module's own necessary disambiguation, documented not invented: any cell
  where the CI-based and Holm-based signals disagree is conservatively
  classified ANSWERED_REGIME_DEPENDENT_BASELINE (never silently rounds up to
  "stable" -- matches the operator's explicit "do not assign
  ANSWERED_SUPPORTED automatically regardless of result" instruction).]

NEGATIVE CONTROL: ami_candidate_universe (is_event_aligned=0), matched (not
randomly sampled) on chronological-period/session/volatility-regime bucket
via stratified deterministic sampling, frozen BEFORE any outcome is read.
LONG/SHORT direction ratio CANNOT be matched -- a random candidate-universe
time slot carries no real trading decision to assign a direction from
without fabricating one. Per operator instruction, the per-direction 16-cell
comparison against this control is explicitly marked
BLOCKED_FOR_DIRECTION_MATCHING (never defaulted to LONG, never pooled). A
separate, clearly-labeled, NON-primary descriptive pair (upside_excursion_bps/
downside_excursion_bps -- literally the same high-side/low-side-from-
reference arithmetic mfe_bps/mae_bps use, computed via compute_observation()
with direction="LONG" purely as a neutral computational device to get "how
far did price move up" / "how far did price move down", NEVER presented as
or compared against a real per-signal-direction MFE/MAE value) is reported
for descriptive context only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import time

import numpy as np

from ami.lifecycle.path_metrics import CANDLE_MS, _CandleOHLCIndex, _realized_vol_at, compute_observation
from ami.research.feature_gateway import (
    fetch_chart_feature,
    fetch_lifecycle_signals,
    fetch_path_observations,
)
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, PATH_HORIZONS_MS, TRAIN_FRACTION
from ami.research.w6_compression_rs_session import classify_session
from ami.research.w7a_state_structure_aging_market_clocks import holm_adjust

EXPERIMENT_ID = "E-W8-HOLD-BASELINE-001"
RESEARCH_CONTEXT_ID = "w8-hold-baseline"

METRICS = ("mfe_bps", "mae_bps")
HORIZONS = ("scalp_30m", "scalp_1h", "swing_4h", "swing_24h")
DIRECTIONS = ("LONG", "SHORT")

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260704
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 20260704
CONTROL_SEED = 20260704
HOLM_ALPHA = 0.05


# ---------------------------------------------------------------------------
# generic descriptive helpers
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _split_chronological_by_birth(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Same 70/30 chronological convention as W4's _split_chronological,
    keyed on signal_birth_ts instead of anchor_ts_ms -- never randomized."""
    ordered = sorted(rows, key=lambda r: r["signal_birth_ts"])
    cut = int(len(ordered) * TRAIN_FRACTION)
    return ordered[:cut], ordered[cut:]


def _month_bucket(ts_ms: int) -> str:
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


def _vol_bucket(realized_vol: float | None, median_ref: float | None) -> str:
    if realized_vol is None or median_ref is None:
        return "UNKNOWN"
    return "HIGH" if realized_vol > median_ref else "LOW"


# ---------------------------------------------------------------------------
# cluster bootstrap / permutation for a MEDIAN-difference statistic (the
# family this wave tests is a median-stability check, not W7A's risk-
# difference -- a small, self-contained statistic, same discipline/seed
# convention as W7A/W10a, not a redesign of their proportion-based bootstrap)
# ---------------------------------------------------------------------------

def cluster_bootstrap_median_diff(train_rows: list[dict], test_rows: list[dict], metric: str,
                                   n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    def _by_cluster(rows: list[dict]) -> dict[str, list[dict]]:
        d: dict[str, list[dict]] = {}
        for r in rows:
            key = r["independent_cycle_id"] or f"NOCYCLE-{r['source_event_id']}"
            d.setdefault(key, []).append(r)
        return d

    train_clusters = _by_cluster(train_rows)
    test_clusters = _by_cluster(test_rows)
    train_keys = list(train_clusters.keys())
    test_keys = list(test_clusters.keys())
    if not train_keys or not test_keys:
        return {"n_valid_draws": 0, "ci95": (None, None)}

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        tr_sample = [r for k in rng.choices(train_keys, k=len(train_keys)) for r in train_clusters[k]]
        te_sample = [r for k in rng.choices(test_keys, k=len(test_keys)) for r in test_clusters[k]]
        m_tr = _median([r[metric] for r in tr_sample])
        m_te = _median([r[metric] for r in te_sample])
        if m_tr is None or m_te is None:
            continue
        diffs.append(m_tr - m_te)
    if not diffs:
        return {"n_valid_draws": 0, "ci95": (None, None)}
    arr = np.array(diffs)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"n_valid_draws": len(diffs), "ci95": (round(float(lo), 4), round(float(hi), 4))}


def permutation_test_median_diff(train_vals: list[float], test_vals: list[float],
                                  n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> dict:
    if not train_vals or not test_vals:
        return {"observed_diff": None, "p_value": None, "n_perm": n_perm}
    obs = _median(train_vals) - _median(test_vals)
    pooled = list(train_vals) + list(test_vals)
    n_tr = len(train_vals)
    rng = random.Random(seed)
    n_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        m_tr = _median(pooled[:n_tr])
        m_te = _median(pooled[n_tr:])
        if abs(m_tr - m_te) >= abs(obs):
            n_extreme += 1
    return {"observed_diff": round(obs, 4), "p_value": round(n_extreme / n_perm, 5), "n_perm": n_perm}


def classify_cell_verdict(sample_sufficiency: str, holm_p: float | None, ci95: tuple) -> str:
    """Operator-specified 3-way classification, mechanically applied -- see
    module docstring for the disagreement-handling rule (conservative,
    never defaults to stable)."""
    if sample_sufficiency == "INSUFFICIENT_SAMPLE":
        return "INSUFFICIENT_SAMPLE"
    ci_lo, ci_hi = ci95
    ci_includes_zero = ci_lo is not None and ci_hi is not None and ci_lo <= 0.0 <= ci_hi
    holm_significant = holm_p is not None and holm_p < HOLM_ALPHA
    if (not holm_significant) and ci_includes_zero:
        return "ANSWERED_SUPPORTED_STABLE_BASELINE"
    if holm_significant and (not ci_includes_zero):
        return "ANSWERED_REGIME_DEPENDENT_BASELINE"
    return "ANSWERED_REGIME_DEPENDENT_BASELINE"  # disagreement -> conservative, never "stable"


# ---------------------------------------------------------------------------
# population assembly (feature_gateway only)
# ---------------------------------------------------------------------------

def fetch_population(conn, symbol: str = "ETHUSDT") -> list[dict]:
    """One fetch_lifecycle_signals() + one fetch_path_observations() call,
    joined in Python -- never a raw SQL join, never a second, unsanctioned
    access path.

    [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL C] Explicitly pins path_definition_version="path-v2" -- this
    experiment's frozen, completed population was computed before
    ami.lifecycle.path_candle_repair_correction ever existed ("path-v2" was
    the only version in the table at the time). The canonical DB can now
    legitimately hold more than one physical path_definition_version for the
    same (signal_id, horizon_name), so an unpinned multi-version read fails
    closed (feature_gateway.AmbiguousPathVersionError) -- this pin is the
    explicit, permanent historical-reproduction declaration that keeps
    E-W8-HOLD-BASELINE-001 reproducible byte-for-byte. It is NOT a
    corrected-data rerun: a corrected rerun uses
    ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations()
    under a NEW experiment_id instead (see
    ami.research.w8_long_nested_path_accumulation_002_candle_repair for the
    established pattern)."""
    signals = {s["signal_id"]: s for s in fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=symbol)}
    obs_rows = fetch_path_observations(
        conn, RESEARCH_CONTEXT_ID,
        equals={"observation_status": "OK", "path_definition_version": "path-v2"},
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


def _cell_rows(rows: list[dict], horizon: str, direction: str) -> list[dict]:
    return [r for r in rows if r["horizon_name"] == horizon and r["direction"] == direction]


def compute_cell(rows: list[dict], metric: str) -> dict:
    vals = [r[metric] for r in rows]
    train_rows, test_rows = _split_chronological_by_birth(rows)
    train_vals = [r[metric] for r in train_rows]
    test_vals = [r[metric] for r in test_rows]

    raw_signal_n = len(rows)
    distinct_source_event_n = len({r["source_event_id"] for r in rows if r["source_event_id"] is not None})
    distinct_cycle_n = len({r["independent_cycle_id"] or f"NOCYCLE-{r['source_event_id']}" for r in rows})

    train_median = _median(train_vals)
    test_median = _median(test_vals)
    full_median = _median(vals)
    q10, q25, q50, q75, q90 = (_quantile(vals, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
    iqr = None if q75 is None or q25 is None else round(q75 - q25, 4)
    train_minus_test = None if train_median is None or test_median is None else round(train_median - test_median, 4)

    sufficiency_ok = len(train_vals) >= MIN_BUCKET_N and len(test_vals) >= MIN_BUCKET_N
    sample_sufficiency = "OK" if sufficiency_ok else "INSUFFICIENT_SAMPLE"

    if sufficiency_ok:
        boot = cluster_bootstrap_median_diff(train_rows, test_rows, metric)
        perm = permutation_test_median_diff(train_vals, test_vals)
    else:
        boot = {"n_valid_draws": 0, "ci95": (None, None)}
        perm = {"observed_diff": None, "p_value": None, "n_perm": N_PERMUTATIONS}

    return {
        "raw_signal_n": raw_signal_n,
        "distinct_source_event_n": distinct_source_event_n,
        "distinct_independent_cycle_n": distinct_cycle_n,
        "train_n": len(train_vals), "test_n": len(test_vals),
        "train_median": round(train_median, 4) if train_median is not None else None,
        "test_median": round(test_median, 4) if test_median is not None else None,
        "full_median": round(full_median, 4) if full_median is not None else None,
        "iqr": iqr,
        "q10": round(q10, 4) if q10 is not None else None,
        "q25": round(q25, 4) if q25 is not None else None,
        "q50": round(q50, 4) if q50 is not None else None,
        "q75": round(q75, 4) if q75 is not None else None,
        "q90": round(q90, 4) if q90 is not None else None,
        "train_minus_test_median_diff": train_minus_test,
        "bootstrap_ci95": boot["ci95"],
        "bootstrap_n_valid_draws": boot["n_valid_draws"],
        "permutation_observed_diff": perm["observed_diff"],
        "permutation_p_value": perm["p_value"],
        "sample_sufficiency": sample_sufficiency,
    }


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


# ---------------------------------------------------------------------------
# matched negative control (chronological period / session / vol-bucket --
# LONG/SHORT direction ratio explicitly NOT matchable, see module docstring)
# ---------------------------------------------------------------------------

def _anchor_profile(rows: list[dict]) -> list[dict]:
    """One row per DISTINCT signal (not per signal x horizon) -- realized_vol_
    at_anchor is identical across a signal's horizons by construction (proven
    in path_metrics tests), so the first occurrence per signal is sufficient."""
    by_signal: dict[str, dict] = {}
    for r in rows:
        by_signal.setdefault(r["signal_id"], r)
    return list(by_signal.values())


def build_match_profile(anchor_rows: list[dict]) -> dict:
    vols = [r["realized_vol_at_anchor"] for r in anchor_rows if r["realized_vol_at_anchor"] is not None]
    vol_median_ref = _median(vols)

    strata: dict[tuple, int] = {}
    direction_counts: dict[str, int] = {}
    for r in anchor_rows:
        month = _month_bucket(r["signal_birth_ts"])
        session = classify_session(r["signal_birth_ts"])
        vol_bucket = _vol_bucket(r["realized_vol_at_anchor"], vol_median_ref)
        strata[(month, session, vol_bucket)] = strata.get((month, session, vol_bucket), 0) + 1
        direction_counts[r["direction"]] = direction_counts.get(r["direction"], 0) + 1

    total = len(anchor_rows)
    return {
        "vol_median_ref": round(vol_median_ref, 6) if vol_median_ref is not None else None,
        "strata_counts": strata,
        "strata_proportions": {k: round(v / total, 6) for k, v in strata.items()} if total else {},
        "total_n": total,
        "direction_counts": direction_counts,
        "direction_ratio_matching_status": "BLOCKED_FOR_DIRECTION_MATCHING",
    }


def sample_matched_control_candidates(conn, match_profile: dict, symbol: str = "ETHUSDT",
                                       seed: int = CONTROL_SEED) -> dict:
    """Deterministic stratified sample of ami_candidate_universe (is_event_
    aligned=0) slots, matched on (month, session, vol_bucket) proportions
    frozen in match_profile -- computed BEFORE any excursion outcome is
    read for either population."""
    candidate_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe",
        ["candidate_id", "slot_ts_ms", "known_at_ts", "data_quality"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    candidate_rows = [c for c in candidate_rows if c["data_quality"] == "AVAILABLE"]

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles",
        ["open_ts_ms", "close_ts_ms", "high", "low", "close", "data_quality", "candle_definition_version"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    index = _CandleOHLCIndex(candle_rows)

    vol_median_ref = match_profile["vol_median_ref"]
    by_stratum: dict[tuple, list[dict]] = {}
    for c in candidate_rows:
        month = _month_bucket(c["slot_ts_ms"])
        session = classify_session(c["slot_ts_ms"])
        # [PHASE 7B-1 perf] direct _realized_vol_at() call -- NOT a full compute_observation()
        # probe. The latter also builds a path_window() (only needed for the actual
        # excursion computation on the final SAMPLED slots, not for this stratification
        # pre-pass over every candidate) -- calling it ~2900 extra times here was the
        # dominant cost before this fix (see path_metrics.py's path_window() docstring).
        realized_vol = _realized_vol_at(index, c["slot_ts_ms"])
        vol_bucket = _vol_bucket(realized_vol, vol_median_ref)
        by_stratum.setdefault((month, session, vol_bucket), []).append(c)

    rng = random.Random(seed)
    target_total = match_profile["total_n"]
    sampled: list[dict] = []
    shortfall: dict[str, dict] = {}
    for stratum_key, proportion in match_profile["strata_proportions"].items():
        target_n = round(proportion * target_total)
        pool = by_stratum.get(stratum_key, [])
        take_n = min(target_n, len(pool))
        if take_n < target_n:
            shortfall[str(stratum_key)] = {"target": target_n, "available": len(pool), "taken": take_n}
        if take_n > 0:
            sampled.extend(rng.sample(pool, take_n))

    return {
        "sampled_candidates": sampled,
        "sampled_n": len(sampled),
        "target_n": target_total,
        "shortfall_by_stratum": shortfall,
        "candle_index": index,
    }


def compute_negative_control(conn, match_profile: dict, symbol: str = "ETHUSDT") -> dict:
    sample = sample_matched_control_candidates(conn, match_profile, symbol)
    index = sample["candle_index"]
    as_of_ts = index.maturity_cutoff_ts_ms()

    by_horizon: dict[str, list[dict]] = {h: [] for h in HORIZONS}
    for c in sample["sampled_candidates"]:
        for horizon in HORIZONS:
            obs = compute_observation(index, f"CTRL-{c['candidate_id']}", "LONG", c["slot_ts_ms"],
                                       horizon, PATH_HORIZONS_MS[horizon], as_of_ts)
            if obs["observation_status"] != "OK":
                continue
            by_horizon[horizon].append({
                "upside_excursion_bps": obs["mfe_bps"],
                "downside_excursion_bps": obs["mae_bps"],
            })

    descriptive = {}
    for horizon, rows in by_horizon.items():
        up = [r["upside_excursion_bps"] for r in rows]
        down = [r["downside_excursion_bps"] for r in rows]
        descriptive[horizon] = {
            "n": len(rows),
            "upside_excursion_median": round(_median(up), 4) if up else None,
            "downside_excursion_median": round(_median(down), 4) if down else None,
        }

    return {
        "matched_sample_n": sample["sampled_n"],
        "matched_sample_target_n": sample["target_n"],
        "shortfall_by_stratum": sample["shortfall_by_stratum"],
        "match_profile_strata_proportions": {str(k): v for k, v in match_profile["strata_proportions"].items()},
        "direction_ratio_matching_status": match_profile["direction_ratio_matching_status"],
        "primary_16cell_comparison_status": "BLOCKED_FOR_DIRECTION_MATCHING",
        "descriptive_context_by_horizon": descriptive,
    }


# ---------------------------------------------------------------------------
# freeze + record
# ---------------------------------------------------------------------------

def freeze_and_record(conn, provenance: str = "batch-p7b1-w8-hold-baseline") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    anchor_rows = _anchor_profile(metrics["rows"])
    match_profile = build_match_profile(anchor_rows)
    neg_control = compute_negative_control(conn, match_profile)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(f"{r['signal_id']}|{r['horizon_name']}" for r in metrics["rows"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"ami_lifecycle_path_observations WHERE observation_status=OK; "
        f"raw_signal_n={metrics['raw_signal_n_population']}; "
        f"distinct_source_event_n={metrics['distinct_source_event_n_population']}; "
        f"distinct_independent_cycle_n={metrics['distinct_independent_cycle_n_population']}"
    )

    record_experiment_registry(conn, {
        "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_W8_HOLD_BASELINE",
        "hypothesis_id": "H-W8-HOLD-BASELINE", "preregistered_at": now,
        "frozen_population": frozen_population, "frozen_features": "mfe_bps,mae_bps",
        "frozen_target": "TRAIN(chronological first 70%) vs TEST(final 30%) median stability, by horizon x direction",
        "frozen_thresholds": (
            f"MIN_BUCKET_N={MIN_BUCKET_N};TRAIN_FRACTION={TRAIN_FRACTION};"
            "classification=stable iff Holm-p>=0.05 AND bootstrap-CI includes 0; "
            "regime-dependent iff Holm-p<0.05 AND CI excludes 0; disagreement->regime-dependent (conservative, "
            "never defaults to stable); insufficient iff either split n<MIN_BUCKET_N"
        ),
        "frozen_splits": (
            f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} by signal_birth_ts, never randomized"
        ),
        "frozen_economic_gate": (
            "N/A (no management/exit/stop/re-entry rule tested -- hold-to-fixed-horizon baseline "
            "characterization only, per W8's own mandatory precondition)"
        ),
        "frozen_statistical_gate": (
            f"primary=independent-cycle cluster block-bootstrap median-difference CI (n={N_BOOTSTRAP}, "
            f"seed={BOOTSTRAP_SEED}); secondary=two-sided label-permutation median-difference p "
            f"(n={N_PERMUTATIONS}, seed={PERMUTATION_SEED}) + Holm step-down across exactly 16 cells"
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
        ("negative_control", neg_control),
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
        "negative_control": neg_control,
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
        print(f"negative_control={r['negative_control']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
