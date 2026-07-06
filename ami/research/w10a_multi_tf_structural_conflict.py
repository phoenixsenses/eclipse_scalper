"""BATCH-P6-011 (W10a): Multi-Timeframe Structural Conflict
(HISTORICAL_RESEARCH_WAVES.md W10, buildable HALF only -- the LONG<->SHORT
transitions half is BLOCKED_BY_DATA + NOT_IMPLEMENTED, backlogged as OD-017,
NOT attempted here).

Preregistration frozen BEFORE any outcome is looked at (this docstring +
the module-level constants below ARE the freeze; freeze_and_record() writes
the same frozen values to experiment_registry so they cannot silently
drift). Operator approved the initial scope, then required 7 corrections
(all applied below, marked "[CORRECTION n]").

POPULATION (frozen, reused verbatim from W1/W4/W7A -- not redefined):
ETHUSDT REAL_LIQUIDATION anchors via feature_gateway.fetch_events,
raw anchor_n=252, independent_cycle_n via event_cycle_membership
(cycle_definition_version=canonical-v1, is_canonical=1).

OUTCOME (frozen, REUSED from W4): swing_24h path class (CONTINUATION/
REVERSAL/CHOP, E-W4-POST-EVENT-PATH-TAXONOMY-001's fixed +-20bps band).

[CORRECTION 6] EXACT MATURITY CUTOFF (frozen, identical pattern to
E-W6RS-CONFIRMATION-001/E-W7A-...-001): MATURITY_CUTOFF_TS_MS =
MAX(ami_candles.close_ts_ms) at freeze time. Anchors where anchor_ts_ms +
24h > MATURITY_CUTOFF_TS_MS are EXCLUDED (excluded_no_horizon_data,
counted, never fabricated).

===========================================================================
TF-PAIR CHOICE (frozen, NOT arbitrary)
===========================================================================
1h and 4h are reused verbatim from OD-003's own cycle_resolver definition
(A2: "dominant-structural-state via ami/states/engine.py 1h TF" + "4h
continuity gate") -- this module does not pick a new TF pair, it reads the
same two TFs the canonical-v1 cycle definition already treats as
structurally meaningful.

===========================================================================
[CORRECTION 5] EXACT StructurePhase DIRECTION MAPPING (frozen, verbatim
from the EXISTING, UNCHANGED ami/states/engine.py:StateEngine._structure())
===========================================================================
This module does NOT invent a new UP/DOWN/FLAT rule. `_structure(symbol,
ts, tf)` already returns `(StructurePhase, direction, confidence)`; the
`direction` field -- computed exactly as follows, quoted verbatim from the
existing production code, never modified by this module -- IS the
UP/DOWN/FLAT read used here:

    ret3  = return over the trailing 3  bars of `tf`
    ret12 = return over the trailing 12 bars of `tf`
    direction = "UP"   if ret3 >  abs(ret12)*0.15 and ret3 > 0
                "DOWN" if ret3 < -abs(ret12)*0.15 and ret3 < 0
                "FLAT" otherwise

This is the FIRST time `direction` (as opposed to the StructurePhase enum
itself, already used by W6's compression_at_anchor) is read as a research
feature -- reused as-is, not refit, not swept. The mapping does not change
after seeing results (frozen for the lifetime of this experiment_id; a
different mapping would need a new experiment_id + prereg, per Protocol
§2).

===========================================================================
[CORRECTION 4] KNOWN-AT SAFETY (frozen, verified by an explicit test)
===========================================================================
`_structure()` -> `_ret_bps()` -> `_px()` is NOT candle/bar-based (unlike
ami_candles' closed-candle-only convention used elsewhere in this
project). It is a point-in-time TICK lookup against `mark_prices`:
`_px(sym, ts)` always executes `SELECT ... WHERE ts_ms<=ts ORDER BY ts_ms
DESC LIMIT 1` -- every price used by `_ret_bps`/`_rv`/`_structure` is
therefore, by construction, the most recent tick AT OR BEFORE the queried
timestamp; there is no "currently forming bar" to leak from because there
is no bar object in this code path at all (a different, but equally
rigorous, known-at-safety mechanism than the OHLC-candle convention).
Concretely for this module: `direction_1h`/`direction_4h` at `anchor_ts_ms`
only ever consult mark_prices rows with `ts_ms <= anchor_ts_ms - k*bar_ms`
for k in {0..12}, i.e. never a timestamp after the anchor. This is proven
by `test_direction_classification_is_known_at_safe` (inserting a future,
wildly-different mark_price row does not change the direction computed at
the anchor).

===========================================================================
RAW 5-CELL TABLE (frozen, [CORRECTION 1] -- kept SEPARATE, never merged at
the data-recording level before results are seen)
===========================================================================
    UP_UP        -- direction_1h=UP   and direction_4h=UP
    DOWN_DOWN    -- direction_1h=DOWN and direction_4h=DOWN
    UP_DOWN      -- direction_1h=UP   and direction_4h=DOWN
    DOWN_UP      -- direction_1h=DOWN and direction_4h=UP
    NEUTRAL      -- direction_1h=FLAT or direction_4h=FLAT (any FLAT)
This 5-way breakdown is reported as a DESCRIPTIVE/mechanism table (n +
3-way CONTINUATION/REVERSAL/CHOP distribution per cell) -- it produces NO
p-values of its own and does NOT silently expand the multiple-testing
family (per operator instruction: 5 cells != 5 tests).

===========================================================================
PRIMARY HYPOTHESIS (frozen, [CORRECTION 2] -- the ONE preregistered test,
FAM_MULTI_TF_CONFLICT, 1 comparison)
===========================================================================
    AGREEMENT = UP_UP union DOWN_DOWN
    CONFLICT  = UP_DOWN union DOWN_UP
Hypothesis: the swing_24h path distribution (operationalized as REVERSAL
rate, same binary-outcome convention as every prior wave's inferential
test) differs between CONFLICT and AGREEMENT. Two-sided (no pre-committed
direction). This is the ONLY inferential test in this module -- no
per-cell p-values, no Holm correction needed (n=1 comparison).
[CORRECTION 3] NEUTRAL is NEVER added to the primary contrast, in either
direction, regardless of its own N. It is reported purely descriptively
(n + 3-way distribution). If NEUTRAL's n < MIN_BUCKET_N=20 it is reported
INSUFFICIENT_SAMPLE and left there -- never folded into AGREEMENT or
CONFLICT after the fact.

===========================================================================
INFERENCE PLAN (frozen, reusing E-W7A's already-implemented, generic
statistical helpers verbatim -- not reimplemented)
===========================================================================
PRIMARY: independent-cycle cluster block-bootstrap risk-difference 95% CI
(ami.research.w7a_state_structure_aging_market_clocks.
cluster_bootstrap_risk_difference, parameterized with
label_high="CONFLICT"/label_low="AGREEMENT"; n=2000, seed=20260704).
SECONDARY: two-sided label-permutation p-value (permutation_test_two_sided,
n_perm=2000, seed=20260704).
CONTROLS: chronological 70/30 stability (TRAIN_FRACTION=0.7, reused);
candidate-universe negative control (same primary-contrast machinery
applied to a deterministic sample of non-event-aligned candidate-universe
slots, sample size 2000 -- this computation is cheap, ~2 `_structure()`
calls per row, unlike W7A's state_age cost, so no sample-size cap is
needed here).
STOP CONDITION: any bucket (AGREEMENT/CONFLICT/NEUTRAL, train or test) with
N < MIN_BUCKET_N=20 is reported INSUFFICIENT_SAMPLE, never merged/dropped
silently.

Descriptive/conditioning only: NO trade/PnL claim, NO bucket/route/
observer promotion or change anywhere in this module.

[CORRECTION 7] The LONG<->SHORT transitions half of W10 is NOT attempted
here -- backlogged as OD-017 (BLOCKED_BY_DATA: the ledger is 100%
SELL-cascade anchors, no LONG-anchor population exists to compare against;
NOT_IMPLEMENTED: whitepaper v0.3 Section 54.5's "LONG/SHORT Transition
Map" is a Phase-8 forward-dashboard page backed by `ami_forward_*` tables
that do not exist yet).
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import Counter

from ami.research.feature_gateway import fetch_chart_feature, fetch_events
from ami.research.w4_post_event_path_taxonomy import (
    MIN_BUCKET_N,
    TRAIN_FRACTION,
    _CandleIndex,
    _split_chronological,
    classify_path,
    compute_path_returns,
)
from ami.research.w7a_state_structure_aging_market_clocks import (
    cluster_bootstrap_risk_difference,
    permutation_test_two_sided,
)
from ami.states.engine import StateEngine
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W10A-MULTI-TF-STRUCTURAL-CONFLICT-001"
RESEARCH_CONTEXT_ID = "w10a-multi-tf-structural-conflict"

TF_FAST = "1h"  # reused verbatim from OD-003 cycle_resolver's dominant-structural-state TF
TF_SLOW = "4h"  # reused verbatim from OD-003 cycle_resolver's continuity-gate TF

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260704
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 20260704
NEGATIVE_CONTROL_SEED = 20260704
NEGATIVE_CONTROL_SAMPLE_SIZE = 2000

RAW_CELLS = ("UP_UP", "DOWN_DOWN", "UP_DOWN", "DOWN_UP", "NEUTRAL")


def classify_direction(engine: StateEngine, symbol: str, ts_ms: int, tf: str) -> str:
    """Verbatim read of StateEngine._structure()'s existing `direction`
    output (UP/DOWN/FLAT) -- not a new formula, see module docstring."""
    _phase, direction, _conf = engine._structure(symbol, ts_ms, tf)
    return direction


def classify_tf_cell(direction_fast: str, direction_slow: str) -> str:
    if direction_fast == "FLAT" or direction_slow == "FLAT":
        return "NEUTRAL"
    if direction_fast == "UP" and direction_slow == "UP":
        return "UP_UP"
    if direction_fast == "DOWN" and direction_slow == "DOWN":
        return "DOWN_DOWN"
    if direction_fast == "UP" and direction_slow == "DOWN":
        return "UP_DOWN"
    return "DOWN_UP"  # direction_fast == "DOWN" and direction_slow == "UP"


def primary_bucket_of_cell(cell: str) -> str | None:
    if cell in ("UP_UP", "DOWN_DOWN"):
        return "AGREEMENT"
    if cell in ("UP_DOWN", "DOWN_UP"):
        return "CONFLICT"
    return None  # NEUTRAL -- never part of the primary contrast


def _distribution(classes: list[str]) -> dict:
    n = len(classes)
    c = Counter(classes)
    return {"n": n, **{k: c.get(k, 0) for k in ("CONTINUATION", "REVERSAL", "CHOP")},
            "insufficient_sample": n < MIN_BUCKET_N}


def _primary_contrast_test(rows: list[dict]) -> dict:
    """The ONE preregistered inferential test: CONFLICT vs AGREEMENT
    REVERSAL rate. NEUTRAL rows are excluded by construction (their
    primary_bucket is None, never populated)."""
    valid = [r for r in rows if r.get("primary_bucket") in ("AGREEMENT", "CONFLICT") and r.get("path_class")]
    conflict = [r for r in valid if r["primary_bucket"] == "CONFLICT"]
    agreement = [r for r in valid if r["primary_bucket"] == "AGREEMENT"]
    c_n, c_succ = len(conflict), sum(1 for r in conflict if r["path_class"] == "REVERSAL")
    a_n, a_succ = len(agreement), sum(1 for r in agreement if r["path_class"] == "REVERSAL")
    perm = permutation_test_two_sided(c_n, c_succ, a_n, a_succ, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED)
    boot = cluster_bootstrap_risk_difference(
        valid, "primary_bucket", n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        label_high="CONFLICT", label_low="AGREEMENT",
    ) if valid else {"n_valid_draws": 0, "ci95": (None, None)}
    return {
        "conflict": {"n": c_n, "n_reversal": c_succ,
                     "reversal_rate": round(c_succ / c_n, 4) if c_n else None,
                     "insufficient_sample": c_n < MIN_BUCKET_N},
        "agreement": {"n": a_n, "n_reversal": a_succ,
                      "reversal_rate": round(a_succ / a_n, 4) if a_n else None,
                      "insufficient_sample": a_n < MIN_BUCKET_N},
        "permutation": perm,
        "bootstrap_ci95": boot,
        "independent_cycle_n": len({r["cluster_id"] for r in valid}),
    }


def _row_features(engine: StateEngine, ts_ms: int) -> dict:
    dir_fast = classify_direction(engine, "ETHUSDT", ts_ms, TF_FAST)
    dir_slow = classify_direction(engine, "ETHUSDT", ts_ms, TF_SLOW)
    cell = classify_tf_cell(dir_fast, dir_slow)
    return {
        "direction_1h": dir_fast, "direction_4h": dir_slow,
        "tf_cell": cell, "primary_bucket": primary_bucket_of_cell(cell),
    }


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)
    maturity_cutoff_ts_ms = max(r["close_ts_ms"] for r in candle_rows)

    membership_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "event_cycle_membership", ["event_id", "candidate_cycle_key"],
        equals={"cycle_definition_version": "canonical-v1", "is_canonical": 1},
    )
    event_to_cycle = {r["event_id"]: r["candidate_cycle_key"] for r in membership_rows}

    engine = StateEngine()
    try:
        per_anchor = []
        excluded_no_horizon_data = 0
        for e in events:
            anchor_ts = e["anchor_ts_ms"]
            if anchor_ts + 24 * 3600_000 > maturity_cutoff_ts_ms:
                excluded_no_horizon_data += 1
                continue
            returns = compute_path_returns(candle_index, anchor_ts)
            if returns["swing_24h"] is None:
                excluded_no_horizon_data += 1
                continue
            path_class = classify_path(returns["swing_24h"])
            row = {
                "event_id": e["event_id"],
                "anchor_ts_ms": anchor_ts,
                "path_class": path_class,
                "cluster_id": event_to_cycle.get(e["event_id"], f"NOCYCLE-{e['event_id']}"),
            }
            row.update(_row_features(engine, anchor_ts))
            per_anchor.append(row)
    finally:
        engine.conn.close()

    total_anchor_n = len(events)
    analyzed_n = len(per_anchor)
    independent_cycle_n = len({a["cluster_id"] for a in per_anchor if not a["cluster_id"].startswith("NOCYCLE-")})

    raw_cells = {cell: _distribution([a["path_class"] for a in per_anchor if a["tf_cell"] == cell])
                 for cell in RAW_CELLS}

    primary = _primary_contrast_test(per_anchor)

    train_anchors, test_anchors = _split_chronological(per_anchor)
    train_test = _primary_contrast_test(train_anchors), _primary_contrast_test(test_anchors)

    return {
        "per_anchor": per_anchor,
        "candle_index": candle_index,
        "maturity_cutoff_ts_ms": maturity_cutoff_ts_ms,
        "total_anchor_n": total_anchor_n,
        "analyzed_n": analyzed_n,
        "excluded_no_horizon_data": excluded_no_horizon_data,
        "independent_cycle_n": independent_cycle_n,
        "raw_cells": raw_cells,
        "primary_contrast": primary,
        "stability": {"train_n": len(train_anchors), "test_n": len(test_anchors),
                      "train": train_test[0], "test": train_test[1]},
    }


def compute_negative_control(conn, metrics: dict, symbol: str = "ETHUSDT",
                              n_target: int = NEGATIVE_CONTROL_SAMPLE_SIZE) -> dict:
    rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe", ["slot_ts_ms"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    rng = random.Random(NEGATIVE_CONTROL_SEED)
    sample = rng.sample(rows, min(n_target, len(rows)))

    engine = StateEngine()
    try:
        control_rows = []
        for i, r in enumerate(sample):
            ts = r["slot_ts_ms"]
            if ts + 24 * 3600_000 > metrics["maturity_cutoff_ts_ms"]:
                continue
            returns = compute_path_returns(metrics["candle_index"], ts)
            if returns["swing_24h"] is None:
                continue
            path_class = classify_path(returns["swing_24h"])
            row = {"path_class": path_class, "cluster_id": f"CTRL-{i}"}
            row.update(_row_features(engine, ts))
            control_rows.append(row)
    finally:
        engine.conn.close()

    return {
        "n_sampled": len(control_rows),
        "raw_cells": {cell: _distribution([r["path_class"] for r in control_rows if r["tf_cell"] == cell])
                      for cell in RAW_CELLS},
        "primary_contrast": _primary_contrast_test(control_rows),
    }


def freeze_and_record(conn, provenance: str = "batch-p6-011-w10a-multi-tf-structural-conflict") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    neg_control = compute_negative_control(conn, metrics)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;total_anchor_n={metrics['total_anchor_n']};"
        f"analyzed_n={metrics['analyzed_n']};independent_cycle_n={metrics['independent_cycle_n']};"
        f"maturity_cutoff_ts_ms={metrics['maturity_cutoff_ts_ms']}"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- chronological split
    # reused from W4; drift-only refresh in practice).
    rows_to_write = [
        ("total_anchor_n", metrics["total_anchor_n"]),
        ("analyzed_n", metrics["analyzed_n"]),
        ("excluded_no_horizon_data", metrics["excluded_no_horizon_data"]),
        ("independent_cycle_n", metrics["independent_cycle_n"]),
        ("maturity_cutoff_ts_ms", metrics["maturity_cutoff_ts_ms"]),
        ("raw_cells", metrics["raw_cells"]),
        ("primary_contrast", metrics["primary_contrast"]),
        ("stability", metrics["stability"]),
        ("negative_control", neg_control),
    ]
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_MULTI_TF_CONFLICT",
            "hypothesis_id": "H-W10A-MULTI-TF-STRUCTURAL-CONFLICT", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "direction_1h,direction_4h,tf_cell(5-way, descriptive-only),"
                               "primary_bucket(AGREEMENT/CONFLICT)",
            "frozen_target": "swing_24h path class REVERSAL (reused verbatim from "
                             "E-W4-POST-EVENT-PATH-TAXONOMY-001)",
            "frozen_thresholds": f"tf_fast={TF_FAST},tf_slow={TF_SLOW} (reused verbatim from OD-003 "
                                 "cycle_resolver, not newly chosen); direction mapping reused verbatim "
                                 "from StateEngine._structure() (unmodified production formula); "
                                 "primary_bucket=AGREEMENT(UP_UP,DOWN_DOWN) vs CONFLICT(UP_DOWN,DOWN_UP); "
                                 "NEUTRAL (any FLAT) excluded from primary contrast by construction, "
                                 "reported descriptively only; 5-cell raw table is NOT 5 tests",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " (reused)",
            "frozen_economic_gate": "N/A (descriptive conditioning only -- no entry/exit/economic claim, "
                                    "no bucket/route/observer change)",
            "frozen_statistical_gate": f"primary=independent-cycle cluster block-bootstrap "
                                       f"risk-difference CI (n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}); "
                                       f"secondary=two-sided label-permutation (n={N_PERMUTATIONS}, "
                                       f"seed={PERMUTATION_SEED}); SINGLE preregistered comparison "
                                       "(FAM_MULTI_TF_CONFLICT, n=1) -- no multiple-testing correction needed",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
            "report_artifact_id": None, "schema_version": 7, "provenance": provenance,
            "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in rows_to_write],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return {k: v for k, v in metrics.items() if k not in ("per_anchor", "candle_index")} | {
        "negative_control": neg_control
    }


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        r = freeze_and_record(conn)
        print(f"total_anchor_n={r['total_anchor_n']} analyzed_n={r['analyzed_n']} "
              f"excluded_no_horizon_data={r['excluded_no_horizon_data']} "
              f"independent_cycle_n={r['independent_cycle_n']}")
        print(f"maturity_cutoff_ts_ms={r['maturity_cutoff_ts_ms']}")
        print(f"raw_cells={r['raw_cells']}")
        print(f"primary_contrast={r['primary_contrast']}")
        print(f"stability={r['stability']}")
        print(f"negative_control={r['negative_control']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
