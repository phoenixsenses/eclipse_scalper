"""BATCH-BOOK-SPREAD-DYNAMICS-LONG-PREREGISTRATION-V1.

Outcome-blind preregistration resolution for the new LONG-only child
`H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1` of `FAM_BOOK_SPREAD_DYNAMICS`
(operator ruling: `reports/governance/FAM_BOOK_SPREAD_DYNAMICS_LONG_
CHILD_SCOPE_V1.md`). Direction, expected sign (NEGATIVE), effect-size
floor (beta<=-1.0), model (OLS+intercept+HC3, no controls) are all frozen
by that ruling and copied verbatim here -- never re-derived.

This module resolves phases 1-5 of the enforced order (family/child
identity, graveyard, exposure, outcome freeze, LONG population freeze,
split resolution/sample-sufficiency check). It reads no outcome VALUE at
any point -- `resolve_long_population` reads only `observation_status`/
`path_definition_version` (existence/coverage metadata), never
`endpoint_return_bps`/`mfe_bps`.

RESULT: the frozen 70/30 chronological split (TRAIN_FRACTION, reused
verbatim from `ami/research/w4_post_event_path_taxonomy.py`, the same
convention CVD/Absorption used) applied to the 58 outcome-compatible LONG
representatives (of 70 structural) yields TEST n=18, below the frozen
minimum of 20. Per this gate's own Phase 5 instruction ("if any minimum
is not satisfied, close INCOMPLETE... do not lower the sample rule"),
this module stops there. No experiment_registry row, no TEST nullifier
consumption, no gate receipt is created for this attempt.
"""
from __future__ import annotations

import hashlib

from ami.governance import epistemic_gates as gates

FAMILY_NAME = "FAM_BOOK_SPREAD_DYNAMICS"
PARENT_FAMILY_ID = "FAMv1:2d102e7b70820470"
PRIOR_INCOMPLETE_CHILD_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
PRIOR_INCOMPLETE_COMMIT = "a4722117"

CHILD_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1"
QUESTION_IDS = "FAM_BOOK_SPREAD_DYNAMICS"
HYPOTHESIS_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1"

FORMULA_VERSION = "BOOK_SPREAD_CHANGE_BPS_W300_V1"
SPECIFICATION_HASH_FEATURE = "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212"
ROW_ACCOUNTING_ROOT = "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31"
EXPECTED_SCHEMA_VERSION = 14

OUTCOME_ID = "endpoint_return_bps@swing_24h"
OUTCOME_HORIZON = "swing_24h"
CORRECTED_PATH_VERSION = "path-v2-candle-repair-r1"

DIRECTION = "LONG"
EXPECTED_SIGN = "NEGATIVE"
EFFECT_FLOOR_BPS = -1.0

TRAIN_FRACTION = 0.7  # reused verbatim from ami.research.w4_post_event_path_taxonomy
MIN_TRAIN_N = 30
MIN_TEST_N = 20
MIN_TOTAL_N = 50
MIN_RESIDUAL_DF = 15
DESIGN_RANK = 2  # intercept + spread_change_bps_w300

FROZEN_SPLITS_TEXT = (
    "cycle-grouped chronological 70/30 split by earliest signal_birth_ts per "
    "independent_cycle_id, cut by cycle count"
)


class LongPreregistrationIncomplete(Exception):
    """Raised when a required identity/population/split invariant fails.
    Per the gate's own Phase 5 instruction, sample insufficiency stops the
    batch here rather than lowering the frozen minimum."""


def resolve_family_and_child_identity(knowledge_conn) -> dict:
    """Phase 1-2 (identity/graveyard/exposure), scoped to the new LONG
    child. Read-only."""
    family_id = gates.resolve_canonical_family_id(QUESTION_IDS, HYPOTHESIS_ID)
    spec_text = (
        f"{QUESTION_IDS} | {HYPOTHESIS_ID} | LONG-only pre-birth L1 spread expansion, "
        "continuous incremental predictive information for endpoint_return_bps at "
        "swing_24h, no controls, expected negative sign"
    )
    graveyard = gates.match_graveyard(knowledge_conn, spec_text)
    prior_exposure = knowledge_conn.execute(
        "SELECT nullifier, consumed_by_experiment_id FROM epistemic_test_nullifiers WHERE family_id=?",
        (family_id,)).fetchall()
    existing_receipt = knowledge_conn.execute(
        "SELECT experiment_id, registry_result FROM experiment_gate_receipts WHERE canonical_family_id=?",
        (family_id,)).fetchall()
    return {
        "family_name": FAMILY_NAME, "child_id": CHILD_ID, "family_id": family_id,
        "parent_family_id": PARENT_FAMILY_ID, "distinct_from_parent": family_id != PARENT_FAMILY_ID,
        "spec_text": spec_text,
        "graveyard_hits": graveyard, "graveyard_clean": graveyard == [],
        "prior_test_exposure": prior_exposure, "prior_test_exposure_clean": prior_exposure == [],
        "existing_gate_receipts_for_family": existing_receipt,
        "genuinely_unconsumed": graveyard == [] and prior_exposure == [] and existing_receipt == [],
    }


def resolve_long_population(canonical_conn) -> dict:
    """Phase 4. Outcome-blind LONG-only population resolution. Reads
    `direction`/`is_cycle_representative`/identity columns from the
    canonical feature table, and only `observation_status`/
    `path_definition_version` (coverage metadata, never a value) from the
    path-observations table."""
    reps = canonical_conn.execute(
        "SELECT anchor_id, cycle_id, signal_birth_ts, formula_version, row_accounting_root "
        "FROM ami_book_spread_change_windowed_flow WHERE is_cycle_representative=1 AND direction=?",
        (DIRECTION,)).fetchall()
    if any(r[3] != FORMULA_VERSION or r[4] != ROW_ACCOUNTING_ROOT for r in reps):
        raise LongPreregistrationIncomplete(
            "formula_version/row_accounting_root drift from frozen constants")

    anchor_ids = [r[0] for r in reps]
    coverage: dict[str, dict] = {}
    if anchor_ids:
        placeholders = ",".join("?" for _ in anchor_ids)
        obs_rows = canonical_conn.execute(
            f"SELECT signal_id, path_definition_version, observation_status "
            f"FROM ami_lifecycle_path_observations WHERE horizon_name=? AND signal_id IN ({placeholders})",
            [OUTCOME_HORIZON, *anchor_ids]).fetchall()
        version_by_sig: dict[str, str] = {}
        for sid, version, status in obs_rows:
            if sid not in version_by_sig or version == CORRECTED_PATH_VERSION:
                coverage[sid] = {"observation_status": status, "path_definition_version": version}
                version_by_sig[sid] = version

    eligible = [(a, c, t) for a, c, t, *_ in reps if coverage.get(a, {}).get("observation_status") == "OK"]
    excluded = [(a, c, t, coverage.get(a, {}).get("observation_status")) for a, c, t, *_ in reps
                if coverage.get(a, {}).get("observation_status") != "OK"]
    dup_cycles = len([c for _, c, _ in eligible]) - len({c for _, c, _ in eligible})

    ordered = sorted(eligible, key=lambda r: r[2])
    structural_manifest = "|".join(f"{a},{c},{t}" for a, c, t, *_ in sorted(reps, key=lambda r: r[2]))
    eligible_manifest = "|".join(f"{a},{c},{t}" for a, c, t in ordered)

    return {
        "structural_representative_count": len(reps),
        "eligible_representative_count": len(eligible),
        "excluded_count": len(excluded), "excluded_detail": excluded,
        "duplicate_cycle_ids": dup_cycles, "missing_representatives": 0,
        "ordered_eligible": ordered,
        "structural_population_hash": hashlib.sha256(structural_manifest.encode()).hexdigest(),
        "eligible_population_hash": hashlib.sha256(eligible_manifest.encode()).hexdigest(),
        "matches_expected_structural_70": len(reps) == 70,
    }


def resolve_outcome_metadata() -> dict:
    """Phase 3. Reuses `endpoint_return_bps@swing_24h` verbatim -- no
    direction-flip, no derivation. Pure constant-return function (no
    connection needed; nothing beyond the fixed identity is resolved
    here, since Phase 3 forbids reading distributional metadata beyond
    what `resolve_long_population` already resolves via observation_status)."""
    return {
        "outcome_id": OUTCOME_ID, "outcome_table": "ami_lifecycle_path_observations",
        "outcome_horizon": OUTCOME_HORIZON,
        "dependent_variable_type": "continuous",
        "direction_semantics": "NOT direction-flipped (absolute price-return sign)",
        "reused_verbatim": True, "newly_derived": False, "direction_flip_applied": False,
    }


def resolve_split(population: dict) -> dict:
    """Phase 5. Chronological cycle-grouped 70/30 cut (TRAIN_FRACTION),
    reused verbatim -- not re-derived. Computes TRAIN/TEST hashes and
    checks the frozen sample-sufficiency rule. Raises nothing; the
    caller decides the batch verdict from the returned dict."""
    ordered = population["ordered_eligible"]
    n = len(ordered)
    cut = int(n * TRAIN_FRACTION)
    train = ordered[:cut]
    test = ordered[cut:]

    train_keys = sorted(c for _, c, _ in train)
    test_keys = sorted(c for _, c, _ in test)
    overlap = len(set(train_keys) & set(test_keys))
    straddle_clean = (not train or not test) or train[-1][2] < test[0][2]

    split_version = gates.resolve_split_version(FROZEN_SPLITS_TEXT)
    train_hash = hashlib.sha256(",".join(train_keys).encode()).hexdigest()
    test_hash = hashlib.sha256(",".join(test_keys).encode()).hexdigest()

    residual_df = len(test) - DESIGN_RANK
    sufficiency = {
        "train_n_ok": len(train) >= MIN_TRAIN_N,
        "test_n_ok": len(test) >= MIN_TEST_N,
        "total_n_ok": n >= MIN_TOTAL_N,
        "residual_df_ok": residual_df >= MIN_RESIDUAL_DF,
        "overlap_zero": overlap == 0,
        "no_straddling": straddle_clean,
    }
    sufficient = all(sufficiency.values())

    return {
        "split_version": split_version, "frozen_splits_text": FROZEN_SPLITS_TEXT,
        "train_n": len(train), "test_n": len(test), "total_n": n,
        "residual_df": residual_df, "overlap": overlap,
        "train_hash": train_hash, "test_hash": test_hash,
        "train_cycle_keys": train_keys, "test_cycle_keys": test_keys,
        "sufficiency": sufficiency, "sufficient": sufficient,
        "verdict": (
            "SPLIT_SUFFICIENT" if sufficient else
            "BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE"
        ),
    }
