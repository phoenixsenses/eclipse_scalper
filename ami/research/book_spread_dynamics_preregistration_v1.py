"""BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1.

Outcome-blind resolution for the first preregistration attempt of
`FAM_BOOK_SPREAD_DYNAMICS` (child `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`),
against the canonical population materialized by M-0036
(`ami_book_spread_change_windowed_flow`, 196 EXACT rows / 97 independent-
cycle representatives, row_accounting_root `33c4f4be...`).

This module performs ONLY phases 1-2 of the enforced preregistration order
(family/child identity, graveyard, prior exposure, outcome-ID resolution)
plus the Phase-3 direction/expected-sign resolution attempt. It reads no
outcome VALUE at any point (no `endpoint_return_bps`/`mfe_bps` selected
anywhere in this module) -- `resolve_outcome_metadata` reads only
`observation_status`/`path_definition_version` (existence/coverage
metadata), never the continuous return columns.

RESULT: Phase 3 cannot be resolved without violating an explicit
prohibition (see `resolve_direction_and_sign`'s docstring for the full
argument). This is a genuine, structural property of this family's frozen
population -- not a placeholder or an oversight -- so this module stops at
Phase 3 and the batch closes `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_
INCOMPLETE`. No experiment_registry row, no TEST nullifier, no gate receipt
is created for this attempt (nothing to authorize -- there is no frozen,
single-sign hypothesis to protect from later degrees of freedom).
"""
from __future__ import annotations

from ami.governance import epistemic_gates as gates

FAMILY_NAME = "FAM_BOOK_SPREAD_DYNAMICS"
CHILD_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
FORMULA_VERSION = "BOOK_SPREAD_CHANGE_BPS_W300_V1"
SPECIFICATION_HASH = "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212"
ROW_ACCOUNTING_ROOT = "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31"
EXPECTED_SCHEMA_VERSION = 14
EXPECTED_FEATURE_ROWS = 196
EXPECTED_REPRESENTATIVE_ROWS = 97

QUESTION_IDS = "FAM_BOOK_SPREAD_DYNAMICS"
HYPOTHESIS_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-V1-DIRECTION-NEUTRAL"

OUTCOME_ID = "endpoint_return_bps@swing_24h"
OUTCOME_HORIZON = "swing_24h"


class PreregistrationIncomplete(Exception):
    """Raised by `resolve_direction_and_sign` -- Phase 3 cannot be closed
    without either an unauthorized population restriction, a derived
    (direction-flipped) outcome, or a forbidden interaction term. Not a
    bug: the frozen population is genuinely direction-mixed and no earlier
    accepted gate (readiness/rehearsal/freeze/migration) scoped it to one
    direction, so scoping it now would be a new, previously-unfrozen
    researcher degree of freedom introduced immediately before modeling."""


def resolve_family_and_child_identity(knowledge_conn) -> dict:
    """Phase 1. Read-only. Resolves the canonical family_id, runs the
    graveyard matcher against this family's frozen spec text, and checks
    for any prior TEST-evidence exposure under that exact family_id."""
    family_id = gates.resolve_canonical_family_id(QUESTION_IDS, HYPOTHESIS_ID)
    spec_text = (
        f"{QUESTION_IDS} | {HYPOTHESIS_ID} | "
        "pre-birth L1 spread expansion relative to its own pre-window baseline, "
        "continuous incremental predictive information for endpoint_return_bps at "
        "swing_24h, controlling for the same frozen control set, direction-neutral "
        "mixed LONG/SHORT population"
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
        "spec_text": spec_text,
        "graveyard_hits": graveyard, "graveyard_clean": graveyard == [],
        "prior_test_exposure": prior_exposure, "prior_test_exposure_clean": prior_exposure == [],
        "existing_gate_receipts_for_family": existing_receipt,
        "genuinely_unconsumed": graveyard == [] and prior_exposure == [] and existing_receipt == [],
    }


def resolve_population(canonical_conn) -> dict:
    """Outcome-blind population resolution against the M-0036 canonical
    tables. Reads only immutable identity/quality columns (anchor_id,
    direction, cycle_id, is_cycle_representative, formula_version,
    row_accounting_root) -- never a feature value or outcome column."""
    rows = canonical_conn.execute(
        "SELECT anchor_id, direction, cycle_id, is_cycle_representative, formula_version, "
        "row_accounting_root FROM ami_book_spread_change_windowed_flow").fetchall()
    if any(r[4] != FORMULA_VERSION or r[5] != ROW_ACCOUNTING_ROOT for r in rows):
        raise PreregistrationIncomplete(
            "formula_version/row_accounting_root drift from frozen constants")
    reps = [r for r in rows if r[3] == 1]
    from collections import Counter
    direction_all = dict(Counter(r[1] for r in rows))
    direction_reps = dict(Counter(r[1] for r in reps))
    return {
        "feature_row_count": len(rows), "representative_count": len(reps),
        "direction_breakdown_all": direction_all,
        "direction_breakdown_representatives": direction_reps,
        "is_direction_mixed": len(direction_reps) > 1,
        "matches_frozen_196_97": len(rows) == EXPECTED_FEATURE_ROWS and len(reps) == EXPECTED_REPRESENTATIVE_ROWS,
    }


def resolve_outcome_metadata(canonical_conn, representative_anchor_ids: list[str]) -> dict:
    """Phase 2. Reuses the existing canonical outcome `endpoint_return_bps@
    swing_24h` verbatim (same identity as the CVD and Absorption
    preregistrations) -- no new outcome is proposed or derived. Reads only
    `observation_status`/`path_definition_version` (coverage metadata);
    `endpoint_return_bps`/`mfe_bps` are never selected by this function."""
    if not representative_anchor_ids:
        return {"outcome_id": OUTCOME_ID, "coverage": {}}
    placeholders = ",".join("?" for _ in representative_anchor_ids)
    coverage = dict(canonical_conn.execute(
        f"SELECT observation_status, COUNT(*) FROM ami_lifecycle_path_observations "
        f"WHERE horizon_name=? AND signal_id IN ({placeholders}) GROUP BY observation_status",
        [OUTCOME_HORIZON, *representative_anchor_ids]).fetchall())
    return {
        "outcome_id": OUTCOME_ID, "outcome_table": "ami_lifecycle_path_observations",
        "outcome_horizon": OUTCOME_HORIZON,
        "dependent_variable_type": "continuous",
        "direction_semantics": "NOT direction-flipped (absolute price-return sign) -- established "
                                "convention, identical to the CVD and Absorption preregistrations",
        "reused_verbatim": True, "newly_derived": False,
        "structurally_compatible": coverage.get("OK", 0) > 0,
        "coverage": coverage,
    }


def resolve_direction_and_sign(population: dict) -> dict:
    """Phase 3. Attempts to freeze ONE expected coefficient sign for the
    (spread_change_bps_w300 -> endpoint_return_bps) association across the
    frozen, direction-mixed population, without result inspection.

    Three paths were considered and all three are blocked:

    1. RESTRICT the population to one direction (as both the CVD and
       Absorption preregistrations did). Blocked: no earlier accepted gate
       for this family (readiness/rehearsal/freeze/migration) scoped the
       population by direction -- the row-accounting freeze's own
       representative rule explicitly "uses no outcome/feature/direction/
       subgroup/route/bucket signal", and the family's own hypothesis text
       ("Does pre-birth L1 spread expansion ... contain continuous
       incremental predictive information for endpoint_return_bps@
       swing_24h, controlling for the same frozen control set?") never
       mentions a direction restriction. Introducing one now, for the
       first time, immediately before modeling, would be a new researcher
       degree of freedom -- exactly what this phase exists to prevent.
    2. FLIP the outcome by signal direction (endpoint_return_bps for
       LONG, -endpoint_return_bps for SHORT) to make a pooled sign
       coherent. Blocked: this derives a NEW outcome specific to this
       family, explicitly forbidden ("Do not invent a new outcome. Do not
       derive an outcome specifically for this family.").
    3. ADD a direction x predictor interaction term (or, equivalently,
       fit two separate per-direction models -- the repository's own
       established convention for direction-mixed populations, see
       `ami/research/w8_hold_baseline.py`'s per-direction cell design).
       Blocked: interactions and subgroup/per-direction rescue are both
       explicitly forbidden by the model-freeze and TEST-policy phases
       ("Do not create interactions"; "no subgroup rescue").

    Absent all three, no single physically-defensible sign for a pooled,
    additive, non-interacting model exists: spread widening's plausible
    mechanisms (liquidity-withdrawal amplification of a subsequent move,
    or reversal-amplification of the cascade's own base rate) both imply
    OPPOSITE algebraic signs on raw, unflipped `endpoint_return_bps`
    depending on whether the anchor is LONG (down-cascade) or SHORT
    (up-cascade) -- an interaction, not an additive effect. Choosing a
    sign anyway would be arbitrary, and choosing it after inspecting TEST
    (or even TRAIN) would violate the no-result-inspection requirement.

    Returns a dict recording the blocked resolution; raises nothing itself
    (the caller decides the batch verdict)."""
    if not population.get("is_direction_mixed"):
        return {"resolved": True, "reason": "population is not direction-mixed; not applicable here"}
    return {
        "resolved": False,
        "direction_breakdown_representatives": population["direction_breakdown_representatives"],
        "path_1_restrict_population_blocked": (
            "no earlier accepted gate scoped this family by direction; the frozen "
            "row-accounting freeze's representative rule uses no direction signal; the "
            "family's own hypothesis text specifies no direction restriction"
        ),
        "path_2_flip_outcome_blocked": (
            "would derive a new, family-specific outcome; explicitly forbidden"
        ),
        "path_3_interaction_or_subgroup_blocked": (
            "interactions are explicitly forbidden by the model-freeze phase; per-direction "
            "subgroup fitting is explicitly forbidden as subgroup rescue"
        ),
        "verdict": "BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE",
    }
